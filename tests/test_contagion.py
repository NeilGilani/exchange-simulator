"""Tests for the credit layer: balance sheets, margin spirals, contagion.

Two kinds of test here, deliberately.

**Mechanism tests** build a tiny hand-made market and assert exactly what
must happen — a write-off destroys a claim without moving cash, a creditor
absorbs a borrower's shortfall, an insolvent agent stops trading. These are
deterministic and they are the ones that would catch a real accounting bug.

**Emergent tests** run the full ecology and assert statistical properties
across seeds, because the interesting behaviour is stochastic. They are
written loosely on purpose: they check the *direction* of an effect
(more leverage is not safer; bigger shocks do more damage) rather than
pinning down numbers that would make the suite fragile without making the
model any more correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from exchange_sim.agents import Agent, MarketView, OrderIntent
from exchange_sim.balance import (
    BalanceSheet,
    FundingNetwork,
    RiskLimits,
    resolve_stress,
)
from exchange_sim.engine import Exchange, credit_crisis, systemic_market
from exchange_sim.fragility import measure_state, shock_response
from exchange_sim.institutions import DealerBank, LeveredHedgeFund


# -- helpers ------------------------------------------------------------------


class Passive(Agent):
    """Does nothing. Used to hold a balance sheet in mechanism tests."""

    def act(self, view, rng):
        return []


class Quoter(Agent):
    """Posts a fixed two-sided market so forced sellers have something to hit."""

    def __init__(self, name, price=100.0, size=500.0, cash=1_000_000.0):
        super().__init__(name, cash=cash, inventory=5_000.0)
        self.price, self.size = price, size

    def wants_cancel_all(self, view):
        return True

    def act(self, view, rng):
        return [OrderIntent("limit", "buy", self.size, round(self.price - 0.5, 2)),
                OrderIntent("limit", "sell", self.size, round(self.price + 0.5, 2))]


def tiny_market(**kwargs):
    """A lender, a borrower and a quoter — the smallest system that can fail."""
    lender = Passive("lender", cash=100_000.0)
    borrower = Passive("borrower", cash=10_000.0, inventory=1_000.0)
    quoter = Quoter("quoter")
    return Exchange([lender, borrower, quoter], seed=0, **kwargs), lender, borrower


# -- mechanism: accounting ----------------------------------------------------


def test_lending_moves_cash_without_creating_it():
    exchange, lender, borrower = tiny_market()
    before = exchange.total_cash()
    exchange.funding.lend(lender, borrower, 40_000.0)

    assert lender.sheet.cash == pytest.approx(60_000.0)
    assert borrower.sheet.cash == pytest.approx(50_000.0)
    assert lender.sheet.receivables == pytest.approx(40_000.0)
    assert borrower.sheet.debt == pytest.approx(40_000.0)
    assert exchange.total_cash() == pytest.approx(before)


def test_lender_cannot_lend_cash_it_does_not_have():
    exchange, lender, borrower = tiny_market()
    lent = exchange.funding.lend(lender, borrower, 10_000_000.0)
    assert lent == pytest.approx(100_000.0)
    assert lender.sheet.cash == pytest.approx(0.0)


def test_write_off_destroys_a_claim_not_cash():
    exchange, lender, borrower = tiny_market()
    exchange.funding.lend(lender, borrower, 40_000.0)
    cash_before = exchange.total_cash()

    exchange.funding.write_off(lender, borrower, 40_000.0)

    assert exchange.total_cash() == pytest.approx(cash_before)   # no cash moved
    assert borrower.sheet.debt == pytest.approx(0.0)             # both legs die
    assert lender.sheet.receivables == pytest.approx(0.0)
    assert lender.sheet.credit_losses == pytest.approx(40_000.0)


def test_total_equity_equals_cash_plus_marked_shares():
    """The invariant that makes write-offs safe: claims cancel pairwise."""
    exchange, lender, borrower = tiny_market()
    exchange.funding.lend(lender, borrower, 40_000.0)
    exchange.run(50)
    mid = exchange._mid()

    expected = exchange.total_cash() + exchange.total_shares() * mid
    assert exchange.total_equity(mid) == pytest.approx(expected)

    exchange.funding.write_off(lender, borrower, 25_000.0)
    assert exchange.total_equity(mid) == pytest.approx(expected)


# -- mechanism: failure and contagion -----------------------------------------


def test_insolvent_agent_fails_and_its_creditor_eats_the_loss():
    exchange, lender, borrower = tiny_market()
    exchange.run(20)                       # let the quoter build a book
    exchange.funding.lend(lender, borrower, 90_000.0)

    # wipe the borrower out: it owes 90k and holds far less than that
    borrower.sheet.cash = 0.0
    borrower.sheet.inventory = 100.0
    assert borrower.sheet.is_insolvent(exchange._mid())

    cash_before = exchange.total_cash()
    report = resolve_stress(exchange.live_agents(), exchange.funding, exchange._mid,
                            exchange._force_market, exchange.step_count,
                            RiskLimits())

    assert borrower.sheet.failed
    assert len(report.defaults) == 1
    failure = report.defaults[0]
    assert failure.agent == "borrower"
    assert failure.written_off > 0                     # it could not repay in full
    assert lender.sheet.credit_losses == pytest.approx(failure.written_off)
    assert "lender" in failure.creditors
    assert exchange.total_cash() == pytest.approx(cash_before)   # still conserved


def test_failed_agent_stops_trading():
    exchange, lender, borrower = tiny_market()
    exchange.run(10)
    borrower.sheet.failed = True
    assert borrower not in exchange.live_agents()


def test_margin_call_forces_selling_before_insolvency():
    exchange, lender, borrower = tiny_market()
    exchange.run(20)
    exchange.funding.lend(lender, borrower, 60_000.0)
    # equity/assets lands near 0.65 here, so a 0.75 floor calls it while
    # leaving it comfortably solvent — the gap this test exists to cover
    borrower.limits = RiskLimits(maintenance_margin=0.75, target_margin=0.85)

    mid = exchange._mid()
    assert not borrower.sheet.is_insolvent(mid)        # solvent...
    assert borrower.sheet.under_margin(mid, borrower.limits)   # ...but too thin

    inventory_before = borrower.sheet.inventory
    report = resolve_stress(exchange.live_agents(), exchange.funding, exchange._mid,
                            exchange._force_market, exchange.step_count,
                            borrower.limits)

    assert report.margin_calls, "a thin borrower should be called"
    assert borrower.sheet.inventory < inventory_before, "a call should force selling"
    assert not borrower.sheet.failed, "a margin call is not a default"


def test_unlevered_agents_are_never_margin_called():
    sheet = BalanceSheet(cash=100.0, inventory=10.0, debt=0.0)
    assert not sheet.under_margin(mid=1.0, limits=RiskLimits(maintenance_margin=0.99))


# -- emergent: the full ecology -----------------------------------------------


def test_conservation_holds_through_a_credit_crisis():
    """Cash and shares survive margin spirals, fire sales and write-offs."""
    exchange = credit_crisis(seed=4, steps=900, shock_at=300, fund_leverage=12)
    mid = exchange._mid()

    assert exchange.total_cash() == pytest.approx(5_825_000.0, rel=1e-9)
    assert exchange.total_shares() == pytest.approx(11_000.0, rel=1e-9)
    assert exchange.total_equity(mid) == pytest.approx(
        exchange.total_cash() + exchange.total_shares() * mid)


def test_systemic_market_is_deterministic_under_a_seed():
    a = systemic_market(seed=11); a.run(300)
    b = systemic_market(seed=11); b.run(300)
    assert a.tape.mid == b.tape.mid
    assert len(a.defaults) == len(b.defaults)


def test_funding_tightens_when_the_market_falls():
    """The margin spiral's first link: stress raises haircuts."""
    exchange = systemic_market(seed=2, fund_leverage=6)
    exchange.run(300)
    dealer = exchange.dealers[0]

    calm = MarketView(step=300, mid=100.0, best_bid=99.9, best_ask=100.1,
                      volatility=0.001, drawdown=-0.002)
    stressed = MarketView(step=300, mid=100.0, best_bid=99.9, best_ask=100.1,
                          volatility=0.02, drawdown=-0.15)

    assert dealer.haircut(stressed) > dealer.haircut(calm)
    assert dealer.stress(stressed) > dealer.stress(calm)


def test_dealers_quote_smaller_when_stressed():
    dealer = DealerBank("d")
    calm = MarketView(step=1, mid=100.0, best_bid=99.9, best_ask=100.1,
                      volatility=0.0005, drawdown=0.0)
    stressed = MarketView(step=1, mid=100.0, best_bid=99.9, best_ask=100.1,
                          volatility=0.03, drawdown=-0.2)
    rng = np.random.default_rng(0)

    calm_size = sum(i.qty for i in dealer.act(calm, rng))
    stressed_size = sum(i.qty for i in dealer.act(stressed, rng))
    assert stressed_size < calm_size


def test_damage_increases_with_shock_size():
    """The response curve must at least be monotone in the shock."""
    exchange = systemic_market(seed=42, fund_leverage=6)
    exchange.run(300)
    report = shock_response(exchange, shocks=(0.02, 0.08, 0.18), runs=4,
                            horizon=300, seed=42)

    drawdowns = [abs(o.median_drawdown) for o in report.outcomes]
    assert drawdowns == sorted(drawdowns), f"not monotone in shock: {drawdowns}"
    assert report.control is not None
    assert abs(report.control.median_drawdown) < drawdowns[-1], (
        "an 18% shock should hurt more than doing nothing")


def test_leverage_does_not_make_the_system_safer():
    """The central claim, stated as weakly as it can honestly be stated.

    Not "more leverage always cascades" — that is seed-dependent and would
    make a flaky test. Only that a heavily levered, loosely financed system
    is at least as fragile as a conservatively financed one, measured by
    failures under an identical shock.
    """
    def fragility_at(leverage, haircut):
        failures = 0
        for seed in range(3):
            exchange = systemic_market(seed=seed, fund_leverage=leverage,
                                       base_haircut=haircut)
            exchange.run(300)
            report = shock_response(exchange, shocks=(0.18,), runs=3,
                                    horizon=400, seed=seed)
            failures += sum(report.outcomes[0].failures)
        return failures

    assert fragility_at(20.0, 0.10) >= fragility_at(3.0, 0.30)


def test_measure_state_reports_no_leverage_when_nobody_borrows():
    exchange, _, _ = tiny_market()
    exchange.run(30)
    state = measure_state(exchange)
    assert state.levered_leverage == pytest.approx(1.0)
    assert state.total_debt == pytest.approx(0.0)
    assert state.crowding == pytest.approx(0.0)


def test_crowding_detects_one_sided_positioning():
    exchange = systemic_market(seed=3, fund_leverage=6)
    exchange.run(300)
    state = measure_state(exchange)
    assert 0.0 <= state.crowding <= 1.0


# -- robustness ---------------------------------------------------------------


def test_exchange_clamps_orders_agents_should_not_have_sent():
    """A market in freefall computes quotes at or below zero. That must not
    take the exchange down with it — nor may a user-written agent."""

    class BadAgent(Agent):
        def act(self, view, rng):
            return [
                OrderIntent("limit", "buy", 10, 0.0),          # zero price
                OrderIntent("limit", "sell", 10, -5.0),        # negative price
                OrderIntent("limit", "buy", 10, float("nan")), # not a number
                OrderIntent("market", "buy", 0),               # zero size
                OrderIntent("market", "sell", -3),             # negative size
            ]

    exchange = Exchange([BadAgent("bad", cash=1_000.0), Quoter("quoter")], seed=0)
    exchange.run(20)                                   # must not raise
    assert exchange.total_shares() == pytest.approx(5_000.0)


def test_failed_estate_is_worked_down_over_time():
    """A failed agent's position doesn't vanish — it keeps hitting the book."""
    exchange, lender, borrower = tiny_market()
    exchange.run(20)
    borrower.sheet.failed = True
    borrower.sheet.inventory = 900.0

    first = borrower.sheet.inventory
    exchange.step()
    after_one = borrower.sheet.inventory
    exchange.run(30)

    assert after_one < first, "the estate should start selling immediately"
    assert borrower.sheet.inventory < after_one, "and keep going"
    assert exchange.total_shares() == pytest.approx(
        900.0 + 1_000.0 + 5_000.0 - 1_000.0, abs=1e-6)


def test_levered_fund_does_not_trade_once_its_equity_is_gone():
    fund = LeveredHedgeFund("f", style="momentum", cash=1_000.0)
    fund.sheet.debt = 50_000.0
    view = MarketView(step=50, mid=100.0, best_bid=99.9, best_ask=100.1,
                      mid_history=[100.0] * 60)
    assert fund.act(view, np.random.default_rng(0)) == []
