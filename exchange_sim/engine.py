"""The exchange: agents submit orders to a real matching engine, trades are
settled agent-to-agent, funding is extended and withdrawn, and agents that
run out of equity fail — taking their creditors with them.

Each step runs in this order, and the order is the model:

1. **Fundamental moves.** Either a slow random walk, or a real historical
   series when one is supplied (see :mod:`exchange_sim.history`).
2. **The funding round.** Dealers read public stress and set a haircut;
   that haircut becomes every borrower's maintenance margin. Loans move
   toward each borrower's target leverage.
3. **Agents act.** Everyone sees the same public view and submits orders.
4. **Stress resolution.** Margin calls force selling; insolvent agents are
   liquidated and their shortfall is written off against creditors, who may
   fail in turn. Runs to a fixed point, so cascades are chains, not events.
5. **Record.** Price, spread, volume, leverage, haircut, failures.

Settlement is exact double-entry: every trade moves cash one way and shares
the other, and lending moves cash without creating it. Total cash and total
shares across all agents are invariants — through margin spirals, defaults
and write-offs alike. If the simulator ever "prints" money or shares,
that's a bug, not a market.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from quantsim import OrderBook

from exchange_sim.agents import (
    Agent,
    FundamentalTrader,
    Liquidator,
    MarketMaker,
    MarketView,
    MomentumTrader,
    NoiseTrader,
)
from exchange_sim.balance import (
    DefaultEvent,
    FundingNetwork,
    MarginCall,
    RiskLimits,
    resolve_stress,
)
from exchange_sim.institutions import (
    DealerBank,
    FundingDesk,
    HFTMarketMaker,
    LeveredHedgeFund,
    PensionFund,
    RetailTrader,
)

#: agents that the engine feeds a fundamental value to each step
_FUNDAMENTAL_AWARE = (FundamentalTrader, PensionFund, LeveredHedgeFund)
#: agents that provide liquidity, and can therefore withdraw it
_LIQUIDITY_PROVIDERS = (MarketMaker, FundamentalTrader, DealerBank)


@dataclass
class Tape:
    """Everything the market produced, step by step."""
    mid: list[float] = field(default_factory=list)
    spread: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    fundamental: list[float] = field(default_factory=list)
    #: system-level state, recorded so fragility can be measured after the fact
    leverage: list[float] = field(default_factory=list)
    haircut: list[float] = field(default_factory=list)
    failures: list[int] = field(default_factory=list)
    forced_volume: list[float] = field(default_factory=list)
    book_depth: list[float] = field(default_factory=list)

    def returns(self) -> np.ndarray:
        mids = np.asarray(self.mid)
        if mids.size < 2:
            return np.zeros(0)
        return mids[1:] / mids[:-1] - 1.0

    def __len__(self) -> int:
        return len(self.mid)


class Exchange:
    def __init__(self, agents: list[Agent], start_price: float = 100.0,
                 fundamental_vol: float = 0.0004, seed: int | None = None,
                 limits: RiskLimits | None = None,
                 fundamental_path: Sequence[float] | None = None,
                 vol_window: int = 30, drawdown_window: int = 250):
        self.book = OrderBook()
        self.agents = {agent.name: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("agent names must be unique")
        for agent in self.agents.values():
            agent._peers = self.agents        # creditors must be reachable
        self.rng = np.random.default_rng(seed)
        self.limits = limits or RiskLimits()
        self.fundamental_path = (list(fundamental_path)
                                 if fundamental_path is not None else None)
        self.fundamental = (self.fundamental_path[0] if self.fundamental_path
                            else start_price)
        self.fundamental_vol = fundamental_vol
        self.last_mid = start_price
        self.last_trade_price: float | None = None
        self.tape = Tape()
        self.step_count = 0
        self._order_counter = 0
        self.vol_window = vol_window
        self.drawdown_window = drawdown_window

        # -- credit layer -----------------------------------------------------
        self.funding = FundingNetwork()
        self.desk = FundingDesk(self.funding)
        self.defaults: list[DefaultEvent] = []
        self.margin_calls: list[MarginCall] = []
        self._last_haircut = float("nan")

    # -- roster ---------------------------------------------------------------

    @property
    def dealers(self) -> list[DealerBank]:
        return [a for a in self.agents.values() if isinstance(a, DealerBank)]

    @property
    def borrowers(self) -> list[Agent]:
        """Anyone who wants to run levered — i.e. wants more assets than equity."""
        return [a for a in self.agents.values()
                if isinstance(a, LeveredHedgeFund) and not a.sheet.failed]

    def live_agents(self) -> list[Agent]:
        return [a for a in self.agents.values() if not a.sheet.failed]

    # -- invariants -----------------------------------------------------------

    def total_cash(self) -> float:
        return sum(a.sheet.cash for a in self.agents.values())

    def total_shares(self) -> float:
        return sum(a.sheet.inventory for a in self.agents.values())

    def total_equity(self, mid: float | None = None) -> float:
        """Sum of everyone's net worth.

        Must always equal ``total_cash + total_shares * mid``: debt and
        receivables cancel pairwise, and a write-off kills both legs at once.
        """
        mid = self._mid() if mid is None else mid
        return sum(a.sheet.equity(mid) for a in self.agents.values())

    def total_debt(self) -> float:
        return sum(a.sheet.debt for a in self.agents.values())

    def aggregate_leverage(self, mid: float | None = None) -> float:
        """System-wide assets/equity across solvent agents."""
        mid = self._mid() if mid is None else mid
        live = self.live_agents()
        equity = sum(a.sheet.equity(mid) for a in live)
        assets = sum(a.sheet.gross_assets(mid) for a in live)
        if equity <= 0:
            return float("inf")
        return assets / equity

    # -- market state ---------------------------------------------------------

    def _mid(self) -> float:
        """Best estimate of the current price.

        Two-sided book: the quote midpoint. One-sided or empty book (e.g.
        mid-crash, when the bid side is swept clean): the last trade price —
        the honest number; falling back to the surviving ask would hide the
        crash the way stale quotes hid the real one in 2010.
        """
        bid, ask = self.book.best_bid(), self.book.best_ask()
        if bid is not None and ask is not None:
            self.last_mid = (bid + ask) / 2
        elif self.last_trade_price is not None:
            self.last_mid = self.last_trade_price
        return self.last_mid

    def _volatility(self) -> float:
        mids = self.tape.mid[-(self.vol_window + 1):]
        if len(mids) < 3:
            return 0.0
        arr = np.asarray(mids)
        return float(np.std(arr[1:] / arr[:-1] - 1.0))

    def _drawdown(self) -> float:
        mids = self.tape.mid[-self.drawdown_window:]
        if not mids:
            return 0.0
        peak = max(mids)
        if peak <= 0:
            return 0.0
        return mids[-1] / peak - 1.0

    def book_depth(self, levels: int = 5) -> float:
        """Shares resting within the top ``levels`` prices, both sides.

        The single most informative number about whether the market can
        absorb a shock — and the one that collapses first when it can't.
        """
        snap = self.book.depth(levels)
        return float(sum(lvl.qty for lvl in snap["bids"])
                     + sum(lvl.qty for lvl in snap["asks"]))

    def view(self) -> MarketView:
        return MarketView(
            step=self.step_count,
            mid=self._mid(),
            best_bid=self.book.best_bid(),
            best_ask=self.book.best_ask(),
            mid_history=self.tape.mid,
            volatility=self._volatility(),
            drawdown=self._drawdown(),
            failures=len(self.defaults),
        )

    # -- order flow -----------------------------------------------------------

    def _settle(self, trades) -> float:
        """Move cash and shares between maker and taker for every fill."""
        volume = 0.0
        for trade in trades:
            maker = self.agents[trade.maker_id.split("#")[0]]
            taker = self.agents[trade.taker_id.split("#")[0]]
            buyer, seller = (taker, maker) if trade.taker_side == "buy" else (maker, taker)
            notional = trade.qty * trade.price
            buyer.sheet.cash -= notional
            buyer.sheet.inventory += trade.qty
            seller.sheet.cash += notional
            seller.sheet.inventory -= trade.qty
            volume += trade.qty
            self.last_trade_price = trade.price
        return volume

    #: minimum tradeable price. A limit below this is clamped, not rejected —
    #: a market in freefall should still be quotable, just not at zero.
    tick = 0.01
    #: a single forced order may consume at most this multiple of resting depth
    forced_depth_multiple = 1.5
    #: ...but always at least this many shares, so liquidation can't stall
    min_forced_clip = 25.0

    def _submit(self, agent: Agent, intent) -> float:
        """Submit one order, clamping anything the book would refuse.

        Agents are allowed to be wrong; the exchange is not allowed to
        crash. In a deep enough collapse an agent will compute a quote at
        or below zero, and a user-written ``act()`` can return anything at
        all — so sizes and prices are validated here rather than trusting
        every agent to do it.
        """
        qty = float(intent.qty)
        if not np.isfinite(qty) or qty <= 0:
            return 0.0
        self._order_counter += 1
        order_id = f"{agent.name}#{self._order_counter}"
        if intent.kind == "market":
            trades = self.book.market(intent.side, qty, id=order_id)
        else:
            price = float(intent.price) if intent.price is not None else 0.0
            if not np.isfinite(price):
                return 0.0
            price = max(round(price, 2), self.tick)
            trades = self.book.limit(intent.side, price, qty, id=order_id)
            agent.open_order_ids.append(order_id)
        return self._settle(trades)

    def _available_depth(self, side: str, levels: int = 10) -> float:
        """Shares resting on the side a market order would consume."""
        snap = self.book.depth(levels)
        book_side = snap["bids"] if side == "sell" else snap["asks"]
        return float(sum(level.qty for level in book_side))

    def _force_market(self, agent: Agent, side: str, qty: float):
        """Execute a forced order (margin call or liquidation) at market.

        Returns ``(filled, notional)``. The fill goes through the real book
        at real prices — a fire sale into a thin book gets a terrible one,
        which is exactly the feedback the model exists to show.

        The order is capped at a multiple of resting depth. Without the cap
        a failed agent dumps its entire book into one print and sweeps the
        price to the tick floor, which is an artefact of instantaneous
        liquidation rather than a market outcome: real estates are worked
        down over hours or days, and the residue here is worked down over
        subsequent steps by :meth:`_work_down_estates`. The cap is generous
        enough that fire sales still hurt — it prevents -100% prints, not
        the fire sale itself.
        """
        qty = float(qty)
        if not np.isfinite(qty) or qty <= 0:
            return 0.0, 0.0
        depth = self._available_depth(side)
        cap = max(self.min_forced_clip, depth * self.forced_depth_multiple)
        qty = min(qty, cap)
        self._order_counter += 1
        order_id = f"{agent.name}#{self._order_counter}"
        trades = self.book.market(side, qty, id=order_id)
        filled = sum(t.qty for t in trades)
        notional = sum(t.qty * t.price for t in trades)
        self._settle(trades)
        return filled, notional

    def _work_down_estates(self) -> float:
        """Sell off what failed agents still hold, a clip at a time.

        A failed agent's position doesn't vanish with it. Because
        :meth:`_force_market` caps any single order at a multiple of resting
        depth, a large estate survives its own liquidation and keeps
        pressing on the book for several steps afterwards — which is what a
        real unwind looks like, and why the damage outlives the failure.
        """
        volume = 0.0
        for agent in self.agents.values():
            if not agent.sheet.failed or agent.sheet.inventory == 0:
                continue
            side = "sell" if agent.sheet.inventory > 0 else "buy"
            filled, _ = self._force_market(agent, side, abs(agent.sheet.inventory))
            volume += filled
        return volume

    def _cancel_all(self, agent: Agent) -> None:
        for order_id in agent.open_order_ids:
            self.book.cancel(order_id)
        agent.open_order_ids.clear()

    # -- one market step ------------------------------------------------------

    def step(self) -> None:
        self.step_count += 1

        # 1. fundamental: historical series if given, else a slow random walk
        if self.fundamental_path is not None:
            idx = min(self.step_count, len(self.fundamental_path) - 1)
            self.fundamental = self.fundamental_path[idx]
        else:
            self.fundamental *= float(np.exp(self.rng.normal(0.0, self.fundamental_vol)))

        view = self.view()

        # 2. funding round — dealer risk appetite becomes everyone's margin rule
        dealers, borrowers = self.dealers, self.borrowers
        if dealers and borrowers:
            self._last_haircut = self.desk.run_round(dealers, borrowers, view)

        # 3. agents act
        volume = 0.0
        order = self.live_agents()
        self.rng.shuffle(order)
        for agent in order:
            if agent.sheet.failed:
                continue
            if isinstance(agent, _FUNDAMENTAL_AWARE):
                agent.fundamental = self.fundamental
            if agent.wants_cancel_all(view):
                self._cancel_all(agent)
            for intent in agent.act(view, self.rng):
                volume += self._submit(agent, intent)

        # 4. stress resolution — margin calls, defaults, cascades
        report = resolve_stress(self.live_agents(), self.funding, self._mid,
                                self._force_market, self.step_count, self.limits)
        for failure in report.defaults:
            self._cancel_all(self.agents[failure.agent])
        self.defaults.extend(report.defaults)
        self.margin_calls.extend(report.margin_calls)
        volume += report.forced_volume
        estate_volume = self._work_down_estates()
        volume += estate_volume

        # 5. record
        mid = self._mid()
        spread = self.book.spread()
        self.tape.mid.append(mid)
        self.tape.spread.append(spread if spread is not None else float("nan"))
        self.tape.volume.append(volume)
        self.tape.fundamental.append(self.fundamental)
        self.tape.leverage.append(self.aggregate_leverage(mid))
        self.tape.haircut.append(self._last_haircut)
        self.tape.failures.append(len(self.defaults))
        self.tape.forced_volume.append(report.forced_volume + estate_volume)
        self.tape.book_depth.append(self.book_depth())

    def run(self, steps: int) -> Tape:
        for _ in range(steps):
            self.step()
        return self.tape

    # -- scenario hooks -------------------------------------------------------

    def withdraw_liquidity(self, until_step: int) -> None:
        """Every liquidity provider steps away until ``until_step``."""
        for agent in self.agents.values():
            if isinstance(agent, _LIQUIDITY_PROVIDERS):
                agent.offline_until = until_step


# -- market builders ----------------------------------------------------------


def default_market(seed: int | None = None, n_noise: int = 12, n_momentum: int = 4,
                   n_fundamental: int = 3, start_price: float = 100.0,
                   **kwargs) -> Exchange:
    """The original ecosystem: 2 market makers + noise + momentum + anchors.

    No leverage, no credit, nobody can fail. Kept unchanged because the
    stylized-fact tests are calibrated against it — excess kurtosis in the
    2-15 range, clear volatility clustering, small raw autocorrelation.
    """
    agents: list[Agent] = [
        MarketMaker("mm1", inventory_skew=0.008),
        MarketMaker("mm2", half_spread=0.07, size=30, inventory_skew=0.008),
    ]
    agents += [NoiseTrader(f"noise{i}", market_frac=0.75) for i in range(n_noise)]
    agents += [MomentumTrader(f"momo{i}", lookback=8 + 4 * i, threshold=0.003, size=5)
               for i in range(n_momentum)]
    agents += [FundamentalTrader(f"fund{i}", size=30) for i in range(n_fundamental)]
    return Exchange(agents, start_price=start_price, seed=seed, **kwargs)


def systemic_market(seed: int | None = None, n_dealers: int = 2, n_hft: int = 3,
                    n_momentum_funds: int = 3, n_value_funds: int = 2,
                    n_real_money: int = 3, n_retail: int = 10, n_noise: int = 6,
                    fund_leverage: float = 4.0, base_haircut: float = 0.18,
                    start_price: float = 100.0, **kwargs) -> Exchange:
    """The full institutional stack: dealers, HFT, levered funds, real-money
    accounts, a pension, retail — with a funding network underneath.

    Unlike :func:`default_market`, agents here can borrow, be margin-called,
    and fail. ``fund_leverage`` and ``base_haircut`` are the two dials that
    matter: together they set how much of the system's risk sits on borrowed
    money, and how fast that money can be pulled.

    Sizes are calibrated so that a fund's typical clip is a fraction of
    resting depth in calm conditions and a large multiple of it once HFT
    withdraws. That ratio — not any individual parameter — is what decides
    whether a shock absorbs or cascades.
    """
    agents: list[Agent] = [
        DealerBank(f"dealer{i}", base_haircut=base_haircut,
                   half_spread=0.06 + 0.01 * i, size=300, max_inventory=6_000,
                   cash=1_500_000.0, lending_capacity=250_000.0)
        for i in range(n_dealers)
    ]
    agents += [HFTMarketMaker(f"hft{i}", half_spread=0.02 + 0.005 * i, size=150,
                              max_inventory=600, cash=200_000.0)
               for i in range(n_hft)]
    agents += [LeveredHedgeFund(f"momofund{i}", style="momentum",
                                target_leverage=fund_leverage, cash=15_000.0,
                                lookback=8 + 3 * i, threshold=0.002,
                                aggression=0.08, max_position_value=80_000.0)
               for i in range(n_momentum_funds)]
    agents += [LeveredHedgeFund(f"valuefund{i}", style="value",
                                target_leverage=fund_leverage * 0.75, cash=15_000.0,
                                band=0.004 + 0.001 * i, aggression=0.08,
                                max_position_value=80_000.0)
               for i in range(n_value_funds)]
    # real-money accounts: unlevered, price-sensitive, the market's anchor
    agents += [FundamentalTrader(f"realmoney{i}", size=120, cash=150_000.0,
                                 inventory=600.0)
               for i in range(n_real_money)]
    agents += [PensionFund("pension", rebalance_every=10, size=200,
                           cash=1_500_000.0, inventory=8_000.0)]
    agents += [RetailTrader(f"retail{i}") for i in range(n_retail)]
    agents += [NoiseTrader(f"noise{i}", market_frac=0.7) for i in range(n_noise)]
    return Exchange(agents, start_price=start_price, seed=seed, **kwargs)


def flash_crash(seed: int | None = None, steps: int = 3_000,
                crash_at: int = 1_500, outage: int = 120) -> Exchange:
    """Scenario: every liquidity provider steps away for ``outage`` steps.

    Market makers go offline and fundamental traders, suddenly unsure what
    anything is worth, stop leaning against the move — which is what real
    liquidity providers did on May 6, 2010. Noise and momentum flow then
    hits a nearly empty book: price gaps, liquidity-taking begets
    liquidity-taking, and the market only heals when providers return.
    No agent decides to crash the market; the crash is what the system
    does when liquidity leaves.

    This is a *liquidity* crisis: nobody borrows and nobody fails. For the
    credit version, see :func:`credit_crisis`.
    """
    exchange = default_market(seed=seed)
    liquidator = Liquidator("liquidator", start_step=crash_at, chunk=150, total=15_000)
    exchange.agents[liquidator.name] = liquidator
    liquidator._peers = exchange.agents
    for _ in range(steps):
        if exchange.step_count + 1 == crash_at:
            exchange.withdraw_liquidity(crash_at + outage)
        exchange.step()
    return exchange


def credit_crisis(seed: int | None = None, steps: int = 1_500, shock_at: int = 300,
                  shock_size: float = 0.15, fund_leverage: float = 6.0,
                  base_haircut: float = 0.18) -> Exchange:
    """Scenario: the fundamental repricing that starts a deleveraging spiral.

    At ``shock_at`` the asset is revealed to be worth ``shock_size`` less
    than everyone thought — the 2007 analogue of subprime losses turning
    out to be real. Nothing else is scripted. What follows is emergent:

    - marks fall, so levered funds' equity falls faster than the price,
    - dealers see volatility and raise haircuts, which margin-calls every
      fund at once,
    - forced selling into a book that HFT has already left pushes marks
      lower, which raises haircuts again,
    - funds that run out of equity fail, and their creditors eat the loss —
      which can push a *dealer* under, tightening funding for everyone still
      standing.

    The shock is exogenous, because in 2007 it was. Everything downstream —
    whether the loss stays contained or becomes a cascade — is the model's
    to decide, and it decides differently depending on leverage, haircuts
    and how much liquidity was real.

    ``shock_at`` is deliberately early. Left alone for long enough, levered
    funds bleed equity and quietly deleverage themselves, and a shock that
    lands on an already-degeared system does very little. That is a real
    result rather than an inconvenience — fragility is a property of the
    *state*, not of the shock — but it means a late shock measures a market
    that is no longer levered. :func:`exchange_sim.fragility.measure_state`
    reports the leverage a shock would actually be landing on.
    """
    exchange = systemic_market(seed=seed, fund_leverage=fund_leverage,
                               base_haircut=base_haircut)
    for _ in range(steps):
        if exchange.step_count + 1 == shock_at:
            # a permanent repricing: the fundamental drifts on from the new level
            exchange.fundamental *= (1.0 - shock_size)
        exchange.step()
    return exchange
