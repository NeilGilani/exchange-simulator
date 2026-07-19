"""The exchange: agents submit orders to a real matching engine, trades are
settled agent-to-agent, and the tape records everything.

Settlement is exact double-entry: every trade moves cash one way and shares
the other, so total cash and total shares across all agents are invariants —
and the test suite asserts it. If the simulator ever "prints" money or
shares, that's a bug, not a market.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class Tape:
    """Everything the market produced, step by step."""
    mid: list[float] = field(default_factory=list)
    spread: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    fundamental: list[float] = field(default_factory=list)

    def returns(self) -> np.ndarray:
        mids = np.asarray(self.mid)
        return mids[1:] / mids[:-1] - 1.0


class Exchange:
    def __init__(self, agents: list[Agent], start_price: float = 100.0,
                 fundamental_vol: float = 0.0004, seed: int | None = None):
        self.book = OrderBook()
        self.agents = {agent.name: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("agent names must be unique")
        self.rng = np.random.default_rng(seed)
        self.fundamental = start_price
        self.fundamental_vol = fundamental_vol
        self.last_mid = start_price
        self.last_trade_price: float | None = None
        self.tape = Tape()
        self.step_count = 0
        self._order_counter = 0

    # -- invariants -----------------------------------------------------------

    def total_cash(self) -> float:
        return sum(agent.cash for agent in self.agents.values())

    def total_shares(self) -> float:
        return sum(agent.inventory for agent in self.agents.values())

    # -- one market step ------------------------------------------------------

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

    def _settle(self, trades) -> float:
        """Move cash and shares between maker and taker for every fill."""
        volume = 0.0
        for trade in trades:
            maker = self.agents[trade.maker_id.split("#")[0]]
            taker = self.agents[trade.taker_id.split("#")[0]]
            buyer, seller = (taker, maker) if trade.taker_side == "buy" else (maker, taker)
            notional = trade.qty * trade.price
            buyer.cash -= notional
            buyer.inventory += trade.qty
            seller.cash += notional
            seller.inventory -= trade.qty
            volume += trade.qty
            self.last_trade_price = trade.price
        return volume

    def step(self) -> None:
        self.step_count += 1
        # fundamental value drifts slowly — the anchor traders lean toward
        self.fundamental *= float(np.exp(self.rng.normal(0.0, self.fundamental_vol)))

        view = MarketView(
            step=self.step_count,
            mid=self._mid(),
            best_bid=self.book.best_bid(),
            best_ask=self.book.best_ask(),
            mid_history=self.tape.mid,
        )
        volume = 0.0
        order = list(self.agents.values())
        self.rng.shuffle(order)
        for agent in order:
            if isinstance(agent, FundamentalTrader):
                agent.fundamental = self.fundamental
            if agent.wants_cancel_all(view):
                for order_id in agent.open_order_ids:
                    self.book.cancel(order_id)
                agent.open_order_ids.clear()
            for intent in agent.act(view, self.rng):
                self._order_counter += 1
                order_id = f"{agent.name}#{self._order_counter}"
                if intent.kind == "market":
                    trades = self.book.market(intent.side, intent.qty, id=order_id)
                else:
                    trades = self.book.limit(intent.side, intent.price, intent.qty,
                                             id=order_id)
                    agent.open_order_ids.append(order_id)
                volume += self._settle(trades)

        mid = self._mid()
        spread = self.book.spread()
        self.tape.mid.append(mid)
        self.tape.spread.append(spread if spread is not None else float("nan"))
        self.tape.volume.append(volume)
        self.tape.fundamental.append(self.fundamental)

    def run(self, steps: int) -> Tape:
        for _ in range(steps):
            self.step()
        return self.tape


def default_market(seed: int | None = None, n_noise: int = 12, n_momentum: int = 4,
                   n_fundamental: int = 3, start_price: float = 100.0) -> Exchange:
    """The standard ecosystem: 2 market makers + noise + momentum + anchors.

    Parameters were calibrated empirically so the emergent return series
    reproduces real-market stylized facts (see README): excess kurtosis in
    the 2–15 range, clear volatility clustering, small raw autocorrelation.
    """
    agents: list[Agent] = [
        MarketMaker("mm1", inventory_skew=0.008),
        MarketMaker("mm2", half_spread=0.07, size=30, inventory_skew=0.008),
    ]
    agents += [NoiseTrader(f"noise{i}", market_frac=0.75) for i in range(n_noise)]
    agents += [MomentumTrader(f"momo{i}", lookback=8 + 4 * i, threshold=0.003, size=5)
               for i in range(n_momentum)]
    agents += [FundamentalTrader(f"fund{i}", size=30) for i in range(n_fundamental)]
    return Exchange(agents, start_price=start_price, seed=seed)


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
    """
    exchange = default_market(seed=seed)
    liquidator = Liquidator("liquidator", start_step=crash_at, chunk=150, total=15_000)
    exchange.agents[liquidator.name] = liquidator
    for _ in range(steps):
        if exchange.step_count + 1 == crash_at:
            for agent in exchange.agents.values():
                if isinstance(agent, (MarketMaker, FundamentalTrader)):
                    agent.offline_until = crash_at + outage
        exchange.step()
    return exchange
