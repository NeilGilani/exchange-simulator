"""Trading agents. Each observes the market and returns order intentions.

The zoo is deliberately small and interpretable — four archetypes are enough
for realistic microstructure to emerge:

- MarketMaker: provides liquidity, quotes around fair value, skews quotes to
  shed inventory. The market's shock absorber.
- NoiseTrader: uninformed flow — random market/limit orders. The market's
  background radiation.
- MomentumTrader: buys what went up, sells what went down. The market's
  amplifier (this is where volatility clustering comes from).
- FundamentalTrader: trades toward a slow-moving fair value. The market's
  anchor (this is what keeps price from wandering to infinity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OrderIntent:
    kind: str            # "limit" | "market"
    side: str            # "buy" | "sell"
    qty: float
    price: float | None = None   # None for market orders


@dataclass
class MarketView:
    """What an agent is allowed to see: public information only."""
    step: int
    mid: float
    best_bid: float | None
    best_ask: float | None
    mid_history: list[float] = field(default_factory=list)


class Agent:
    """Base agent: subclasses implement act(view) -> list[OrderIntent]."""

    def __init__(self, name: str, cash: float = 0.0, inventory: float = 0.0):
        self.name = name
        self.cash = cash
        self.inventory = inventory
        self.open_order_ids: list[str] = []

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        raise NotImplementedError

    def wants_cancel_all(self, view: MarketView) -> bool:
        """Return True to pull every resting order before acting this step."""
        return False

    def wealth(self, mid: float) -> float:
        return self.cash + self.inventory * mid


class MarketMaker(Agent):
    """Quotes both sides around the mid, skewing quotes against inventory.

    The inventory skew is the classic Avellaneda–Stoikov intuition in one
    line: long inventory pushes both quotes down (eager to sell, reluctant
    to buy), short inventory pushes them up.
    """

    def __init__(self, name: str, half_spread: float = 0.05, size: float = 40,
                 inventory_skew: float = 0.002, max_inventory: float = 600,
                 cash: float = 100_000.0):
        super().__init__(name, cash=cash)
        self.half_spread = half_spread
        self.size = size
        self.inventory_skew = inventory_skew
        self.max_inventory = max_inventory
        self.offline_until = -1   # scenario hook: withdraw liquidity until step N

    def wants_cancel_all(self, view: MarketView) -> bool:
        return True  # re-quote fresh every step

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if view.step < self.offline_until:
            return []  # liquidity withdrawn — the flash-crash ingredient
        skew = self.inventory * self.inventory_skew
        bid = view.mid - self.half_spread - skew
        ask = view.mid + self.half_spread - skew
        intents = []
        if self.inventory < self.max_inventory and bid > 0:
            intents.append(OrderIntent("limit", "buy", self.size, round(bid, 2)))
        if self.inventory > -self.max_inventory:
            intents.append(OrderIntent("limit", "sell", self.size, round(max(ask, 0.01), 2)))
        return intents


class NoiseTrader(Agent):
    """Uninformed flow: occasional random market orders, otherwise passive
    limit orders scattered near the touch."""

    def __init__(self, name: str, activity: float = 0.5, market_frac: float = 0.5,
                 mean_size: float = 8, cash: float = 20_000.0, inventory: float = 100.0):
        super().__init__(name, cash=cash, inventory=inventory)
        self.activity = activity
        self.market_frac = market_frac
        self.mean_size = mean_size

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if rng.random() > self.activity:
            return []
        side = "buy" if rng.random() < 0.5 else "sell"
        qty = float(1 + rng.geometric(1 / self.mean_size))
        if rng.random() < self.market_frac:
            return [OrderIntent("market", side, qty)]
        # exponential offsets: most limits cluster near the touch, but a fat
        # tail of deep "stub" orders sits far away — the resting liquidity
        # that crash prints hit when the top of the book is swept
        offset = float(rng.exponential(0.5))
        price = view.mid - offset if side == "buy" else view.mid + offset
        if price <= 0:
            return []
        return [OrderIntent("limit", side, qty, round(price, 2))]


class MomentumTrader(Agent):
    """Return-chaser: market-buys after rallies, market-sells after dips.

    Momentum flow begets more momentum flow, which is exactly how volatility
    clusters in real markets — one shock recruits amplifiers.
    """

    def __init__(self, name: str, lookback: int = 12, threshold: float = 0.0015,
                 size: float = 12, cash: float = 30_000.0, inventory: float = 100.0):
        super().__init__(name, cash=cash, inventory=inventory)
        self.lookback = lookback
        self.threshold = threshold
        self.size = size

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        h = view.mid_history
        if len(h) <= self.lookback:
            return []
        ret = h[-1] / h[-self.lookback - 1] - 1.0
        if ret > self.threshold:
            return [OrderIntent("market", "buy", self.size)]
        if ret < -self.threshold:
            return [OrderIntent("market", "sell", self.size)]
        return []


class Liquidator(Agent):
    """Executes a large forced sale as relentless market orders.

    This is the flash-crash trigger from May 6, 2010: a big sell program
    that keeps hitting the book regardless of price. Harmless when
    liquidity providers are present; catastrophic when they step away.
    """

    def __init__(self, name: str, start_step: int, chunk: float = 30,
                 total: float = 3_000, cash: float = 0.0):
        super().__init__(name, cash=cash, inventory=total)
        self.start_step = start_step
        self.chunk = chunk
        self.remaining = total

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if view.step < self.start_step or self.remaining <= 0:
            return []
        qty = min(self.chunk, self.remaining)
        self.remaining -= qty
        return [OrderIntent("market", "sell", qty)]


class FundamentalTrader(Agent):
    """Values the asset at a slow-moving fundamental (told to it by the
    engine) and leans against deviations with limit orders."""

    def __init__(self, name: str, band: float = 0.004, size: float = 15,
                 cash: float = 50_000.0, inventory: float = 200.0):
        super().__init__(name, cash=cash, inventory=inventory)
        self.band = band
        self.size = size
        self.fundamental = None  # engine sets this each step
        self.offline_until = -1  # scenario hook: uncertainty -> step away

    def wants_cancel_all(self, view: MarketView) -> bool:
        return True

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if self.fundamental is None or view.step < self.offline_until:
            return []
        deviation = view.mid / self.fundamental - 1.0
        if deviation < -self.band:   # cheap vs fundamental: bid aggressively
            return [OrderIntent("limit", "buy", self.size, round(view.mid + 0.01, 2))]
        if deviation > self.band:    # rich vs fundamental: offer aggressively
            price = max(view.mid - 0.01, 0.01)
            return [OrderIntent("limit", "sell", self.size, round(price, 2))]
        return []
