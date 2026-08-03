"""The institutional zoo: who actually holds the risk.

The four original archetypes (market maker, noise, momentum, fundamental)
are enough to grow microstructure. They are not enough to grow a credit
crisis, because none of them can borrow and none of them can fail.

These five can. What separates them is not their signal — several trade on
much the same view — but their *balance sheet* and their *funding*:

- :class:`DealerBank` quotes markets **and** finances everyone else. It is
  the single most important agent here, because when it gets scared it does
  two things at once: it widens its quotes (less liquidity) and it raises
  haircuts (less funding). Those two levers are what turn a price drop into
  a spiral.
- :class:`LeveredHedgeFund` runs a strategy at 3-6x. Sharp, fast, and the
  first to be forced out when the haircut moves against it.
- :class:`PensionFund` is enormous, unlevered, and slow. Patient capital —
  the only thing in the model that reliably buys during a crash.
- :class:`RetailTrader` is small, herds on recent returns, and capitulates
  into drawdowns.
- :class:`HFTMarketMaker` posts the tightest spread in the book and the
  smallest size, and vanishes the instant volatility spikes. Most of the
  visible depth in calm markets; none of it in a crash.

The funding channel matters more than any single agent. :class:`FundingDesk`
translates dealer risk appetite into each borrower's maintenance margin, so
a haircut rising from 20% to 35% mechanically margin-calls every fund
financed at 4x — no agent "decides" to start the spiral.
"""
from __future__ import annotations

import numpy as np

from exchange_sim.agents import Agent, MarketView, OrderIntent
from exchange_sim.balance import FundingNetwork, RiskLimits


class DealerBank(Agent):
    """Market maker of last resort, and the system's lender.

    Two behaviours that ordinary market makers don't have:

    1. **Risk-off quoting.** Spread widens and size shrinks with realised
       volatility, drawdown and visible failures.
    2. **Haircut setting.** It decides how levered its borrowers may be.
       Under stress it raises the haircut, which is a margin call on
       everybody it finances, delivered simultaneously.
    """

    def __init__(self, name: str, half_spread: float = 0.06, size: float = 45,
                 inventory_skew: float = 0.006, max_inventory: float = 900,
                 cash: float = 400_000.0, lending_capacity: float = 300_000.0,
                 base_haircut: float = 0.18, risk_aversion: float = 60.0,
                 panic_threshold: float = 0.12):
        super().__init__(name, cash=cash,
                         limits=RiskLimits(maintenance_margin=0.10, target_margin=0.18))
        self.half_spread = half_spread
        self.size = size
        self.inventory_skew = inventory_skew
        self.max_inventory = max_inventory
        self.lending_capacity = lending_capacity
        self.base_haircut = base_haircut
        self.risk_aversion = risk_aversion
        self.panic_threshold = panic_threshold
        self.offline_until = -1

    def wants_cancel_all(self, view: MarketView) -> bool:
        return True

    def stress(self, view: MarketView) -> float:
        """0 = calm, 1 = fully risk-off. Built from public information only."""
        vol_stress = min(1.0, view.volatility * self.risk_aversion)
        dd_stress = min(1.0, abs(view.drawdown) / self.panic_threshold)
        failure_stress = min(1.0, view.failures * 0.5)
        return max(vol_stress, dd_stress, failure_stress)

    def haircut(self, view: MarketView) -> float:
        """Minimum equity/assets it will finance a borrower at.

        Calm: ``base_haircut``. Stressed: up to ~2.5x that. This is the
        Brunnermeier-Pedersen margin spiral in one line — funding tightens
        exactly when borrowers can least afford it.
        """
        return min(0.60, self.base_haircut * (1.0 + 1.5 * self.stress(view)))

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if view.step < self.offline_until or self.sheet.failed:
            return []
        stress = self.stress(view)
        # risk-off: quote wider and smaller, but never fully disappear
        half_spread = self.half_spread * (1.0 + 4.0 * stress)
        size = self.size * max(0.15, 1.0 - 0.85 * stress)
        skew = self.inventory * self.inventory_skew
        bid = view.mid - half_spread - skew
        ask = view.mid + half_spread - skew
        intents = []
        if self.inventory < self.max_inventory and bid > 0:
            intents.append(OrderIntent("limit", "buy", size, round(bid, 2)))
        if self.inventory > -self.max_inventory:
            intents.append(OrderIntent("limit", "sell", size, round(max(ask, 0.01), 2)))
        return intents


class HFTMarketMaker(Agent):
    """Tightest quote in the book, smallest size, first to leave.

    Runs flat: any inventory is dumped immediately. Contributes most of the
    depth you can see in a calm market and essentially none of the depth
    you need in a falling one — the "phantom liquidity" complaint, as a
    mechanism rather than an accusation.
    """

    def __init__(self, name: str, half_spread: float = 0.02, size: float = 25,
                 vol_tolerance: float = 0.0025, max_inventory: float = 120,
                 cash: float = 60_000.0):
        super().__init__(name, cash=cash)
        self.half_spread = half_spread
        self.size = size
        self.vol_tolerance = vol_tolerance
        self.max_inventory = max_inventory

    def wants_cancel_all(self, view: MarketView) -> bool:
        return True

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if self.sheet.failed:
            return []
        if view.volatility > self.vol_tolerance:
            return []   # switched off — not offline, just unwilling
        intents: list[OrderIntent] = []
        # go home flat before quoting again
        if abs(self.inventory) > self.max_inventory:
            side = "sell" if self.inventory > 0 else "buy"
            intents.append(OrderIntent("market", side, abs(self.inventory) * 0.5))
        bid = view.mid - self.half_spread
        ask = view.mid + self.half_spread
        if bid > 0:
            intents.append(OrderIntent("limit", "buy", self.size, round(bid, 2)))
        intents.append(OrderIntent("limit", "sell", self.size, round(max(ask, 0.01), 2)))
        return intents


class LeveredHedgeFund(Agent):
    """A strategy run on borrowed money.

    ``style`` picks the signal — ``"momentum"`` chases trends, ``"value"``
    leans against deviations from fundamental. The signal is almost beside
    the point: what makes this agent dangerous is that it sizes to
    ``target_leverage``, so a move against it forces selling regardless of
    what it believes.

    Crowding is deliberate. Several funds sharing a style will reach for the
    same trade and be forced out of it at the same moment, which is how a
    position that looks liquid agent-by-agent turns out not to be.
    """

    def __init__(self, name: str, style: str = "momentum", target_leverage: float = 4.0,
                 lookback: int = 10, threshold: float = 0.002, band: float = 0.004,
                 aggression: float = 0.25, cash: float = 40_000.0,
                 inventory: float = 0.0, max_position_value: float = 400_000.0):
        super().__init__(name, cash=cash, inventory=inventory,
                         limits=RiskLimits(maintenance_margin=0.20, target_margin=0.30))
        if style not in ("momentum", "value"):
            raise ValueError("style must be 'momentum' or 'value'")
        self.style = style
        self.target_leverage = target_leverage
        self.lookback = lookback
        self.threshold = threshold
        self.band = band
        self.aggression = aggression
        self.max_position_value = max_position_value
        self.fundamental: float | None = None   # engine sets this for value funds

    def target_leverage_for(self, view: MarketView) -> float:
        return self.target_leverage

    def _signal(self, view: MarketView) -> float:
        """-1 short conviction .. +1 long conviction."""
        if self.style == "momentum":
            h = view.mid_history
            if len(h) <= self.lookback:
                return 0.0
            ret = h[-1] / h[-self.lookback - 1] - 1.0
            if abs(ret) < self.threshold:
                return 0.0
            return float(np.clip(ret / (self.threshold * 8), -1.0, 1.0))
        if self.fundamental is None:
            return 0.0
        deviation = view.mid / self.fundamental - 1.0
        if abs(deviation) < self.band:
            return 0.0
        return float(np.clip(-deviation / (self.band * 6), -1.0, 1.0))

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if self.sheet.failed:
            return []
        equity = self.sheet.equity(view.mid)
        if equity <= 0:
            return []
        signal = self._signal(view)
        # size to conviction x leverage, but never beyond the mandate cap
        desired_value = np.clip(signal * self.target_leverage * equity,
                                -self.max_position_value, self.max_position_value)
        desired_shares = desired_value / view.mid
        gap = desired_shares - self.inventory
        # move only part of the way each step: real books can't be flipped at once
        qty = abs(gap) * self.aggression
        if qty < 1:
            return []
        return [OrderIntent("market", "buy" if gap > 0 else "sell", float(qty))]


class PensionFund(Agent):
    """Huge, unlevered, slow, and mandated to stay invested.

    Rebalances on a slow clock toward a target allocation, which makes it a
    natural buyer of things that just fell. It is the only agent here that
    reliably supplies liquidity in a crash — and, being slow, it supplies it
    a long way after the crash has happened.
    """

    def __init__(self, name: str, rebalance_every: int = 25, size: float = 60,
                 band: float = 0.01, cash: float = 900_000.0, inventory: float = 4_000.0):
        super().__init__(name, cash=cash, inventory=inventory)
        self.rebalance_every = rebalance_every
        self.size = size
        self.band = band
        self.fundamental: float | None = None

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if self.sheet.failed or self.fundamental is None:
            return []
        if view.step % self.rebalance_every != 0:
            return []
        deviation = view.mid / self.fundamental - 1.0
        if deviation < -self.band and self.cash > view.mid * self.size:
            return [OrderIntent("limit", "buy", self.size, round(view.mid - 0.02, 2))]
        if deviation > self.band and self.inventory > self.size:
            return [OrderIntent("limit", "sell", self.size,
                                round(max(view.mid + 0.02, 0.01), 2))]
        return []


class RetailTrader(Agent):
    """Small, sentiment-driven, and prone to capitulation.

    Herds on recent returns like a momentum trader, but with two twists
    that matter: it trades in small clips, and past a drawdown threshold it
    panics and dumps its position outright regardless of signal.
    """

    def __init__(self, name: str, activity: float = 0.35, herd: float = 0.6,
                 lookback: int = 20, panic_drawdown: float = 0.06,
                 mean_size: float = 4, cash: float = 8_000.0, inventory: float = 60.0):
        super().__init__(name, cash=cash, inventory=inventory)
        self.activity = activity
        self.herd = herd
        self.lookback = lookback
        self.panic_drawdown = panic_drawdown
        self.mean_size = mean_size
        self.panicked = False

    def act(self, view: MarketView, rng: np.random.Generator) -> list[OrderIntent]:
        if self.sheet.failed:
            return []
        if view.drawdown < -self.panic_drawdown and self.inventory > 0:
            self.panicked = True
            return [OrderIntent("market", "sell", float(self.inventory))]
        if rng.random() > self.activity:
            return []
        h = view.mid_history
        trend = 0.0
        if len(h) > self.lookback:
            trend = h[-1] / h[-self.lookback - 1] - 1.0
        # herd: probability of buying rises with the recent trend
        p_buy = 0.5 + self.herd * float(np.clip(trend * 40, -0.45, 0.45))
        side = "buy" if rng.random() < p_buy else "sell"
        qty = float(1 + rng.geometric(1 / self.mean_size))
        if side == "buy" and self.cash < view.mid * qty:
            return []
        if side == "sell" and self.inventory < qty:
            return []
        return [OrderIntent("market", side, qty)]


class FundingDesk:
    """Turns dealer risk appetite into everybody else's margin requirement.

    Each step it:

    1. asks every solvent dealer for its haircut and takes the **strictest**
       (funding is only as loose as your least generous lender),
    2. writes that haircut into each borrower's :class:`RiskLimits`, so the
       existing margin machinery does the calling,
    3. moves loans toward each borrower's target leverage, subject to what
       dealers can actually fund.

    Step 2 is the important one. Nobody decides to start a deleveraging
    spiral; a haircut moves from 0.20 to 0.34 and every fund financed at 4x
    is suddenly below maintenance at the same instant.
    """

    def __init__(self, funding: FundingNetwork, roll_fraction: float = 0.25,
                 leverage_buffer: float = 0.75):
        self.funding = funding
        self.roll_fraction = roll_fraction
        #: funds borrow to a fraction of the maximum they *could*, so calm
        #: markets leave headroom above the margin line. Without this the
        #: system sits exactly on the line and is called every other step.
        self.leverage_buffer = leverage_buffer
        self.haircut_history: list[float] = []

    def system_haircut(self, dealers, view: MarketView) -> float:
        live = [d for d in dealers if not d.sheet.failed]
        if not live:
            return 1.0   # nobody left to finance you: no leverage at all
        return max(d.haircut(view) for d in live)

    def run_round(self, dealers, borrowers, view: MarketView) -> float:
        haircut = self.system_haircut(dealers, view)
        self.haircut_history.append(haircut)
        mid = view.mid

        for borrower in borrowers:
            if borrower.sheet.failed:
                continue
            # funding conditions ARE the margin requirement
            borrower.limits.maintenance_margin = haircut
            borrower.limits.target_margin = min(0.95, haircut * 1.4)

            equity = borrower.sheet.equity(mid)
            if equity <= 0:
                continue
            allowed_leverage = 1.0 / max(haircut, 1e-6)
            target = min(borrower.target_leverage_for(view),
                         allowed_leverage * self.leverage_buffer)
            desired_debt = max(0.0, equity * (target - 1.0))
            gap = desired_debt - borrower.sheet.debt

            if gap > 0:
                self._extend(dealers, borrower, gap * self.roll_fraction)
            elif gap < 0:
                self._withdraw(borrower, -gap * self.roll_fraction)
        return haircut

    def _extend(self, dealers, borrower, amount: float) -> None:
        candidates = [d for d in dealers
                      if not d.sheet.failed and d.lending_capacity > 0]
        if not candidates:
            return
        per_dealer = amount / len(candidates)
        for dealer in candidates:
            outstanding = sum(self.funding.borrowers_of(dealer.name).values())
            room = dealer.lending_capacity - outstanding
            self.funding.lend(dealer, borrower, min(per_dealer, max(room, 0.0)))

    def _withdraw(self, borrower, amount: float) -> None:
        """Borrower repays from spare cash. If it has none, the margin
        machinery will make it sell — which is the whole problem."""
        for lender_name, owed in sorted(
                self.funding.creditors_of(borrower.name).items(), key=lambda kv: -kv[1]):
            if amount <= 0:
                break
            lender = borrower._peers[lender_name]
            paid = self.funding.repay(lender, borrower, min(amount, owed))
            amount -= paid
