"""Balance sheets, leverage, and the machinery of failure.

The original simulator settled trades between agents but nobody could ever
run out of money: ``buyer.cash -= notional`` with no check. That makes a
market where prices move but nothing can *break*. It is enough to produce
microstructure (spreads, fat tails, a 2010-style liquidity crash) and
nowhere near enough to produce a *credit* crisis.

This module adds the missing half:

- **Balance sheets.** Every agent has cash, inventory, debt and receivables.
  Equity is what's left when you mark everything to market.
- **Leverage and margin.** An agent financing a big book with a thin equity
  cushion gets a margin call when prices move against it, and must sell to
  restore the ratio. Selling moves the price. Moving the price margin-calls
  somebody else. That is the deleveraging spiral, and nobody programs it.
- **Default.** An agent whose equity goes negative *fails*. Its book is
  liquidated into the order book (a fire sale, at whatever price the book
  will bear) and whatever it cannot repay is written off against its
  creditors — who may then fail themselves.

The cascade loop in :func:`resolve_stress` runs to a fixed point, so one
failure can take down a chain. This is the 2008 mechanism rather than the
2010 one: not "liquidity stepped away" but "the loss was somebody else's
asset".

Conservation still holds, and the tests still assert it. Lending moves cash
without creating it; a write-off destroys a *claim* (debt on one side,
receivable on the other) and touches no cash at all. So across all agents:

    total cash    is invariant
    total shares  is invariant
    total equity  == total cash + total shares x mid      (debt/receivables
                                                           cancel pairwise)

If a write-off ever leaks value, ``test_contagion.py`` fails.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    """Margin rules an agent is financed under.

    ``maintenance_margin`` is the equity/assets ratio below which the agent
    is called; ``target_margin`` is what it deleverages back up to (you are
    not called down to the line, you are called back to comfort — which is
    why margin calls sell *more* than the shortfall implies).
    """

    maintenance_margin: float = 0.25
    target_margin: float = 0.35
    max_deleverage_fraction: float = 0.35   # cap on inventory sold per step


@dataclass
class BalanceSheet:
    """Cash, shares, and the claims between agents.

    ``debt`` is money owed to other agents; ``receivables`` is money owed by
    them. The two net out across the whole system, which is exactly why one
    agent's failure is another agent's loss.
    """

    cash: float = 0.0
    inventory: float = 0.0
    debt: float = 0.0
    receivables: float = 0.0
    failed: bool = False
    #: cumulative losses absorbed from counterparties that defaulted
    credit_losses: float = 0.0

    # -- marks ---------------------------------------------------------------

    def equity(self, mid: float) -> float:
        """Net worth marked to the current price."""
        return self.cash + self.inventory * mid + self.receivables - self.debt

    def gross_assets(self, mid: float) -> float:
        """Balance-sheet size. Shorts count as exposure, not as a credit."""
        return max(self.cash, 0.0) + abs(self.inventory) * mid + self.receivables

    def leverage(self, mid: float) -> float:
        """Assets / equity. ``inf`` once equity is gone."""
        equity = self.equity(mid)
        if equity <= 0:
            return float("inf")
        return self.gross_assets(mid) / equity

    def margin_ratio(self, mid: float) -> float:
        """Equity as a fraction of assets — the number the margin desk watches."""
        assets = self.gross_assets(mid)
        if assets <= 0:
            return 1.0
        return self.equity(mid) / assets

    def is_insolvent(self, mid: float) -> bool:
        return self.equity(mid) <= 0.0

    def under_margin(self, mid: float, limits: RiskLimits) -> bool:
        if self.debt <= 0:
            return False   # unlevered agents are never called
        return self.margin_ratio(mid) < limits.maintenance_margin


class FundingNetwork:
    """Who lent how much to whom.

    A dict of ``(lender, borrower) -> amount``. Small, but it is the whole
    reason failure propagates: without a funding graph a default is a
    private misfortune, with one it is everybody's problem.
    """

    def __init__(self) -> None:
        self.loans: dict[tuple[str, str], float] = {}

    # -- graph queries -------------------------------------------------------

    def creditors_of(self, borrower: str) -> dict[str, float]:
        return {lender: amt for (lender, b), amt in self.loans.items()
                if b == borrower and amt > 0}

    def borrowers_of(self, lender: str) -> dict[str, float]:
        return {borrower: amt for (l, borrower), amt in self.loans.items()
                if l == lender and amt > 0}

    def total_lent(self) -> float:
        return sum(self.loans.values())

    # -- mutations -----------------------------------------------------------

    def lend(self, lender, borrower, amount: float) -> float:
        """Move ``amount`` of cash from lender to borrower as a loan.

        Returns the amount actually lent — a lender will not hand over cash
        it does not have. Cash is conserved: it moves, it is not created.
        """
        amount = min(amount, max(lender.sheet.cash, 0.0))
        if amount <= 0:
            return 0.0
        lender.sheet.cash -= amount
        lender.sheet.receivables += amount
        borrower.sheet.cash += amount
        borrower.sheet.debt += amount
        key = (lender.name, borrower.name)
        self.loans[key] = self.loans.get(key, 0.0) + amount
        return amount

    def repay(self, lender, borrower, amount: float) -> float:
        """Borrower repays principal. Cash moves back; the claim shrinks."""
        key = (lender.name, borrower.name)
        outstanding = self.loans.get(key, 0.0)
        amount = min(amount, outstanding, max(borrower.sheet.cash, 0.0))
        if amount <= 0:
            return 0.0
        borrower.sheet.cash -= amount
        borrower.sheet.debt -= amount
        lender.sheet.cash += amount
        lender.sheet.receivables -= amount
        self.loans[key] = outstanding - amount
        return amount

    def write_off(self, lender, borrower, amount: float) -> float:
        """Extinguish a claim the borrower cannot honour.

        No cash moves — the cash was lost when it wasn't repaid. What is
        destroyed is the *claim*: debt on one side, receivable on the other.
        Because both legs die together, total equity across the system stays
        equal to total cash + total shares x mid.
        """
        key = (lender.name, borrower.name)
        outstanding = self.loans.get(key, 0.0)
        amount = min(amount, outstanding)
        if amount <= 0:
            return 0.0
        borrower.sheet.debt -= amount
        lender.sheet.receivables -= amount
        lender.sheet.credit_losses += amount
        self.loans[key] = outstanding - amount
        return amount


@dataclass
class DefaultEvent:
    """One agent failing, and what it cost everyone else."""

    step: int
    agent: str
    equity: float
    liquidated_shares: float
    repaid: float
    written_off: float
    creditors: dict[str, float] = field(default_factory=dict)
    #: how deep in the cascade this failure sat (0 = failed first)
    generation: int = 0


@dataclass
class MarginCall:
    step: int
    agent: str
    margin_ratio: float
    shares_sold: float


@dataclass
class StressReport:
    """Everything the risk machinery did during one step."""

    margin_calls: list[MarginCall] = field(default_factory=list)
    defaults: list[DefaultEvent] = field(default_factory=list)
    forced_volume: float = 0.0

    @property
    def cascade_depth(self) -> int:
        """0 if nothing failed, 1 for isolated failures, >1 for a chain."""
        if not self.defaults:
            return 0
        return max(d.generation for d in self.defaults) + 1


def _deleverage_target(sheet: BalanceSheet, mid: float, limits: RiskLimits) -> float:
    """Shares to sell to restore ``target_margin``.

    Selling inventory at the mark and repaying debt shrinks assets and debt
    by the same amount, leaving equity unchanged. So we need::

        equity / (assets - x) >= target      =>      x >= assets - equity/target
    """
    equity = sheet.equity(mid)
    assets = sheet.gross_assets(mid)
    if equity <= 0:
        return abs(sheet.inventory)          # everything must go
    shrink = assets - equity / limits.target_margin
    if shrink <= 0:
        return 0.0
    return min(shrink / mid, abs(sheet.inventory))


def resolve_stress(agents, funding: FundingNetwork, mid: float, execute,
                   step: int, limits: RiskLimits) -> StressReport:
    """Run margin calls and default resolution to a fixed point.

    ``execute(agent, side, qty) -> (filled_qty, notional)`` submits a market
    order on the agent's behalf and reports what actually happened. Forced
    selling goes through the *real* book at real prices, so a fire sale into
    a thin book gets a terrible fill — which is the entire point.

    The loop repeats because resolution is reflexive: liquidating a failed
    agent pushes the price down, which can margin-call a healthy one, whose
    forced selling pushes the price down further, whose creditors then take
    write-offs and fail in turn. It terminates because each pass either
    fails a new agent (finite) or does nothing.
    """
    report = StressReport()
    generation = 0
    live = [a for a in agents if not a.sheet.failed]

    while True:
        current_mid = mid() if callable(mid) else mid
        progressed = False

        # -- margin calls: still solvent, but financed too thin --------------
        for agent in live:
            sheet = agent.sheet
            if sheet.failed or sheet.is_insolvent(current_mid):
                continue
            if not sheet.under_margin(current_mid, limits):
                continue
            want = _deleverage_target(sheet, current_mid, limits)
            cap = abs(sheet.inventory) * limits.max_deleverage_fraction
            qty = min(want, cap)
            if qty <= 0:
                continue
            side = "sell" if sheet.inventory > 0 else "buy"
            filled, notional = execute(agent, side, qty)
            if filled <= 0:
                continue
            progressed = True
            report.forced_volume += filled
            report.margin_calls.append(
                MarginCall(step, agent.name, sheet.margin_ratio(current_mid), filled))
            _repay_what_you_can(agent, funding)
            current_mid = mid() if callable(mid) else mid

        # -- defaults: equity is gone ---------------------------------------
        newly_failed = [a for a in live
                        if not a.sheet.failed and a.sheet.is_insolvent(current_mid)]
        if not newly_failed:
            if not progressed:
                break
            # forced selling may have pushed somebody under; re-check once more
            continue

        for agent in newly_failed:
            report.defaults.append(
                _default(agent, funding, current_mid, execute, step, generation))
            current_mid = mid() if callable(mid) else mid

        report.forced_volume += sum(d.liquidated_shares for d in report.defaults[-len(newly_failed):])
        live = [a for a in live if not a.sheet.failed]
        generation += 1

    return report


def _repay_what_you_can(agent, funding: FundingNetwork) -> float:
    """Push spare cash back to creditors, largest claim first."""
    repaid = 0.0
    for lender_name, _ in sorted(funding.creditors_of(agent.name).items(),
                                 key=lambda kv: -kv[1]):
        lender = agent._peers[lender_name]
        repaid += funding.repay(lender, agent, agent.sheet.cash)
    return repaid


def _default(agent, funding: FundingNetwork, mid: float, execute, step: int,
             generation: int) -> DefaultEvent:
    """Liquidate a failed agent and push the shortfall onto its creditors."""
    sheet = agent.sheet
    equity_at_failure = sheet.equity(mid)

    # Fire sale: dump the whole book at market, whatever it fetches.
    liquidated = 0.0
    if sheet.inventory != 0:
        side = "sell" if sheet.inventory > 0 else "buy"
        liquidated, _ = execute(agent, side, abs(sheet.inventory))

    sheet.failed = True   # stop acting before creditors are touched

    creditors = funding.creditors_of(agent.name)
    total_owed = sum(creditors.values())
    repaid = written_off = 0.0
    losses: dict[str, float] = {}

    if total_owed > 0:
        available = max(sheet.cash, 0.0)
        for lender_name, owed in sorted(creditors.items(), key=lambda kv: -kv[1]):
            lender = agent._peers[lender_name]
            share = owed / total_owed
            paid = funding.repay(lender, agent, available * share)
            repaid += paid
            loss = funding.write_off(lender, agent, owed - paid)
            written_off += loss
            if loss > 0:
                losses[lender_name] = loss

    return DefaultEvent(step=step, agent=agent.name, equity=equity_at_failure,
                        liquidated_shares=liquidated, repaid=repaid,
                        written_off=written_off, creditors=losses,
                        generation=generation)
