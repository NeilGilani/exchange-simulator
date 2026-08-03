"""Fragility measurement: how big a shock can this market currently absorb?

**This module does not predict anything.** It cannot tell you that prices
will fall, or when. Nothing in this repository can, and you should be
suspicious of anything that claims otherwise — a model that reliably called
crashes would stop working the moment anyone traded on it.

What it does instead is the question risk desks and central banks actually
ask, and the only one an agent-based model can honestly answer:

    Given the state the market is in *right now* — this much leverage,
    this much resting depth, positions this crowded — how large a shock
    does it take before losses stop being absorbed and start cascading?

The method is a counterfactual sweep. Freeze the current state, deep-copy
it, hit the copy with a shock of size *s*, and run it forward. Do that many
times per shock size with different random draws, and across a ladder of
sizes. What comes back is not a forecast, it is a **response curve**: how
the same market handles a 2% repricing versus an 8% one.

The interesting output is where that curve bends. Below the bend a shock is
just a price move; above it, margin calls force selling into a book that
has already thinned, and failures start. That bend is the
:attr:`FragilityReport.absorption_threshold`.

Read it as *"the system is currently in a state where a shock larger than
X cascades"* — a statement about fragility, which is measurable, not about
what will happen, which isn't.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from exchange_sim.institutions import LeveredHedgeFund
from exchange_sim.stats import max_drawdown


@dataclass
class SystemState:
    """A snapshot of how fragile the market is, before any shock is applied.

    Every field is computable from information a regulator would actually
    have. None of them says anything about direction.
    """

    step: int
    mid: float
    #: assets/equity across all solvent agents
    aggregate_leverage: float
    #: assets/equity across levered agents only — the number that bites
    levered_leverage: float
    #: shares resting in the top 5 price levels, both sides
    book_depth: float
    #: how one-sided levered positioning is. 1.0 = everyone the same way
    crowding: float
    #: distance from the margin line for the tightest levered agent
    margin_headroom: float
    #: fraction of all credit extended by the single largest lender
    funding_concentration: float
    total_debt: float
    failures: int

    def summary(self) -> str:
        return (f"leverage {self.levered_leverage:.1f}x · depth {self.book_depth:,.0f} "
                f"· crowding {self.crowding:.0%} · headroom {self.margin_headroom:+.1%}")


@dataclass
class ShockOutcome:
    """What happened across repeated runs of one shock size."""

    shock: float
    drawdowns: list[float] = field(default_factory=list)
    failures: list[int] = field(default_factory=list)
    cascade_depths: list[int] = field(default_factory=list)
    forced_volumes: list[float] = field(default_factory=list)
    first_failures: list[str] = field(default_factory=list)
    #: median drawdown of the unshocked control, filled in by shock_response.
    #: Without it, amplification counts the market's ordinary wander as
    #: damage caused by the shock — and flatters every number in the table.
    baseline_drawdown: float = 0.0

    @property
    def median_drawdown(self) -> float:
        return float(np.median(self.drawdowns)) if self.drawdowns else 0.0

    @property
    def worst_drawdown(self) -> float:
        return float(np.min(self.drawdowns)) if self.drawdowns else 0.0

    @property
    def cascade_probability(self) -> float:
        """Fraction of runs in which at least one agent failed."""
        if not self.failures:
            return 0.0
        return float(np.mean([f > 0 for f in self.failures]))

    @property
    def excess_drawdown(self) -> float:
        """Drawdown attributable to the shock, net of the market's own wander."""
        return abs(self.median_drawdown) - abs(self.baseline_drawdown)

    @property
    def amplification(self) -> float:
        """Shock-attributable drawdown as a multiple of the shock itself.

        1.0 means the market simply repriced by the shock and stopped.
        Above 1.0 means the mechanism added to it — forced selling into
        thinning liquidity, which margin-calls someone else. This is the
        number that separates a repricing from a crisis.

        Measured against an unshocked control run from the same state, so
        that a market which happens to be choppy doesn't read as fragile.
        """
        if self.shock <= 0:
            return 1.0
        return max(0.0, self.excess_drawdown) / self.shock


@dataclass
class FragilityReport:
    state: SystemState
    outcomes: list[ShockOutcome]
    horizon: int
    runs: int
    #: the unshocked control: what this market did over the same horizon
    #: with nothing done to it. Every other row is read against this one.
    control: ShockOutcome | None = None

    @property
    def absorption_threshold(self) -> float | None:
        """Largest shock this market absorbed without a single failure.

        ``None`` if even the smallest tested shock caused failures — which
        means the market was already fragile before you shocked it.
        """
        clean = [o.shock for o in self.outcomes if o.cascade_probability == 0.0]
        return max(clean) if clean else None

    @property
    def cascade_threshold(self) -> float | None:
        """Smallest shock that caused failures in a majority of runs."""
        bad = [o.shock for o in self.outcomes if o.cascade_probability >= 0.5]
        return min(bad) if bad else None

    def first_to_fail(self, top: int = 5) -> list[tuple[str, int]]:
        """Which agents failed first, and how often. The weak link."""
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            for name in outcome.first_failures:
                counts[name] = counts.get(name, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])[:top]

    def summary(self) -> str:
        lines = [f"state: {self.state.summary()}", ""]
        if self.control is not None:
            lines.append(f"control (no shock): median drawdown "
                         f"{self.control.median_drawdown:.1%} over {self.horizon} steps")
            lines.append("")
        lines.append(f"{'shock':>7} {'median dd':>10} {'worst dd':>9} "
                     f"{'excess':>8} {'amplif':>7} {'cascade':>8}")
        for outcome in self.outcomes:
            lines.append(f"{outcome.shock:>6.1%} {outcome.median_drawdown:>10.1%} "
                         f"{outcome.worst_drawdown:>9.1%} {outcome.excess_drawdown:>8.1%} "
                         f"{outcome.amplification:>7.2f}x "
                         f"{outcome.cascade_probability:>8.0%}")
        absorbed = self.absorption_threshold
        cascade = self.cascade_threshold
        lines.append("")
        lines.append(f"absorbed cleanly up to : "
                     f"{'nothing tested' if absorbed is None else f'{absorbed:.1%}'}")
        lines.append(f"cascades from          : "
                     f"{'not within tested range' if cascade is None else f'{cascade:.1%}'}")
        weak = self.first_to_fail(3)
        if weak:
            lines.append("first to fail          : "
                         + ", ".join(f"{n} ({c})" for n, c in weak))
        return "\n".join(lines)


# -- measurement --------------------------------------------------------------


def measure_state(exchange) -> SystemState:
    """Read the current fragility of a market without perturbing it."""
    mid = exchange._mid()
    live = exchange.live_agents()
    levered = [a for a in live if a.sheet.debt > 0]

    equity = sum(a.sheet.equity(mid) for a in levered)
    assets = sum(a.sheet.gross_assets(mid) for a in levered)
    if not levered:
        # nobody is running on borrowed money — possibly because the levered
        # agents already failed. Either way there is no leverage left to
        # amplify anything, and reporting "inf" would say the opposite.
        levered_leverage = 1.0
    elif equity <= 0:
        levered_leverage = float("inf")
    else:
        levered_leverage = assets / equity

    # directional crowding across levered agents: 1.0 = all the same way
    positions = [a.sheet.inventory for a in levered]
    gross = sum(abs(p) for p in positions)
    crowding = abs(sum(positions)) / gross if gross > 0 else 0.0

    headrooms = [a.sheet.margin_ratio(mid) - a.limits.maintenance_margin
                 for a in levered]
    margin_headroom = min(headrooms) if headrooms else float("inf")

    lent_by: dict[str, float] = {}
    for (lender, _), amount in exchange.funding.loans.items():
        lent_by[lender] = lent_by.get(lender, 0.0) + amount
    total_lent = sum(lent_by.values())
    concentration = (max(lent_by.values()) / total_lent) if total_lent > 0 else 0.0

    return SystemState(
        step=exchange.step_count,
        mid=mid,
        aggregate_leverage=exchange.aggregate_leverage(mid),
        levered_leverage=levered_leverage,
        book_depth=exchange.book_depth(),
        crowding=crowding,
        margin_headroom=margin_headroom,
        funding_concentration=concentration,
        total_debt=exchange.total_debt(),
        failures=len(exchange.defaults),
    )


def _run_shocked_copy(exchange, shock: float, horizon: int, seed: int):
    """Deep-copy the market, apply a shock, run it forward.

    The copy is a genuine counterfactual: same balance sheets, same resting
    orders, same funding graph. Only the random draws and the shock differ.
    """
    clone = copy.deepcopy(exchange)
    clone.rng = np.random.default_rng(seed)
    baseline_defaults = len(clone.defaults)
    start_mid = clone._mid()
    start_index = len(clone.tape.mid)

    clone.fundamental *= (1.0 - shock)
    clone.run(horizon)

    path = np.asarray([start_mid] + clone.tape.mid[start_index:])
    new_defaults = clone.defaults[baseline_defaults:]
    depth = (max(d.generation for d in new_defaults) + 1) if new_defaults else 0
    first = new_defaults[0].agent if new_defaults else None
    forced = float(sum(clone.tape.forced_volume[start_index:]))
    return max_drawdown(path), len(new_defaults), depth, forced, first


def shock_response(exchange, shocks=(0.02, 0.04, 0.06, 0.08, 0.12),
                   runs: int = 8, horizon: int = 300,
                   seed: int = 0) -> FragilityReport:
    """Hit a copy of this market with each shock, many times, and report.

    ``shocks`` are fractional repricings of the fundamental. ``runs`` is how
    many independent draws per shock size — the spread across runs matters
    as much as the median, because fragility shows up as a *widening* of
    outcomes before it shows up as a worse average.

    Nothing here touches ``exchange``; every run is on a deep copy.
    """
    state = measure_state(exchange)

    def sweep(shock: float, index: int) -> ShockOutcome:
        outcome = ShockOutcome(shock=shock)
        for run in range(runs):
            dd, failures, depth, forced, first = _run_shocked_copy(
                exchange, shock, horizon, seed=seed + 1_000 * index + run)
            outcome.drawdowns.append(dd)
            outcome.failures.append(failures)
            outcome.cascade_depths.append(depth)
            outcome.forced_volumes.append(forced)
            if first is not None:
                outcome.first_failures.append(first)
        return outcome

    # the control shares its seeds with nothing else, so it is a clean read
    control = sweep(0.0, index=900)
    outcomes = []
    for i, shock in enumerate(shocks):
        outcome = sweep(shock, index=i)
        outcome.baseline_drawdown = control.median_drawdown
        outcomes.append(outcome)
    return FragilityReport(state=state, outcomes=outcomes, horizon=horizon,
                           runs=runs, control=control)


def leverage_sweep(build, leverages=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
                   shock: float = 0.15, warmup: int = 800, horizon: int = 600,
                   runs: int = 4) -> dict[float, ShockOutcome]:
    """Same shock, same everything — vary only how much of it is financed.

    ``build(leverage, seed)`` must return a fresh :class:`Exchange`. This is
    the cleanest experiment in the repository: it isolates leverage as the
    variable that decides whether an identical shock is a price move or a
    cascade.
    """
    results: dict[float, ShockOutcome] = {}
    for leverage in leverages:
        outcome = ShockOutcome(shock=shock)
        controls: list[float] = []
        for run in range(runs):
            exchange = build(leverage, run)
            exchange.run(warmup)
            dd, failures, depth, forced, first = _run_shocked_copy(
                exchange, shock, horizon, seed=10_000 + run)
            outcome.drawdowns.append(dd)
            outcome.failures.append(failures)
            outcome.cascade_depths.append(depth)
            outcome.forced_volumes.append(forced)
            if first is not None:
                outcome.first_failures.append(first)
            # unshocked control from the same warmed-up state
            controls.append(_run_shocked_copy(exchange, 0.0, horizon,
                                              seed=20_000 + run)[0])
        outcome.baseline_drawdown = float(np.median(controls))
        results[leverage] = outcome
    return results
