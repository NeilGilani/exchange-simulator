"""Probabilistic forecasting — and the scorecard that says whether it works.

A forecast is a probability distribution over what happens next, and it is
only worth anything if somebody scores it. This module does both, and keeps
them in the same file on purpose: it should be impossible to use the
forecaster without tripping over the evidence about how good it is.

What is being forecast
    Not "the market will fall on Tuesday". The output is a **conditional
    distribution**: run the current state forward many times with different
    random draws and a *free* fundamental (the future is not fed in), and
    report the spread — median path, quantile fan, probability of a drawdown
    worse than x%, probability that somebody fails.

Why this is a real forecast and the earlier replay was not
    :mod:`exchange_sim.history` hands the model the future as an input. Here
    ``fundamental_path`` is switched off at the forecast origin, so the
    simulated future is generated, not read. That makes the output falsifiable
    — which is what lets it be scored at all.

How it is scored
    With proper scoring rules, against a baseline, out of sample:

    - **CRPS** for the whole distribution (lower is better). A proper rule,
      so the best score comes from reporting your true beliefs.
    - **Brier score** for threshold events like "down more than 10%".
    - **Skill score** against a random walk fitted to trailing volatility:
      ``1 - CRPS_model / CRPS_baseline``. Positive means the agent model beat
      the null; zero means it added nothing; **negative means it was worse**.

    :func:`walk_forward` runs this over real history, forecasting from each
    origin using only data available at that origin.

Read the skill score before you read anything else. A forecast that has not
beaten a random walk is not a forecast, it is a simulation with error bars —
and the honest thing is to report the number either way rather than show you
a confident-looking fan chart and let you assume.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from exchange_sim.stats import max_drawdown


@dataclass
class Forecast:
    """A distribution over what happens next, from one origin."""

    origin_step: int
    origin_mid: float
    horizon: int
    runs: int
    #: terminal return of each simulated path, relative to origin_mid
    terminal_returns: np.ndarray
    #: worst peak-to-trough drawdown along each path
    drawdowns: np.ndarray
    #: agents that failed during each path
    failures: np.ndarray
    #: percentile bands of the price path, for a fan chart
    fan: dict[str, np.ndarray] = field(default_factory=dict)
    label: str = "agent-model"

    # -- reading the distribution --------------------------------------------

    def quantile(self, q: float) -> float:
        """Terminal return at quantile ``q`` (0-1)."""
        return float(np.quantile(self.terminal_returns, q))

    def median(self) -> float:
        return self.quantile(0.5)

    def interval(self, level: float = 0.90) -> tuple[float, float]:
        tail = (1.0 - level) / 2
        return self.quantile(tail), self.quantile(1.0 - tail)

    def prob_return_below(self, threshold: float) -> float:
        return float(np.mean(self.terminal_returns < threshold))

    def prob_drawdown_worse_than(self, threshold: float) -> float:
        """P(peak-to-trough drawdown exceeds ``threshold``), e.g. 0.10."""
        return float(np.mean(self.drawdowns < -abs(threshold)))

    def prob_any_failure(self) -> float:
        return float(np.mean(self.failures > 0))

    def summary(self) -> str:
        lo, hi = self.interval(0.90)
        lines = [
            f"forecast [{self.label}] from step {self.origin_step}, "
            f"{self.horizon} steps ahead, {self.runs} paths",
            "",
            f"  median return        {self.median():+.1%}",
            f"  90% interval         {lo:+.1%} to {hi:+.1%}",
            f"  P(down > 5%)         {self.prob_return_below(-0.05):.0%}",
            f"  P(down > 10%)        {self.prob_return_below(-0.10):.0%}",
            f"  P(drawdown > 10%)    {self.prob_drawdown_worse_than(0.10):.0%}",
            f"  P(drawdown > 20%)    {self.prob_drawdown_worse_than(0.20):.0%}",
            f"  P(any agent fails)   {self.prob_any_failure():.0%}",
        ]
        return "\n".join(lines)


# -- scoring rules ------------------------------------------------------------


def crps(samples: np.ndarray, actual: float) -> float:
    """Continuous Ranked Probability Score from an ensemble. Lower is better.

    Uses the standard sample estimator

        CRPS = E|X - y| - 0.5 * E|X - X'|

    which rewards a distribution for being both accurate and sharp. It is a
    *proper* scoring rule: you cannot improve your expected score by
    reporting anything other than your true belief — no credit for hedging
    with an implausibly wide fan.
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        return float("nan")
    accuracy = float(np.mean(np.abs(x - actual)))
    # E|X - X'| via the sorted-sample identity, O(n log n) rather than O(n^2)
    xs = np.sort(x)
    n = xs.size
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    spread = float(2.0 * np.sum(weights * xs) / (n * n))
    return accuracy - 0.5 * spread


def brier_score(probability: float, occurred: bool) -> float:
    """Squared error of a probabilistic claim. Lower is better, 0.25 = coin flip."""
    return float((probability - (1.0 if occurred else 0.0)) ** 2)


def pinball_loss(samples: np.ndarray, actual: float, q: float) -> float:
    """Quantile loss — how well the ``q`` quantile is placed."""
    prediction = float(np.quantile(samples, q))
    delta = actual - prediction
    return float(max(q * delta, (q - 1.0) * delta))


def skill_score(model: float, baseline: float) -> float:
    """1 - model/baseline. Positive beats the baseline, negative is worse."""
    if baseline == 0:
        return 0.0
    return 1.0 - model / baseline


# -- forecasters --------------------------------------------------------------


def forecast(exchange, horizon: int = 250, runs: int = 100, seed: int = 0,
             free_fundamental: bool = True) -> Forecast:
    """Monte Carlo the market forward from its current state.

    ``free_fundamental=True`` (the default, and the only honest setting for
    a real forecast) detaches any historical fundamental path at the origin,
    so the simulated future is *generated* rather than read from data the
    forecaster should not have. Set it False only to study transmission of a
    known path, and don't call the result a forecast if you do.
    """
    origin_mid = exchange._mid()
    origin_step = exchange.step_count
    baseline_failures = len(exchange.defaults)

    terminal, drawdowns, failures, paths = [], [], [], []
    for run in range(runs):
        clone = copy.deepcopy(exchange)
        clone.rng = np.random.default_rng(seed + run)
        if free_fundamental:
            clone.fundamental_path = None
        start = len(clone.tape.mid)
        clone.run(horizon)
        path = np.asarray(clone.tape.mid[start:], dtype=float)
        if path.size == 0:
            continue
        paths.append(path)
        terminal.append(path[-1] / origin_mid - 1.0)
        drawdowns.append(max_drawdown(np.concatenate([[origin_mid], path])))
        failures.append(len(clone.defaults) - baseline_failures)

    stacked = np.vstack(paths) if paths else np.zeros((1, horizon))
    fan = {f"p{int(q * 100)}": np.quantile(stacked, q, axis=0)
           for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return Forecast(origin_step=origin_step, origin_mid=origin_mid,
                    horizon=horizon, runs=len(terminal),
                    terminal_returns=np.asarray(terminal),
                    drawdowns=np.asarray(drawdowns),
                    failures=np.asarray(failures), fan=fan)


def random_walk_forecast(returns, horizon: int = 250, runs: int = 100,
                         seed: int = 0, origin_mid: float = 100.0,
                         origin_step: int = 0) -> Forecast:
    """The null model: a driftless random walk at trailing volatility.

    This is the bar. An agent-based model with twenty-odd interacting
    institutions and a funding network ought to beat a Gaussian random walk
    fitted to recent volatility. If it doesn't, that is the finding, and
    :func:`walk_forward` will say so.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sigma = float(np.std(r)) if r.size > 1 else 0.01
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma, size=(runs, horizon))
    paths = origin_mid * np.exp(np.cumsum(steps, axis=1))

    terminal = paths[:, -1] / origin_mid - 1.0
    drawdowns = np.array([
        max_drawdown(np.concatenate([[origin_mid], path])) for path in paths])
    fan = {f"p{int(q * 100)}": np.quantile(paths, q, axis=0)
           for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return Forecast(origin_step=origin_step, origin_mid=origin_mid,
                    horizon=horizon, runs=runs, terminal_returns=terminal,
                    drawdowns=drawdowns, failures=np.zeros(runs), fan=fan,
                    label="random-walk")


# -- evaluation ---------------------------------------------------------------


@dataclass
class Scorecard:
    """Out-of-sample forecast performance against a baseline.

    ``crps_skill`` is the headline. Positive means the agent model carried
    information the random walk did not. Zero or negative means it did not,
    and no amount of mechanistic realism elsewhere changes that.
    """

    origins: int
    horizon_months: int
    model_crps: float
    baseline_crps: float
    model_brier: float
    baseline_brier: float
    actuals: list[float] = field(default_factory=list)
    model_medians: list[float] = field(default_factory=list)
    coverage_90: float = float("nan")

    @property
    def crps_skill(self) -> float:
        return skill_score(self.model_crps, self.baseline_crps)

    @property
    def brier_skill(self) -> float:
        return skill_score(self.model_brier, self.baseline_brier)

    @property
    def directional_accuracy(self) -> float:
        """Fraction of origins where the sign of the median call was right."""
        if not self.actuals:
            return float("nan")
        hits = [(m > 0) == (a > 0)
                for m, a in zip(self.model_medians, self.actuals)]
        return float(np.mean(hits))

    @property
    def base_rate(self) -> float:
        """Fraction of the sampled outcomes that were positive."""
        if not self.actuals:
            return float("nan")
        return float(np.mean([a > 0 for a in self.actuals]))

    @property
    def naive_directional_accuracy(self) -> float:
        """What you'd score by always predicting the more common direction.

        The comparison that stops directional accuracy from being a lie.
        Equities rise in most 12-month windows, so a model that says "up"
        every single time scores extremely well while knowing nothing. Any
        honest directional claim has to clear *this*, not 50%.
        """
        rate = self.base_rate
        return max(rate, 1.0 - rate)

    @property
    def directional_edge(self) -> float:
        """Directional accuracy above always-guess-the-majority. Can be negative."""
        return self.directional_accuracy - self.naive_directional_accuracy

    def verdict(self) -> str:
        if self.crps_skill > 0.05:
            return "beats the random-walk baseline"
        if self.crps_skill < -0.05:
            return "WORSE than a random walk — do not use for forecasting"
        return "indistinguishable from a random walk"

    def summary(self) -> str:
        return "\n".join([
            f"walk-forward scorecard — {self.origins} origins, "
            f"{self.horizon_months} months ahead, out of sample",
            "",
            f"  CRPS  agent model      {self.model_crps:.4f}   (lower is better)",
            f"  CRPS  random walk      {self.baseline_crps:.4f}",
            f"  CRPS  skill            {self.crps_skill:+.1%}",
            "",
            f"  Brier P(down>10%) model    {self.model_brier:.4f}",
            f"  Brier P(down>10%) baseline {self.baseline_brier:.4f}",
            f"  Brier skill                {self.brier_skill:+.1%}",
            "",
            f"  directional accuracy   {self.directional_accuracy:.0%}",
            f"  always-guess-majority  {self.naive_directional_accuracy:.0%}  "
            f"<- the number to beat, NOT 50%",
            f"  directional edge       {self.directional_edge:+.0%}",
            "",
            f"  90% interval coverage  {self.coverage_90:.0%}  (target 90%; "
            f"below target = overconfident)",
            "",
            f"  VERDICT: {self.verdict()}",
        ])


def walk_forward(series, build=None, origins: int = 12, warmup_steps: int = 400,
                 horizon_steps: int = 200, months_ahead: int = 12,
                 runs: int = 40, seed: int = 0, min_history: int = 120,
                 **build_kwargs) -> Scorecard:
    """Forecast from many historical origins using only prior data, then score.

    At each origin the model is shown history up to that point and nothing
    after it: the fundamental path is truncated at the origin and detached
    before forecasting. The realised outcome ``months_ahead`` later is then
    compared against both the agent model and a random walk fitted to the
    same trailing window.

    This is the only function in the repository that can honestly be said to
    test predictive skill, and it is deliberately unforgiving.
    """
    from exchange_sim.engine import systemic_market
    from exchange_sim.history import to_fundamental_path

    build = build or systemic_market
    closes = np.asarray(series.closes, dtype=float)
    last_origin = len(closes) - months_ahead - 1
    if last_origin <= min_history:
        raise ValueError(
            f"series too short: need more than {min_history + months_ahead} "
            f"points, got {len(closes)}")

    indices = np.linspace(min_history, last_origin, origins).astype(int)
    model_crps, base_crps, model_brier, base_brier = [], [], [], []
    actuals, medians, covered = [], [], []

    for n, index in enumerate(indices):
        history_closes = closes[:index + 1]
        actual = float(closes[index + months_ahead] / closes[index] - 1.0)

        # The model's intrinsic uncertainty is a free parameter, not a fact.
        # Left at its default the forecast fan is absurdly narrow and every
        # calibration number below is meaningless, so fit per-step volatility
        # to the trailing window — using only data available at this origin.
        trailing = history_closes[1:] / history_closes[:-1] - 1.0
        step_vol = _step_volatility(trailing, horizon_steps, months_ahead)

        # replay history up to the origin, then cut the future off
        path = to_fundamental_path(
            _as_series(series, history_closes, index), warmup_steps)
        exchange = build(seed=seed + n, start_price=100.0,
                         fundamental_path=path, fundamental_vol=step_vol,
                         **build_kwargs)
        exchange.run(warmup_steps)

        model = forecast(exchange, horizon=horizon_steps, runs=runs,
                         seed=seed + 1000 * n, free_fundamental=True)
        # the baseline gets exactly the same volatility estimate — the only
        # thing being compared is what the agent machinery adds on top of it
        baseline = random_walk_forecast(
            np.array([-step_vol, step_vol]),
            horizon=horizon_steps, runs=max(runs, 200), seed=seed + 7919 * n,
            origin_mid=exchange._mid())

        model_crps.append(crps(model.terminal_returns, actual))
        base_crps.append(crps(baseline.terminal_returns, actual))
        occurred = actual < -0.10
        model_brier.append(brier_score(model.prob_return_below(-0.10), occurred))
        base_brier.append(brier_score(baseline.prob_return_below(-0.10), occurred))

        lo, hi = model.interval(0.90)
        covered.append(lo <= actual <= hi)
        actuals.append(actual)
        medians.append(model.median())

    return Scorecard(
        origins=len(indices), horizon_months=months_ahead,
        model_crps=float(np.mean(model_crps)),
        baseline_crps=float(np.mean(base_crps)),
        model_brier=float(np.mean(model_brier)),
        baseline_brier=float(np.mean(base_brier)),
        actuals=actuals, model_medians=medians,
        coverage_90=float(np.mean(covered)))


def _as_series(template, closes, index):
    from exchange_sim.history import Series
    return Series(template.symbol, list(template.dates[:index + 1]),
                  [float(c) for c in closes], template.is_real, template.source)


def _step_volatility(monthly_returns, horizon_steps: int, months_ahead: int,
                     window: int = 60) -> float:
    """Per-step sigma such that ``horizon_steps`` carries ``months_ahead`` of risk.

    Both the model and the baseline are given this same number, so the
    comparison isolates what the agent machinery contributes rather than
    rewarding whichever one happened to be tuned to the right variance.
    """
    r = np.asarray(monthly_returns, dtype=float)
    r = r[np.isfinite(r)][-window:]
    sigma_month = float(np.std(r)) if r.size > 1 else 0.04
    sigma_total = sigma_month * np.sqrt(months_ahead)
    return sigma_total / np.sqrt(max(horizon_steps, 1))
