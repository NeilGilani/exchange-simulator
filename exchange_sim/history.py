"""Replaying real history — and being honest about what that proves.

The simulator's fundamental value is normally a slow random walk. This
module lets you drive it from a **real price series** instead, so the agent
ecology trades through 2008, 2020 or 2022 and you can compare what emerges
against what actually happened.

What this is
    A test of the *mechanism*. Feed the model the path of real fundamentals
    and ask: does an ecology with this much leverage and this much liquidity
    reproduce the drawdown, the volatility clustering and the funding
    stress that showed up in reality? If it does, the mechanism is
    plausible. If it doesn't, the mechanism is wrong or incomplete.

What this is **not**
    A backtest of a trading strategy, and emphatically not a forecast. The
    model is being *told* the fundamental path. It is not discovering that
    2008 was coming — it is being handed 2008 and asked what a levered
    market does when handed 2008. Any apparent "prediction" here is the
    answer leaking in through the input.

Data sources are pluggable via :class:`DataSource`. Real market data has to
be fetched over the network, which many sandboxes (including Claude Code's
default network policy) block. When that happens you get a clear error
naming the host rather than a silent fallback to fake data, because a
synthetic series quietly standing in for a real one is exactly how a model
ends up "validated" against nothing.

For offline work, :class:`SyntheticEpisode` provides regime-shaped series
that are **clearly labelled synthetic** and never masquerade as real.
"""
from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class DataUnavailable(RuntimeError):
    """Raised when a real series could not be fetched.

    Deliberately *not* caught anywhere that would substitute synthetic data
    in its place. If you asked for SPY and the network is blocked, you get
    an error naming the host — not a plausible-looking fake.
    """


@dataclass
class Series:
    """A named price series with dates."""

    symbol: str
    dates: list[str]
    closes: list[float]
    #: True only for series that came from a real data source
    is_real: bool = False
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.closes)

    def slice(self, start: str | None = None, end: str | None = None) -> "Series":
        lo = 0 if start is None else next(
            (i for i, d in enumerate(self.dates) if d >= start), len(self.dates))
        hi = len(self.dates) if end is None else next(
            (i for i, d in enumerate(self.dates) if d > end), len(self.dates))
        return Series(self.symbol, self.dates[lo:hi], self.closes[lo:hi],
                      self.is_real, self.source)

    def returns(self) -> np.ndarray:
        c = np.asarray(self.closes, dtype=float)
        if c.size < 2:
            return np.zeros(0)
        return c[1:] / c[:-1] - 1.0

    def label(self) -> str:
        kind = "real" if self.is_real else "SYNTHETIC"
        span = f"{self.dates[0]}..{self.dates[-1]}" if self.dates else "empty"
        return f"{self.symbol} [{kind}, {self.source}] {span} ({len(self)} points)"


class DataSource(Protocol):
    """Anything that can produce a :class:`Series`."""

    def fetch(self, symbol: str) -> Series:  # pragma: no cover - protocol
        ...


class StooqSource:
    """Free daily closes from stooq.com. No API key required.

    Symbols follow stooq's convention: ``spy.us``, ``^spx``, ``aapl.us``.
    Requires outbound network access to ``stooq.com``.
    """

    host = "stooq.com"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    def fetch(self, symbol: str) -> Series:
        url = f"https://{self.host}/q/d/l/?s={symbol}&i=d"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            raise DataUnavailable(
                f"could not reach {self.host} for {symbol!r}: {exc}. "
                f"This sandbox may block outbound network access — see the "
                f"README section 'Running against real data'."
            ) from exc
        return _parse_csv(payload, symbol, source=f"stooq:{symbol}")


class CSVSource:
    """Read a series from a local CSV with ``Date`` and ``Close`` columns.

    The escape hatch when the network is blocked but you have real data:
    download it anywhere, drop the file next to the script, point at it.
    Series loaded this way are marked real, because they are.
    """

    def __init__(self, path: str):
        self.path = path

    def fetch(self, symbol: str = "") -> Series:
        try:
            with open(self.path, newline="") as handle:
                payload = handle.read()
        except OSError as exc:
            raise DataUnavailable(f"could not read {self.path}: {exc}") from exc
        return _parse_csv(payload, symbol or self.path, source=f"csv:{self.path}")


def _parse_csv(payload: str, symbol: str, source: str) -> Series:
    reader = csv.DictReader(io.StringIO(payload))
    if reader.fieldnames is None:
        raise DataUnavailable(f"{source}: empty response")
    columns = {name.lower().strip(): name for name in reader.fieldnames}
    if "close" not in columns or "date" not in columns:
        raise DataUnavailable(
            f"{source}: expected Date and Close columns, got {reader.fieldnames}")
    dates: list[str] = []
    closes: list[float] = []
    for row in reader:
        try:
            close = float(row[columns["close"]])
        except (TypeError, ValueError):
            continue           # holidays and blank rows
        if close <= 0:
            continue
        dates.append(row[columns["date"]])
        closes.append(close)
    if not closes:
        raise DataUnavailable(f"{source}: no usable rows")
    return Series(symbol=symbol, dates=dates, closes=closes, is_real=True,
                  source=source)


@dataclass
class SyntheticEpisode:
    """A regime-shaped series for offline work. **Not real data.**

    Every series it produces is marked ``is_real=False`` and prints as
    ``SYNTHETIC``. It exists so the replay machinery can be developed and
    tested without a network connection — not so results can be quoted.

    The shapes are caricatures of well-known regimes, built from a drift, a
    volatility level and an optional crash. Calling one ``"gfc"`` describes
    its *shape*, not its provenance.
    """

    name: str = "calm"
    length: int = 750
    seed: int = 0

    #: drift, daily vol, crash depth, crash start (fraction), crash length
    SHAPES: dict = field(default_factory=lambda: {
        "calm":     (0.0004, 0.007, 0.00, 0.00, 0),
        "bull":     (0.0012, 0.009, 0.00, 0.00, 0),
        "gfc":      (-0.0004, 0.021, 0.48, 0.35, 260),
        "covid":    (0.0003, 0.016, 0.33, 0.45, 24),
        "bear2022": (-0.0007, 0.013, 0.20, 0.20, 400),
    })

    def fetch(self, symbol: str = "") -> Series:
        if self.name not in self.SHAPES:
            raise ValueError(f"unknown episode {self.name!r}; "
                             f"choose from {sorted(self.SHAPES)}")
        drift, vol, depth, start_frac, crash_len = self.SHAPES[self.name]
        rng = np.random.default_rng(self.seed)
        steps = rng.normal(drift, vol, self.length)
        if depth > 0 and crash_len > 0:
            start = int(self.length * start_frac)
            end = min(self.length, start + crash_len)
            # spread the drawdown over the crash window, log-linearly
            per_step = np.log(1.0 - depth) / max(end - start, 1)
            steps[start:end] += per_step
            steps[start:end] += rng.normal(0.0, vol * 1.5, end - start)
        closes = 100.0 * np.exp(np.cumsum(steps))
        dates = [f"t{i:05d}" for i in range(self.length)]
        return Series(symbol=symbol or f"SYNTH-{self.name}", dates=dates,
                      closes=[float(c) for c in closes], is_real=False,
                      source=f"synthetic:{self.name}:seed{self.seed}")


def load(symbol: str = "spy.us", source: DataSource | None = None) -> Series:
    """Fetch a series. Defaults to real data from stooq.

    Raises :class:`DataUnavailable` rather than falling back to synthetic
    data, so a blocked network can never be mistaken for a successful load.
    """
    return (source or StooqSource()).fetch(symbol)


def to_fundamental_path(series: Series, steps: int, start_price: float = 100.0
                        ) -> list[float]:
    """Stretch a daily series across ``steps`` simulation steps.

    Each market step is far shorter than a trading day, so the daily path is
    interpolated in log space and rebased to ``start_price``. Interpolation
    smooths intraday structure the daily series never contained; the
    resulting shape is the *fundamental*, not a claim about the tape.
    """
    closes = np.asarray(series.closes, dtype=float)
    if closes.size < 2:
        raise ValueError("need at least two observations")
    source_grid = np.linspace(0.0, 1.0, closes.size)
    target_grid = np.linspace(0.0, 1.0, steps)
    path = np.exp(np.interp(target_grid, source_grid, np.log(closes)))
    return list(path / path[0] * start_price)


def replay(series: Series, steps: int = 1_500, seed: int | None = None,
           build=None, start_price: float = 100.0, **kwargs):
    """Run an agent ecology through a historical fundamental path.

    ``build(seed=..., start_price=..., fundamental_path=..., **kwargs)`` must
    return an :class:`~exchange_sim.engine.Exchange`; it defaults to
    :func:`~exchange_sim.engine.systemic_market` so the credit layer is
    live and the replay can show funding stress, not just prices.

    Remember what the output means. The fundamental is an input, so the
    model is not anticipating the drawdown — it is being handed one. The
    question the replay answers is *how a levered market transmits it*:
    whether the emergent price overshoots the fundamental, when haircuts
    tighten, and who fails.
    """
    from exchange_sim.engine import systemic_market

    build = build or systemic_market
    path = to_fundamental_path(series, steps, start_price=start_price)
    exchange = build(seed=seed, start_price=start_price,
                     fundamental_path=path, **kwargs)
    exchange.run(steps)
    return exchange


def replay_report(exchange, series: Series) -> str:
    """Compare what the market did against the fundamental it was fed."""
    from exchange_sim.stats import max_drawdown, stylized_facts

    mids = np.asarray(exchange.tape.mid)
    fundamentals = np.asarray(exchange.tape.fundamental)
    facts = stylized_facts(exchange.tape.returns())
    real_dd = max_drawdown(np.asarray(series.closes))
    sim_dd = max_drawdown(mids)
    fund_dd = max_drawdown(fundamentals)
    gap = np.abs(mids / fundamentals - 1.0)

    lines = [
        f"replay: {series.label()}",
        "",
        f"  input drawdown (the series)   {real_dd:+.1%}",
        f"  fundamental drawdown (fed in) {fund_dd:+.1%}",
        f"  emergent drawdown (the tape)  {sim_dd:+.1%}",
        f"  overshoot                     {sim_dd - fund_dd:+.1%}"
        f"   <- the model's contribution",
        "",
        f"  median |price - fundamental|  {np.median(gap):.2%}",
        f"  max    |price - fundamental|  {np.max(gap):.2%}",
        f"  excess kurtosis               {facts['excess_kurtosis']:+.2f}",
        f"  vol clustering (L1)           {facts['vol_clustering_lag1']:+.3f}",
        f"  failures                      {len(exchange.defaults)}",
        f"  margin calls                  {len(exchange.margin_calls)}",
    ]
    if not series.is_real:
        lines += ["", "  NOTE: synthetic input — this validates the machinery, "
                      "not the model."]
    return "\n".join(lines)
