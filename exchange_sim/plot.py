"""Charts for the README and for exploring runs (requires matplotlib)."""
from __future__ import annotations

import numpy as np

from exchange_sim.engine import Tape
from exchange_sim.stats import stylized_facts


def _dark(fig, axes) -> None:
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(color="#21262d", linewidth=0.6)


def plot_market(tape: Tape, path: str, title: str = "simulated market",
                crash_at: int | None = None, outage: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_price, ax_vol, ax_hist) = plt.subplots(
        3, 1, figsize=(10, 8.5), gridspec_kw={"height_ratios": [3, 1, 2]}
    )
    _dark(fig, (ax_price, ax_vol, ax_hist))
    steps = np.arange(len(tape.mid))

    ax_price.plot(steps, tape.mid, color="#3fb950", linewidth=1.1, label="mid price")
    ax_price.plot(steps, tape.fundamental, color="#58a6ff", linewidth=1.0,
                  alpha=0.8, label="fundamental value")
    if crash_at is not None and outage is not None:
        ax_price.axvspan(crash_at, crash_at + outage, color="#f85149", alpha=0.18,
                         label="liquidity withdrawn")
    ax_price.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                    fontsize=9)
    ax_price.set_title(title, color="#e6edf3", fontsize=11)

    ax_vol.bar(steps, tape.volume, color="#a371f7", width=1.0, alpha=0.7)
    ax_vol.set_ylabel("volume", color="#8b949e", fontsize=8)

    returns = tape.returns()
    facts = stylized_facts(returns)
    r_std = (returns - returns.mean()) / (returns.std() or 1.0)
    bins = np.linspace(-8, 8, 81)
    ax_hist.hist(r_std, bins=bins, density=True, color="#3fb950", alpha=0.75,
                 label="simulated returns")
    x = np.linspace(-8, 8, 400)
    ax_hist.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="#58a6ff",
                 linewidth=1.2, label="normal distribution")
    ax_hist.set_yscale("log")
    ax_hist.set_ylim(1e-5, 1.5)
    ax_hist.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                   fontsize=9)
    ax_hist.set_title(
        f"fat tails on a log scale — excess kurtosis {facts['excess_kurtosis']:.1f}, "
        f"vol clustering {facts['vol_clustering_lag1']:+.2f}, "
        f"raw autocorr {facts['raw_autocorr_lag1']:+.2f}",
        color="#8b949e", fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_crisis(exchange, path: str, title: str = "credit crisis",
                shock_at: int | None = None) -> None:
    """Four panels telling the contagion story in order: price detaches from
    fundamental, funding tightens, depth evaporates, agents fail.

    The point of stacking them on a shared x-axis is that the *sequence* is
    the mechanism — haircuts move before the failures, and depth collapses
    before the worst of the price move.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tape = exchange.tape
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
    _dark(fig, axes)
    ax_price, ax_fund, ax_depth, ax_fail = axes
    steps = np.arange(len(tape.mid))

    ax_price.plot(steps, tape.mid, color="#3fb950", linewidth=1.1, label="mid price")
    ax_price.plot(steps, tape.fundamental, color="#58a6ff", linewidth=1.0,
                  alpha=0.85, label="fundamental value")
    if shock_at is not None:
        ax_price.axvline(shock_at, color="#f85149", linewidth=1.2, alpha=0.9,
                         label="fundamental shock")
    ax_price.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                    fontsize=9)
    ax_price.set_title(title, color="#e6edf3", fontsize=11)

    ax_fund.plot(steps, np.asarray(tape.haircut) * 100, color="#f0883e", linewidth=1.1)
    ax_fund.set_ylabel("haircut %", color="#8b949e", fontsize=8)
    ax_fund.set_title("funding tightens first", color="#8b949e", fontsize=8, loc="left")

    ax_depth.plot(steps, tape.book_depth, color="#a371f7", linewidth=1.0)
    ax_depth.set_ylabel("book depth", color="#8b949e", fontsize=8)
    ax_depth.set_title("then liquidity leaves", color="#8b949e", fontsize=8, loc="left")

    ax_fail.step(steps, tape.failures, color="#f85149", linewidth=1.3, where="post")
    ax_fail.set_ylabel("failures", color="#8b949e", fontsize=8)
    ax_fail.set_xlabel("step", color="#8b949e", fontsize=8)
    ax_fail.set_title("then agents fail", color="#8b949e", fontsize=8, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_fragility(report, path: str, title: str = "shock absorption") -> None:
    """The response curve: damage and cascade probability versus shock size.

    Read the bend, not the level. Below it a shock is a price move; above
    it, forced selling turns the shock into something larger than itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_dd, ax_prob) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    _dark(fig, (ax_dd, ax_prob))

    shocks = np.array([o.shock for o in report.outcomes]) * 100
    median = np.array([abs(o.median_drawdown) for o in report.outcomes]) * 100
    worst = np.array([abs(o.worst_drawdown) for o in report.outcomes]) * 100

    ax_dd.plot(shocks, median, color="#3fb950", marker="o", linewidth=1.3,
               label="median drawdown")
    ax_dd.fill_between(shocks, median, worst, color="#f85149", alpha=0.22,
                       label="median to worst case")
    ax_dd.plot(shocks, shocks, color="#8b949e", linestyle="--", linewidth=1.0,
               label="shock repriced 1:1 (no amplification)")
    if report.control is not None:
        ax_dd.axhline(abs(report.control.median_drawdown) * 100, color="#58a6ff",
                      linestyle=":", linewidth=1.1, label="unshocked control")
    ax_dd.set_ylabel("drawdown %", color="#8b949e", fontsize=8)
    ax_dd.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
                 fontsize=8)
    ax_dd.set_title(f"{title} — {report.state.summary()}", color="#e6edf3", fontsize=10)

    prob = np.array([o.cascade_probability for o in report.outcomes]) * 100
    ax_prob.bar(shocks, prob, color="#f0883e", width=max(shocks[0] * 0.6, 0.4),
                alpha=0.85)
    ax_prob.set_ylabel("cascade probability %", color="#8b949e", fontsize=8)
    ax_prob.set_xlabel("fundamental shock %", color="#8b949e", fontsize=8)
    ax_prob.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
