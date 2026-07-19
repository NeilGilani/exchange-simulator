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
