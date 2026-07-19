"""Stylized-fact analysis: does the simulated market behave like a real one?

Real markets famously exhibit (Cont, 2001, "Empirical properties of asset
returns"):
  1. fat-tailed returns (excess kurtosis > 0)
  2. volatility clustering (|r_t| autocorrelated)
  3. near-zero autocorrelation of raw returns (no free momentum)
This module measures all three so the tests can assert the simulator earns
them — emergently, from agent interaction, not by construction.
"""
from __future__ import annotations

import numpy as np


def excess_kurtosis(returns: np.ndarray) -> float:
    r = np.asarray(returns) - np.mean(returns)
    variance = np.mean(r**2)
    if variance == 0:
        return 0.0
    return float(np.mean(r**4) / variance**2 - 3.0)


def autocorr(series: np.ndarray, lag: int = 1) -> float:
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    denom = float(np.dot(s, s))
    if denom == 0:
        return 0.0
    return float(np.dot(s[:-lag], s[lag:]) / denom)


def volatility_clustering(returns: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation of absolute returns — positive means calm follows
    calm and storm follows storm."""
    return autocorr(np.abs(returns), lag=lag)


def max_drawdown(mids: np.ndarray) -> float:
    mids = np.asarray(mids, dtype=float)
    peaks = np.maximum.accumulate(mids)
    return float((mids / peaks - 1.0).min())


def stylized_facts(returns: np.ndarray) -> dict:
    return {
        "excess_kurtosis": excess_kurtosis(returns),
        "vol_clustering_lag1": volatility_clustering(returns, 1),
        "vol_clustering_lag5": volatility_clustering(returns, 5),
        "raw_autocorr_lag1": autocorr(returns, 1),
        "daily_vol": float(np.std(returns)),
    }
