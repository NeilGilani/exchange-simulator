"""Tests for the historical replay layer.

The behaviour most worth protecting here is not numerical — it is that
synthetic data can never quietly pass itself off as real, and that a
blocked network fails loudly instead of falling back to something
plausible-looking. A model "validated" against silently-substituted fake
data is worse than one that was never validated at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from exchange_sim.history import (
    CSVSource,
    DataUnavailable,
    Series,
    SyntheticEpisode,
    replay,
    replay_report,
    to_fundamental_path,
)


def test_synthetic_series_is_labelled_synthetic():
    series = SyntheticEpisode(name="gfc", length=300, seed=1).fetch()
    assert series.is_real is False
    assert "SYNTHETIC" in series.label()
    assert series.source.startswith("synthetic:")


def test_synthetic_episodes_are_deterministic():
    a = SyntheticEpisode(name="covid", length=200, seed=7).fetch()
    b = SyntheticEpisode(name="covid", length=200, seed=7).fetch()
    assert a.closes == b.closes


def test_unknown_episode_is_rejected():
    with pytest.raises(ValueError, match="unknown episode"):
        SyntheticEpisode(name="nonexistent").fetch()


def test_crash_episodes_actually_fall():
    calm = SyntheticEpisode(name="calm", length=600, seed=0).fetch()
    gfc = SyntheticEpisode(name="gfc", length=600, seed=0).fetch()
    assert min(gfc.closes) / gfc.closes[0] < min(calm.closes) / calm.closes[0]


def test_missing_csv_raises_rather_than_falling_back():
    with pytest.raises(DataUnavailable, match="could not read"):
        CSVSource("/nonexistent/path/to/prices.csv").fetch("spy")


def test_csv_without_close_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Volume\n2020-01-01,100\n")
    with pytest.raises(DataUnavailable, match="expected Date and Close"):
        CSVSource(str(path)).fetch("spy")


def test_csv_source_is_marked_real(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,Close\n2020-01-01,100.0\n2020-01-02,101.5\n"
                    "2020-01-03,\n2020-01-06,99.0\n")
    series = CSVSource(str(path)).fetch("spy")
    assert series.is_real is True
    assert series.closes == [100.0, 101.5, 99.0]      # blank row skipped
    assert len(series.dates) == len(series.closes)


def test_sp500_source_parses_its_own_column_name(tmp_path, monkeypatch):
    """The dataset's close column is ``SP500``, not ``Close``."""
    from exchange_sim import history

    payload = ("Date,SP500,Dividend\n"
               "2007-01-01,1424.16,27.0\n"
               "2008-01-01,1378.76,28.0\n"
               "2009-03-01,757.13,22.0\n")

    class FakeResponse:
        def read(self):
            return payload.encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(history.urllib.request, "urlopen",
                        lambda *a, **k: FakeResponse())
    series = history.SP500Source().fetch()

    assert series.is_real is True
    assert series.closes == [1424.16, 1378.76, 757.13]
    assert series.source == "github:datasets/s-and-p-500"


def test_sp500_source_raises_when_github_is_blocked(monkeypatch):
    from exchange_sim import history

    def boom(*a, **k):
        raise OSError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(history.urllib.request, "urlopen", boom)
    with pytest.raises(DataUnavailable, match="raw.githubusercontent.com"):
        history.SP500Source().fetch()


def test_fundamental_path_is_rebased_and_follows_the_series():
    series = SyntheticEpisode(name="bear2022", length=250, seed=2).fetch()
    path = to_fundamental_path(series, steps=1_000, start_price=100.0)

    assert len(path) == 1_000
    assert path[0] == pytest.approx(100.0)
    # same total return as the source series, up to interpolation
    assert path[-1] / path[0] == pytest.approx(series.closes[-1] / series.closes[0],
                                               rel=1e-6)


def test_series_slice_respects_bounds():
    series = Series("x", ["2020-01-01", "2020-06-01", "2021-01-01"],
                    [1.0, 2.0, 3.0], is_real=True, source="test")
    sliced = series.slice("2020-02-01", "2020-12-31")
    assert sliced.closes == [2.0]


def test_replay_tracks_the_fundamental_it_was_fed():
    series = SyntheticEpisode(name="gfc", length=300, seed=5).fetch()
    exchange = replay(series, steps=600, seed=1)

    mids = np.asarray(exchange.tape.mid)
    fundamentals = np.asarray(exchange.tape.fundamental)
    # the market is anchored to the path, not free-running
    assert np.corrcoef(mids, fundamentals)[0, 1] > 0.9
    assert np.median(np.abs(mids / fundamentals - 1.0)) < 0.15


def test_replay_report_flags_synthetic_input():
    series = SyntheticEpisode(name="calm", length=200, seed=0).fetch()
    exchange = replay(series, steps=300, seed=0)
    report = replay_report(exchange, series)
    assert "SYNTHETIC" in report
    assert "validates the machinery, not the model" in report


def test_replay_conserves_cash_and_shares():
    series = SyntheticEpisode(name="covid", length=250, seed=3).fetch()
    exchange = replay(series, steps=500, seed=2)
    mid = exchange._mid()
    assert exchange.total_equity(mid) == pytest.approx(
        exchange.total_cash() + exchange.total_shares() * mid)
