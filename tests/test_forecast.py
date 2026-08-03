"""Tests for the forecaster and, more importantly, for its scorecard.

The forecaster is easy to get superficially right and easy to get
dishonestly right. These tests guard the second problem: that scoring rules
are actually proper, that the forecast can't secretly read the future, and
that directional accuracy is always reported next to the base rate it has
to beat.
"""
from __future__ import annotations

import numpy as np
import pytest

from exchange_sim.engine import systemic_market
from exchange_sim.forecast import (
    Scorecard,
    brier_score,
    crps,
    forecast,
    pinball_loss,
    random_walk_forecast,
    skill_score,
    walk_forward,
)
from exchange_sim.history import SyntheticEpisode


# -- scoring rules ------------------------------------------------------------


def test_crps_rewards_accuracy():
    accurate = np.random.default_rng(0).normal(0.0, 0.05, 500)
    biased = accurate + 0.5
    assert crps(accurate, 0.0) < crps(biased, 0.0)


def test_crps_rewards_sharpness_when_both_are_centred():
    rng = np.random.default_rng(1)
    sharp = rng.normal(0.0, 0.02, 2000)
    vague = rng.normal(0.0, 0.40, 2000)
    assert crps(sharp, 0.0) < crps(vague, 0.0)


def test_crps_punishes_overconfidence():
    """A sharp distribution in the wrong place must score worse than a wide
    one — otherwise the rule would reward bluffing."""
    rng = np.random.default_rng(2)
    confident_wrong = rng.normal(0.0, 0.01, 2000)
    honest_wide = rng.normal(0.0, 0.30, 2000)
    assert crps(confident_wrong, 0.60) > crps(honest_wide, 0.60)


def test_brier_score_bounds():
    assert brier_score(1.0, True) == pytest.approx(0.0)
    assert brier_score(0.0, True) == pytest.approx(1.0)
    assert brier_score(0.5, True) == pytest.approx(0.25)


def test_pinball_loss_is_minimised_at_the_true_quantile():
    samples = np.random.default_rng(3).normal(0.0, 1.0, 5000)
    true_q90 = float(np.quantile(samples, 0.9))
    assert pinball_loss(samples, true_q90, 0.9) < pinball_loss(samples, 0.0, 0.9)


def test_skill_score_signs():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)     # model better
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)     # no better
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)    # worse


# -- the forecaster -----------------------------------------------------------


def test_forecast_returns_a_usable_distribution():
    exchange = systemic_market(seed=1, fund_leverage=6)
    exchange.run(200)
    f = forecast(exchange, horizon=100, runs=20, seed=0)

    assert f.runs == 20
    assert f.terminal_returns.size == 20
    lo, hi = f.interval(0.90)
    assert lo <= f.median() <= hi
    assert 0.0 <= f.prob_any_failure() <= 1.0
    assert 0.0 <= f.prob_drawdown_worse_than(0.10) <= 1.0
    assert len(f.fan["p50"]) == 100


def test_forecast_does_not_mutate_the_exchange():
    exchange = systemic_market(seed=1, fund_leverage=6)
    exchange.run(200)
    before_step, before_mid = exchange.step_count, exchange._mid()
    forecast(exchange, horizon=50, runs=5, seed=0)
    assert exchange.step_count == before_step
    assert exchange._mid() == before_mid


def test_forecast_detaches_the_future_it_must_not_see():
    """The whole difference between a forecast and a replay.

    With ``free_fundamental=True`` the historical path must be cut off, so
    the simulated future is generated rather than read out of the input.
    """
    series = SyntheticEpisode(name="gfc", length=300, seed=1).fetch()
    from exchange_sim.history import to_fundamental_path

    path = to_fundamental_path(series, 600)
    exchange = systemic_market(seed=1, fundamental_path=path)
    exchange.run(200)
    assert exchange.fundamental_path is not None

    free = forecast(exchange, horizon=150, runs=12, seed=0, free_fundamental=True)
    fed = forecast(exchange, horizon=150, runs=12, seed=0, free_fundamental=False)

    # the crash is in the fed-in path; a forecast that cannot see it should
    # not reproduce it identically
    assert not np.allclose(free.terminal_returns, fed.terminal_returns)
    assert exchange.fundamental_path is not None      # original untouched


def test_random_walk_baseline_is_driftless_and_scales_with_vol():
    calm = random_walk_forecast(np.array([-0.001, 0.001]), horizon=200,
                                runs=800, seed=0)
    wild = random_walk_forecast(np.array([-0.02, 0.02]), horizon=200,
                                runs=800, seed=0)
    assert abs(calm.median()) < 0.02
    assert wild.terminal_returns.std() > calm.terminal_returns.std()


# -- the scorecard: the part that must not flatter the model ------------------


def test_directional_accuracy_is_reported_against_the_base_rate():
    """A model that always says "up" in an up-trending sample scores high
    while knowing nothing. The scorecard has to expose that."""
    card = Scorecard(origins=10, horizon_months=12, model_crps=1.0,
                     baseline_crps=1.0, model_brier=0.2, baseline_brier=0.2,
                     actuals=[0.1] * 9 + [-0.1],          # 90% up
                     model_medians=[0.05] * 10)           # always says up

    assert card.directional_accuracy == pytest.approx(0.9)
    assert card.base_rate == pytest.approx(0.9)
    assert card.naive_directional_accuracy == pytest.approx(0.9)
    assert card.directional_edge == pytest.approx(0.0)   # no edge at all
    assert "the number to beat" in card.summary()


def test_scorecard_verdict_calls_out_a_model_worse_than_the_baseline():
    worse = Scorecard(origins=5, horizon_months=12, model_crps=2.0,
                      baseline_crps=1.0, model_brier=0.3, baseline_brier=0.2)
    assert worse.crps_skill < 0
    assert "WORSE than a random walk" in worse.verdict()

    better = Scorecard(origins=5, horizon_months=12, model_crps=0.5,
                       baseline_crps=1.0, model_brier=0.1, baseline_brier=0.2)
    assert better.crps_skill > 0
    assert "beats" in better.verdict()

    same = Scorecard(origins=5, horizon_months=12, model_crps=1.0,
                     baseline_crps=1.0, model_brier=0.2, baseline_brier=0.2)
    assert "indistinguishable" in same.verdict()


def test_walk_forward_runs_out_of_sample_and_scores_both_models():
    series = SyntheticEpisode(name="bull", length=400, seed=0).fetch()
    card = walk_forward(series, origins=3, warmup_steps=120, horizon_steps=60,
                        months_ahead=12, runs=6, seed=0, min_history=120)

    assert card.origins == 3
    assert np.isfinite(card.model_crps)
    assert np.isfinite(card.baseline_crps)
    assert 0.0 <= card.coverage_90 <= 1.0
    assert len(card.actuals) == 3


def test_walk_forward_refuses_a_series_that_is_too_short():
    series = SyntheticEpisode(name="calm", length=60, seed=0).fetch()
    with pytest.raises(ValueError, match="series too short"):
        walk_forward(series, origins=3, months_ahead=12, min_history=120)
