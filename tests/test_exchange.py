import numpy as np
import pytest

from exchange_sim import (
    Exchange,
    FundamentalTrader,
    Liquidator,
    MarketMaker,
    MomentumTrader,
    NoiseTrader,
    default_market,
    flash_crash,
    max_drawdown,
    stylized_facts,
)


# ---------------------------------------------------------------------------
# Accounting invariants — if these break, the exchange is "printing" assets
# ---------------------------------------------------------------------------

def test_cash_and_shares_are_conserved():
    exchange = default_market(seed=1)
    cash0, shares0 = exchange.total_cash(), exchange.total_shares()
    exchange.run(1_000)
    assert exchange.total_cash() == pytest.approx(cash0)
    assert exchange.total_shares() == pytest.approx(shares0)


def test_conservation_holds_through_a_flash_crash():
    exchange = flash_crash(seed=7, steps=2_000, crash_at=1_000, outage=150)
    # recompute initial totals from a fresh identical setup
    fresh = flash_crash(seed=7, steps=0, crash_at=1, outage=1)
    assert exchange.total_cash() == pytest.approx(fresh.total_cash())
    assert exchange.total_shares() == pytest.approx(fresh.total_shares())


def test_wealth_is_zero_sum_relative_to_the_market():
    """Trading redistributes wealth; marked at a common price it must sum to
    the same total as doing nothing (no wealth is created by shuffling)."""
    exchange = default_market(seed=3)
    mid0 = exchange.last_mid
    wealth0 = sum(agent.wealth(mid0) for agent in exchange.agents.values())
    exchange.run(500)
    wealth1 = sum(agent.wealth(mid0) for agent in exchange.agents.values())
    assert wealth1 == pytest.approx(wealth0)


# ---------------------------------------------------------------------------
# Market mechanics
# ---------------------------------------------------------------------------

def test_market_produces_a_two_sided_book_and_positive_spread():
    exchange = default_market(seed=5)
    exchange.run(300)
    assert exchange.book.best_bid() is not None
    assert exchange.book.best_ask() is not None
    assert exchange.book.spread() > 0
    spreads = np.array(exchange.tape.spread)
    assert np.nanmean(spreads) > 0


def test_determinism_same_seed_same_market():
    tape_a = default_market(seed=9).run(400)
    tape_b = default_market(seed=9).run(400)
    assert tape_a.mid == tape_b.mid
    assert tape_a.volume == tape_b.volume


def test_different_seeds_differ():
    tape_a = default_market(seed=1).run(400)
    tape_b = default_market(seed=2).run(400)
    assert tape_a.mid != tape_b.mid


def test_price_stays_anchored_to_fundamental():
    exchange = default_market(seed=11)
    tape = exchange.run(3_000)
    gap = np.abs(np.array(tape.mid) / np.array(tape.fundamental) - 1.0)
    assert np.median(gap) < 0.05  # fundamental traders keep price tethered


def test_duplicate_agent_names_rejected():
    with pytest.raises(ValueError):
        Exchange([NoiseTrader("dup"), NoiseTrader("dup")])


# ---------------------------------------------------------------------------
# Emergent stylized facts (Cont 2001) — the whole point of the project.
# Averaged over seeds so the assertions test the mechanism, not one lucky run.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def facts_across_seeds():
    all_facts = []
    for seed in (1, 2, 3):
        tape = default_market(seed=seed).run(3_000)
        all_facts.append(stylized_facts(tape.returns()))
    return {key: np.mean([f[key] for f in all_facts]) for key in all_facts[0]}


def test_returns_are_fat_tailed(facts_across_seeds):
    assert facts_across_seeds["excess_kurtosis"] > 0.5


def test_volatility_clusters(facts_across_seeds):
    assert facts_across_seeds["vol_clustering_lag1"] > 0.05


def test_raw_returns_are_nearly_uncorrelated(facts_across_seeds):
    assert abs(facts_across_seeds["raw_autocorr_lag1"]) < 0.15


# ---------------------------------------------------------------------------
# The flash crash scenario
# ---------------------------------------------------------------------------

def test_flash_crash_is_deeper_than_baseline_and_recovers():
    crashes, baselines = [], []
    for seed in (7, 11, 23):
        crashed = flash_crash(seed=seed, steps=2_600, crash_at=1_500, outage=200)
        mids = np.array(crashed.tape.mid)
        pre = mids[1_499]
        crash_window_dd = mids[1_500:1_760].min() / pre - 1.0
        crashes.append(crash_window_dd)
        # same seed, no scenario
        baseline = default_market(seed=seed).run(2_600)
        baselines.append(max_drawdown(np.array(baseline.mid)[1_500:1_760]))
        # V-shape: price ends within a few percent of pre-crash level
        assert abs(mids[-1] / pre - 1.0) < 0.10
    assert np.mean(crashes) < -0.02          # a real gap down, on average
    assert np.mean(crashes) < np.mean(baselines) - 0.01  # deeper than calm markets


def test_liquidity_softens_the_same_sell_program():
    """The identical sell program is materially milder when liquidity
    providers stay in the market — the crash needs BOTH ingredients
    (big seller + liquidity withdrawal), which is the 2010 lesson."""
    with_liquidity, without_liquidity = [], []
    for seed in (7, 11, 23):
        exchange = default_market(seed=seed)
        exchange.agents["liq"] = Liquidator("liq", start_step=1_500,
                                            chunk=150, total=15_000)
        tape = exchange.run(2_600)
        mids = np.array(tape.mid)
        with_liquidity.append(mids[1_500:1_760].min() / mids[1_499] - 1.0)

        crashed = flash_crash(seed=seed, steps=2_600, crash_at=1_500, outage=200)
        crashed_mids = np.array(crashed.tape.mid)
        without_liquidity.append(
            crashed_mids[1_500:1_760].min() / crashed_mids[1_499] - 1.0
        )
    # same seller, same seeds: providers present => meaningfully shallower dip
    assert np.mean(with_liquidity) > np.mean(without_liquidity) + 0.015


def test_offline_market_makers_submit_nothing():
    maker = MarketMaker("mm", cash=1_000)
    maker.offline_until = 100
    from exchange_sim.agents import MarketView
    view = MarketView(step=50, mid=100.0, best_bid=99.9, best_ask=100.1)
    assert maker.act(view, np.random.default_rng(0)) == []
