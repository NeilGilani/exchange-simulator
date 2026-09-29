# 🏛️ exchange-simulator

**An agent-based market where realistic behaviour — and realistic failure — *emerges*.** Dealer banks, levered hedge funds, HFT market makers, pension funds and retail traders trade against each other on a real price-time priority matching engine (from [quantsim](https://github.com/NeilGilani/quantsim)), financed by a real funding network. Nobody programs the market to have fat tails, volatility clustering, flash crashes or deleveraging spirals — those emerge from the interaction, and the test suite verifies them statistically.

![Credit crisis: an 18% repricing with hedge funds targeting 20x leverage. Haircuts tighten, book depth collapses, then two funds fail.](docs/credit-crisis.png)

*An 18% fundamental repricing hits a market whose hedge funds target 20x leverage (measured leverage of the levered funds just before the shock: about 2.8x). Nothing after the red line is scripted. Haircuts rise to 25%, resting book depth falls from ~836 shares (pre-shock average) to 99, the price trades as much as 25% below its pre-shock level, and 26 steps later two momentum funds run out of equity and fail — writing 9,559 off against both dealers, split pro-rata. With a 3x leverage target the same shock produces no failures and no write-offs.*

---

## ⚠️ Read this before anything else

This simulator **cannot predict crashes, and does not try to.** If you came here hoping for something that tells you when SPY will fall, this is not it, and you should be sceptical of anything that claims to be — a model that reliably called crashes would stop working the moment anyone traded on it.

Three specific reasons it can't:

1. **The information isn't in the input.** The 2007 trigger was subprime loan performance, ratings-model error and off-balance-sheet exposure nobody could see. A model fed price data contains none of that.
2. **Reliable crash-timing is self-destroying.** Anything that worked would be arbitraged away as soon as it was used.
3. **The best-funded attempts don't do it either.** The Fed, Bank of England and ECB all run agent-based systemic-risk models, with regulatory data you and I can't get. **None of them forecast crisis timing.** They do scenario analysis and fragility measurement, because that's what this class of model can honestly deliver.

**What it does instead** is answer the question those institutions actually ask, and the one an agent-based model can answer honestly:

> Given the state this market is in right now — this much leverage, this much resting depth, positions this crowded — how large a shock does it take before losses stop being absorbed and start cascading?

That is a statement about **fragility**, which is measurable. It is not a statement about what will happen, which isn't. See [`fragility.py`](exchange_sim/fragility.py).

---

## 🚀 Quickstart

```bash
pip install "exchange-simulator[plot] @ git+https://github.com/NeilGilani/exchange-simulator.git"

exchange-sim run --steps 4000 --seed 42 --plot market.png
exchange-sim run --scenario credit-crisis --leverage 20 --plot crisis.png
exchange-sim fragility --leverage 20 --plot fragility.png
exchange-sim replay --synthetic gfc --plot replay.png
exchange-sim forecast --plot fan.png        # a forecast...
exchange-sim backtest                       # ...and its measured skill
```

**No terminal?** Open [colab.research.google.com](https://colab.research.google.com), make a new notebook, and run one cell:

```python
!pip install -q "exchange-simulator[plot] @ git+https://github.com/NeilGilani/exchange-simulator.git"

from exchange_sim import systemic_market, shock_response
ex = systemic_market(seed=42, fund_leverage=20, base_haircut=0.10)
ex.run(300)
print(shock_response(ex, runs=8).summary())
```

---

## 🧱 The five layers

### 1. Balance sheets, leverage and default — [`balance.py`](exchange_sim/balance.py)

Every agent has cash, inventory, **debt** and **receivables**. Equity is what's left marked to market. An agent financed too thin gets a margin call and must sell; an agent whose equity goes negative **fails**, its book is liquidated into the order book at whatever price it will bear, and whatever it can't repay is **written off against its creditors** — who may then fail themselves. The cascade loop runs to a fixed point, so one failure can take down a chain.

Conservation still holds through all of it. Lending moves cash without creating it; a write-off destroys a *claim* (debt on one side, receivable on the other) and touches no cash. So across all agents:

```
total cash    invariant
total shares  invariant
total equity  == total cash + total shares × mid      (claims cancel pairwise)
```

### 2. The institutional zoo — [`institutions.py`](exchange_sim/institutions.py)

What separates these agents is not their signal — several trade on much the same view — but their **balance sheet** and their **funding**.

| Agent | Role | What it contributes |
|---|---|---|
| **DealerBank** | quotes markets *and* finances everyone else | the spread — and the haircut. When scared it does both at once: quotes wider **and** lends less |
| **LeveredHedgeFund** | momentum or value, run at 3–6x | forced selling. Sized to leverage, so a move against it forces sales regardless of what it believes |
| **HFTMarketMaker** | tightest quote, smallest size, runs flat | most of the visible depth in calm markets, none of it in a crash |
| **PensionFund** | enormous, unlevered, slow | the only reliable buyer in a crash — arriving long after the crash |
| **RetailTrader** | small, herds on returns, capitulates | panic supply at the bottom |
| **MarketMaker / NoiseTrader / MomentumTrader / FundamentalTrader** | the original four | microstructure: spread, background flow, amplification, anchoring |

The **funding channel** is meant to matter more than any single agent. `FundingDesk` translates dealer risk appetite into every borrower's maintenance margin, so a haircut moving from 20% to 34% should margin-call every fund financed at 4x, **simultaneously**. No agent decides to start a spiral.

> **Known issue.** `Exchange.step` passes the exchange-wide `RiskLimits` to `resolve_stress`, so the per-borrower limits `FundingDesk` writes are currently read only by the fragility headroom metric, not by margin calls. Wiring them through changes the headline credit-crisis run above (to no failures), so the fix and the updated figures belong in the same change.

### 3. The fragility gauge — [`fragility.py`](exchange_sim/fragility.py)

Freeze the market's current state, deep-copy it, hit the copy with a shock of size *s*, run it forward. Many times per shock size, across a ladder of sizes — plus an **unshocked control**, so the market's own wander isn't counted as damage the shock caused.

![Shock absorption curve: damage and cascade probability versus shock size](docs/fragility.png)

```
state: leverage 2.6x · depth 519 · crowding 70% · headroom +13.5%
control (no shock): median drawdown -7.3% over 400 steps

  shock  median dd  worst dd   excess  amplif  cascade
  2.0%      -8.3%    -10.5%     1.0%    0.51x       0%
  5.0%     -11.4%    -15.1%     4.1%    0.82x       0%
  8.0%     -13.6%    -14.8%     6.3%    0.79x       0%
 12.0%     -18.2%    -22.1%    10.9%    0.91x      12%
 18.0%     -25.5%    -32.2%    18.2%    1.01x      50%

absorbed cleanly up to : 8.0%
cascades from          : 18.0%
first to fail          : momofund2 (2), momofund1 (2), momofund0 (1)
```

Read the **bend**, not the level. Below 8% a shock is just a repricing. By 18% it takes out funds half the time. The gauge also names the weak link — which agent fails *first*, most often.

### 4. Historical replay — [`history.py`](exchange_sim/history.py)

Drive the fundamental from a **real price series** instead of a random walk, and watch the ecology trade through it:

```bash
exchange-sim replay --symbol spy.us --start 2007-01-01 --end 2009-06-30 --plot gfc.png
```

**Be clear about what this proves.** It's a test of the *mechanism* — does an ecology with this much leverage reproduce the drawdown and funding stress that actually showed up? It is **not** a forecast and not a strategy backtest. The model is being *handed* 2008 and asked what a levered market does with it. It is not discovering that 2008 was coming.

Replaying the real 2006–2009 S&P 500 (a −50.8% drawdown) through ecologies at different leverage:

| fund leverage | emergent drawdown | overshoot vs fundamental | failures |
|---|---|---|---|
| 1.5x | −52.3% | −1.5% | 1 |
| 3x | −49.9% | +0.9% | 3 |
| 6x | −100.0% | −49.2% | 10 |
| 20x | −100.0% | −49.2% | 12 |

The bottom two rows are **the model breaking, not a finding**. At 6x and above, a 50% fundamental decline leaves no equilibrium: forced selling clears the book and price pins to the tick floor, so −100.0% is a floor artefact rather than a market outcome. Trust the 1.5–3x rows; treat anything that hits exactly −100% as out of range. Reporting the failure mode rather than quietly showing you the 3x row is the point.

### 5. Forecasting, and its scorecard — [`forecast.py`](exchange_sim/forecast.py)

The model *can* produce a genuine forecast: a conditional probability distribution over future paths, generated with the future detached (`free_fundamental=True`), so it is falsifiable rather than read off an input.

```bash
exchange-sim forecast --leverage 6 --plot fan.png
```

```
  median return        +0.1%
  90% interval         -1.2% to +1.6%
  P(drawdown > 10%)    0%
  P(any agent fails)   0%
```

**And then you have to score it.** The forecaster and the scoring harness live in the same module deliberately — it should be hard to use one without meeting the other.

```bash
exchange-sim backtest --origins 14
```

Walk-forward over real S&P 500 history, 1950–2024. At each origin the model sees only data available then; the outcome 12 months later is scored with CRPS (a proper scoring rule) against a random walk fitted to the same trailing volatility:

```
  CRPS  agent model      0.2124   (lower is better)
  CRPS  random walk      0.1228
  CRPS  skill            -73.0%

  directional accuracy   86%
  always-guess-majority  93%  <- the number to beat, NOT 50%
  directional edge       -7%

  90% interval coverage  21%  (target 90%; below target = overconfident)

  VERDICT: WORSE than a random walk — do not use for forecasting
```

**Read that carefully, because it is the honest answer to "can it predict crashes?".**

- **CRPS skill −73%.** The agent model is substantially *worse* than a driftless random walk. Twenty-odd interacting institutions and a funding network added negative information.
- **Directional accuracy 86% looks great and is worthless.** Equities rose in 93% of the sampled 12-month windows, so always saying "up" scores 93%. The model scores 86%. Its directional edge is **−7%**. This is exactly the number a dishonest write-up would quote alone, which is why `Scorecard` refuses to print it without the base rate beside it.
- **90% coverage is 21%.** The fan chart is wildly overconfident: outcomes land outside its "90% interval" four times out of five.

Every forecast the CLI prints carries a pointer to this backtest. A fan chart looks authoritative whether or not it contains information, so the plot draws the random-walk baseline over the top and captions itself "NOT shown to beat a random walk".

**Why it fails is more interesting than that it fails.** The mechanism is sound — the model reproduces fat tails, clustering, margin spirals and contagion — but the thing that would actually move prices twelve months out is information the model does not contain and cannot generate: earnings, policy, credit conditions, war. A faithful mechanism fed no information forecasts nothing. That gap is not closable by making the mechanism better.

Use it for the counterfactual question it answers well (*how does this structure transmit a shock?*), not the predictive one it fails (*what shock is coming?*).

---

## 🔬 What the tests actually verify

**61 tests.** Three kinds, deliberately.

**Mechanism tests** build a tiny hand-made market and assert exactly what must happen — deterministic, and the ones that would catch a real accounting bug:

1. Lending moves cash without creating it; a lender can't lend what it doesn't have
2. A write-off destroys a claim and touches **no cash**
3. `total equity == total cash + total shares × mid`, before and after write-offs
4. An insolvent agent fails, is liquidated, and **its creditor absorbs the shortfall**
5. A thin-but-solvent borrower is margin-called and forced to sell — without defaulting
6. Cash and shares survive a full credit crisis: margin spirals, fire sales, write-offs
7. A failed agent's estate keeps hitting the book for steps afterwards, not all at once
8. The exchange clamps orders agents should never have sent (zero/negative/NaN prices and sizes) instead of crashing

**Emergent tests** run the full ecology and assert statistical properties across seeds — written loosely on purpose, checking the *direction* of an effect rather than pinning numbers that would make the suite fragile without making the model correct:

9. **Fat tails** — excess kurtosis of returns > 0
10. **Volatility clustering** — |returns| autocorrelated
11. **No free momentum** — raw return autocorrelation ≈ 0
12. **Anchoring** — median gap between price and fundamental stays bounded
13. **The flash-crash double experiment** — the same sell program with liquidity present (mild dip) and withdrawn (crash > 2%, V-recovery). The crash needs **both** ingredients — exactly the finding of the SEC/CFTC report on May 6, 2010
14. **Funding tightens as the market falls**, and dealers quote smaller when stressed
15. **Damage is monotone in shock size**, measured against an unshocked control
16. **A heavily levered, loosely financed system is at least as fragile** as a conservative one, under an identical shock

**Forecast-honesty tests** guard against the model flattering itself:

17. CRPS is **proper** — a sharp distribution in the wrong place scores worse than an honest wide one, so bluffing can't win
18. A forecast **cannot see the future**: `free_fundamental=True` detaches the historical path, and a forecast made with it differs from one made without
19. Forecasting **does not mutate** the exchange it forecasts from
20. `Scorecard` always reports directional accuracy **next to the base rate** it must beat, and its summary contains the words "the number to beat"
21. The verdict says "WORSE than a random walk" when skill is negative

Stylized facts 9–11 mirror the classic properties of asset returns (Cont, 2001).

---

## 🧪 Run your own experiments

The cleanest experiment in the repo: **same shock, same everything, vary only leverage.**

```python
from exchange_sim import systemic_market, leverage_sweep

results = leverage_sweep(
    lambda lev, seed: systemic_market(seed=seed, fund_leverage=lev, base_haircut=0.10),
    leverages=(3, 6, 12, 20), shock=0.18,
)
for lev, out in results.items():
    print(f"{lev:>2}x  cascade probability {out.cascade_probability:.0%}  "
          f"failures {out.failures}")
```

Or write a new agent — one `act()` method, drop it in, measure how it changes the market:

```python
from exchange_sim import Agent, OrderIntent

class PanicSeller(Agent):
    def act(self, view, rng):
        if view.drawdown < -0.05 and self.inventory > 0:
            return [OrderIntent("market", "sell", self.inventory)]
        return []
```

### Running against real data

`exchange-sim replay` fetches from [stooq](https://stooq.com) by default, which needs outbound network access. **Many sandboxes block this** — including Claude Code's default network policy. When that happens you get an error naming the host, *never* a silent fallback to synthetic data:

```
could not load data: could not reach stooq.com for 'spy.us': Tunnel connection failed: 403 Forbidden
```

Three ways round it:

```bash
exchange-sim replay --sp500 --start 2006-01-01 --end 2009-12-31   # real, via GitHub
exchange-sim replay --csv ./spy.csv                # any CSV with Date,Close columns
exchange-sim replay --synthetic gfc                # SYNTHETIC, and labelled as such
```

`--sp500` pulls **real** S&P 500 closes from a [public dataset repo](https://github.com/datasets/s-and-p-500), which usually works even in sandboxes that block market-data hosts — they still need GitHub to fetch code. Two caveats: the series is **monthly** (four years ≈ 48 points, so intra-month structure in the replay is interpolation artefact, not data), and it is consumed as a *fundamental*, not as a tape.

Synthetic episodes (`calm`, `bull`, `gfc`, `covid`, `bear2022`) are regime-*shaped* series for offline work. They are always marked `is_real=False` and print as `SYNTHETIC`. Calling one `"gfc"` describes its shape, not its provenance — a synthetic series quietly standing in for a real one is exactly how a model ends up "validated" against nothing.

---

## 📊 The other scenarios

```bash
exchange-sim run --scenario default        # the original four-archetype market
exchange-sim run --scenario systemic       # full institutional stack, credit live
exchange-sim run --scenario flash-crash    # 2010: liquidity withdrawal
exchange-sim run --scenario credit-crisis  # 2008: repricing into leverage
```

`flash-crash` and `credit-crisis` are the two different ways a market breaks. The first is a *liquidity* crisis — nobody borrows, nobody fails, the book just empties. The second is a *credit* crisis — the loss is somebody else's asset.

![Flash-crash scenario, seed 42: with liquidity providers withdrawn, price detaches from the fundamental in both directions inside the outage window](docs/flash-crash.png)

*Seed 42, the run in the figure: inside the outage window the price dips about 1.5% and then spikes about 6.5% above the fundamental. Across the test seeds the dislocation goes both ways; the mean trough is about −6%.*

---

## ✅ Tests

```bash
pip install -e ".[dev]" && pytest
```

## 📄 License

MIT. Simulated money only — though the lessons about leverage are uncomfortably real.
