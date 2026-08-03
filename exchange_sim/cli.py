"""CLI: run a market, stress it, replay history, chart any of it."""
from __future__ import annotations

import argparse

import numpy as np

from exchange_sim.engine import (
    credit_crisis,
    default_market,
    flash_crash,
    systemic_market,
)
from exchange_sim.forecast import forecast, random_walk_forecast, walk_forward
from exchange_sim.fragility import measure_state, shock_response
from exchange_sim.stats import max_drawdown, stylized_facts

BOLD, CYAN, DIM, RED, RESET = (
    "\033[1m", "\033[36m", "\033[2m", "\033[31m", "\033[0m")


def _print_market(exchange, label: str, steps: int, seed: int) -> None:
    tape = exchange.tape
    mids = np.array(tape.mid)
    facts = stylized_facts(tape.returns())
    print(f"\n{BOLD}{CYAN}exchange-sim{RESET} — {label} · "
          f"{len(exchange.agents)} agents · {steps:,} steps · seed {seed}\n")
    print(f"{BOLD}Market{RESET}")
    print(f"  price            {mids[0]:,.2f} → {mids[-1]:,.2f}  "
          f"(range {mids.min():,.2f} – {mids.max():,.2f})")
    print(f"  mean spread      {np.nanmean(tape.spread):.3f}")
    print(f"  total volume     {sum(tape.volume):,.0f} shares")
    print(f"  max drawdown     {max_drawdown(mids) * 100:+.1f}%\n")
    print(f"{BOLD}Stylized facts{RESET} {DIM}(real markets: >0, >0, ≈0){RESET}")
    print(f"  excess kurtosis      {facts['excess_kurtosis']:+.2f}")
    print(f"  vol clustering (L1)  {facts['vol_clustering_lag1']:+.3f}")
    print(f"  raw autocorr (L1)    {facts['raw_autocorr_lag1']:+.3f}\n")

    if exchange.total_debt() > 0 or exchange.defaults or exchange.margin_calls:
        state = measure_state(exchange)
        print(f"{BOLD}Credit{RESET}")
        print(f"  outstanding debt {exchange.total_debt():,.0f}")
        print(f"  levered leverage {state.levered_leverage:.2f}x")
        print(f"  margin calls     {len(exchange.margin_calls):,}")
        print(f"  failures         {len(exchange.defaults)}")
        if exchange.defaults:
            depth = max(d.generation for d in exchange.defaults) + 1
            lost = sum(d.written_off for d in exchange.defaults)
            first = exchange.defaults[0]
            print(f"  cascade depth    {depth}   {DIM}(1 = isolated, "
                  f">1 = failures caused failures){RESET}")
            print(f"  written off      {lost:,.0f}")
            print(f"  first to fail    {first.agent} at step {first.step}")
        print()

    print(f"{BOLD}Accounting{RESET} {DIM}(must be invariant){RESET}")
    print(f"  total cash       {exchange.total_cash():,.2f}")
    print(f"  total shares     {exchange.total_shares():,.0f}\n")


def _cmd_run(args) -> None:
    if args.scenario == "flash-crash":
        exchange = flash_crash(seed=args.seed, steps=args.steps,
                               crash_at=args.crash_at, outage=args.outage)
    elif args.scenario == "credit-crisis":
        exchange = credit_crisis(seed=args.seed, steps=args.steps,
                                 shock_at=args.crash_at, shock_size=args.shock,
                                 fund_leverage=args.leverage,
                                 base_haircut=args.haircut)
    elif args.scenario == "systemic":
        exchange = systemic_market(seed=args.seed, fund_leverage=args.leverage,
                                   base_haircut=args.haircut)
        exchange.run(args.steps)
    else:
        exchange = default_market(seed=args.seed)
        exchange.run(args.steps)

    _print_market(exchange, args.scenario, args.steps, args.seed)

    if args.plot:
        if args.scenario == "credit-crisis":
            from exchange_sim.plot import plot_crisis
            plot_crisis(exchange, args.plot, shock_at=args.crash_at,
                        title=f"exchange-sim — credit crisis (seed {args.seed})")
        else:
            from exchange_sim.plot import plot_market
            crash_kwargs = ({"crash_at": args.crash_at, "outage": args.outage}
                            if args.scenario == "flash-crash" else {})
            plot_market(exchange.tape, args.plot,
                        title=f"exchange-sim — {args.scenario} (seed {args.seed})",
                        **crash_kwargs)
        print(f"chart saved to {args.plot}\n")


def _cmd_fragility(args) -> None:
    exchange = systemic_market(seed=args.seed, fund_leverage=args.leverage,
                               base_haircut=args.haircut)
    exchange.run(args.warmup)
    shocks = tuple(float(s) / 100 for s in args.shocks)
    report = shock_response(exchange, shocks=shocks, runs=args.runs,
                            horizon=args.horizon, seed=args.seed)
    print(f"\n{BOLD}{CYAN}exchange-sim fragility{RESET} — "
          f"{args.runs} runs x {len(shocks)} shocks, {args.horizon}-step horizon, "
          f"measured after {args.warmup} steps\n")
    print(report.summary())
    print(f"\n{DIM}This measures how much shock the market can currently absorb.")
    print(f"It is not a forecast, and nothing here predicts a crash.{RESET}\n")
    if args.plot:
        from exchange_sim.plot import plot_fragility
        plot_fragility(report, args.plot,
                       title=f"shock absorption (seed {args.seed})")
        print(f"chart saved to {args.plot}\n")


def _cmd_replay(args) -> None:
    from exchange_sim.history import (
        CSVSource,
        DataUnavailable,
        SP500Source,
        StooqSource,
        SyntheticEpisode,
        replay,
        replay_report,
    )

    if args.csv:
        source = CSVSource(args.csv)
    elif args.sp500:
        source = SP500Source()
    elif args.synthetic:
        source = SyntheticEpisode(name=args.synthetic, length=args.length,
                                  seed=args.seed)
    else:
        source = StooqSource()
    # --sp500 names its own series; --symbol only means anything to stooq
    symbol = "SP500" if args.sp500 else args.symbol
    try:
        series = source.fetch(symbol)
    except DataUnavailable as exc:
        print(f"\n{RED}could not load data{RESET}: {exc}\n")
        print(f"{DIM}Blocked? Try --sp500 (served from GitHub, which is often "
              f"allowed when market-data hosts are not), or --synthetic gfc "
              f"for a clearly-labelled "
              f"synthetic series, or download a CSV with Date/Close columns "
              f"and pass --csv PATH.{RESET}\n")
        raise SystemExit(2)

    if args.start or args.end:
        series = series.slice(args.start, args.end)
    exchange = replay(series, steps=args.steps, seed=args.seed,
                      fund_leverage=args.leverage, base_haircut=args.haircut)
    print()
    print(replay_report(exchange, series))
    print(f"\n{DIM}The fundamental path is an INPUT. The model is not "
          f"anticipating this drawdown —")
    print(f"it is being handed one, and showing how a levered market "
          f"transmits it.{RESET}\n")
    if args.plot:
        from exchange_sim.plot import plot_crisis
        plot_crisis(exchange, args.plot, title=f"replay — {series.label()}")
        print(f"chart saved to {args.plot}\n")


def _cmd_forecast(args) -> None:
    exchange = systemic_market(seed=args.seed, fund_leverage=args.leverage,
                               base_haircut=args.haircut)
    exchange.run(args.warmup)
    model = forecast(exchange, horizon=args.horizon, runs=args.runs,
                     seed=args.seed)
    baseline = random_walk_forecast(exchange.tape.returns(), horizon=args.horizon,
                                    runs=max(args.runs, 200), seed=args.seed,
                                    origin_mid=exchange._mid())
    print()
    print(model.summary())
    print(f"\n{DIM}same horizon, random-walk baseline: median "
          f"{baseline.median():+.1%}, 90% interval "
          f"{baseline.interval(0.9)[0]:+.1%} to {baseline.interval(0.9)[1]:+.1%}{RESET}")
    print(f"\n{RED}This distribution has NOT been shown to beat a random walk.{RESET}")
    print(f"{DIM}Run `exchange-sim backtest` to see the measured skill before "
          f"you trust any number above.{RESET}\n")
    if args.plot:
        from exchange_sim.plot import plot_forecast
        plot_forecast(model, args.plot, baseline=baseline,
                      title=f"forecast — {args.horizon} steps, {model.runs} paths")
        print(f"chart saved to {args.plot}\n")


def _cmd_backtest(args) -> None:
    from exchange_sim.history import (
        CSVSource, DataUnavailable, SP500Source, SyntheticEpisode,
    )
    if args.csv:
        source = CSVSource(args.csv)
    elif args.synthetic:
        source = SyntheticEpisode(name=args.synthetic, length=args.length,
                                  seed=args.seed)
    else:
        source = SP500Source()
    try:
        series = source.fetch("SP500")
    except DataUnavailable as exc:
        print(f"\n{RED}could not load data{RESET}: {exc}\n")
        raise SystemExit(2)
    if args.start or args.end:
        series = series.slice(args.start, args.end)

    print(f"\n{BOLD}{CYAN}exchange-sim backtest{RESET} — {series.label()}")
    print(f"{DIM}forecasting from each origin using only data available then"
          f"{RESET}\n")
    card = walk_forward(series, origins=args.origins, warmup_steps=args.warmup,
                        horizon_steps=args.horizon, months_ahead=args.months,
                        runs=args.runs, seed=args.seed,
                        fund_leverage=args.leverage, base_haircut=args.haircut)
    print(card.summary())
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exchange-sim",
        description="Agent-based market simulator with a real matching engine, "
                    "balance sheets and a funding network.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a market scenario")
    run.add_argument("--steps", type=int, default=4_000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--scenario", default="default",
                     choices=["default", "systemic", "flash-crash", "credit-crisis"])
    run.add_argument("--crash-at", type=int, default=2_000,
                     help="step at which the scenario's trigger fires")
    run.add_argument("--outage", type=int, default=200,
                     help="flash-crash: steps liquidity providers stay away")
    run.add_argument("--shock", type=float, default=0.15,
                     help="credit-crisis: size of the fundamental repricing")
    run.add_argument("--leverage", type=float, default=6.0,
                     help="target leverage for hedge funds")
    run.add_argument("--haircut", type=float, default=0.10,
                     help="dealers' calm-market haircut; lower = looser funding")
    run.add_argument("--plot", metavar="PATH.png", help="save a chart")
    run.set_defaults(func=_cmd_run)

    frag = sub.add_parser("fragility",
                          help="measure how large a shock this market absorbs")
    frag.add_argument("--seed", type=int, default=42)
    frag.add_argument("--warmup", type=int, default=300,
                      help="steps to run before measuring")
    frag.add_argument("--horizon", type=int, default=400,
                      help="steps to run after each shock")
    frag.add_argument("--runs", type=int, default=8, help="runs per shock size")
    frag.add_argument("--shocks", type=float, nargs="+",
                      default=[2, 5, 8, 12, 18], help="shock sizes in percent")
    frag.add_argument("--leverage", type=float, default=6.0)
    frag.add_argument("--haircut", type=float, default=0.10,
                      help="dealers' calm-market haircut; lower = looser funding")
    frag.add_argument("--plot", metavar="PATH.png")
    frag.set_defaults(func=_cmd_fragility)

    rep = sub.add_parser("replay", help="drive the fundamental from a real series")
    rep.add_argument("--symbol", default="spy.us", help="stooq symbol, e.g. spy.us")
    rep.add_argument("--csv", help="local CSV with Date/Close columns")
    rep.add_argument("--sp500", action="store_true",
                     help="real monthly S&P 500 from a GitHub dataset — often "
                          "reachable when market-data hosts are blocked")
    rep.add_argument("--synthetic", metavar="NAME",
                     help="use a SYNTHETIC episode instead of real data "
                          "(calm, bull, gfc, covid, bear2022)")
    rep.add_argument("--length", type=int, default=750,
                     help="synthetic series length")
    rep.add_argument("--start", help="ISO date to start from")
    rep.add_argument("--end", help="ISO date to end at")
    rep.add_argument("--steps", type=int, default=1_500)
    rep.add_argument("--seed", type=int, default=42)
    rep.add_argument("--leverage", type=float, default=6.0)
    rep.add_argument("--haircut", type=float, default=0.10,
                     help="dealers' calm-market haircut; lower = looser funding")
    rep.add_argument("--plot", metavar="PATH.png")
    rep.set_defaults(func=_cmd_replay)

    fc = sub.add_parser("forecast",
                        help="probability distribution over the next N steps")
    fc.add_argument("--seed", type=int, default=42)
    fc.add_argument("--warmup", type=int, default=300)
    fc.add_argument("--horizon", type=int, default=250)
    fc.add_argument("--runs", type=int, default=100)
    fc.add_argument("--leverage", type=float, default=6.0)
    fc.add_argument("--haircut", type=float, default=0.10)
    fc.add_argument("--plot", metavar="PATH.png")
    fc.set_defaults(func=_cmd_forecast)

    bt = sub.add_parser("backtest",
                        help="score forecasts against real history, out of sample")
    bt.add_argument("--csv", help="local CSV with Date/Close columns")
    bt.add_argument("--synthetic", metavar="NAME")
    bt.add_argument("--length", type=int, default=750)
    bt.add_argument("--start", default="1950-01-01")
    bt.add_argument("--end", default="2024-12-31")
    bt.add_argument("--origins", type=int, default=14)
    bt.add_argument("--months", type=int, default=12,
                    help="forecast horizon in months of real time")
    bt.add_argument("--warmup", type=int, default=350)
    bt.add_argument("--horizon", type=int, default=180)
    bt.add_argument("--runs", type=int, default=40)
    bt.add_argument("--seed", type=int, default=0)
    bt.add_argument("--leverage", type=float, default=6.0)
    bt.add_argument("--haircut", type=float, default=0.10)
    bt.set_defaults(func=_cmd_backtest)

    # bare `exchange-sim --steps N` still works, as it did before subcommands
    if argv is None:
        import sys
        argv = sys.argv[1:]
    argv = list(argv)
    if not argv or argv[0] not in {"run", "fragility", "replay",
                                   "forecast", "backtest", "-h", "--help"}:
        argv = ["run"] + argv

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
