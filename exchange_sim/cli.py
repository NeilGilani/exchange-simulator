"""CLI: run a market, print its stylized facts, optionally chart it."""
from __future__ import annotations

import argparse

import numpy as np

from exchange_sim.engine import default_market, flash_crash
from exchange_sim.stats import max_drawdown, stylized_facts

BOLD, CYAN, DIM, RESET = "\033[1m", "\033[36m", "\033[2m", "\033[0m"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="exchange-sim",
        description="Agent-based market simulator on a real matching engine.",
    )
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=["default", "flash-crash"],
                        default="default")
    parser.add_argument("--crash-at", type=int, default=2_000)
    parser.add_argument("--outage", type=int, default=200,
                        help="flash-crash: steps liquidity providers stay away")
    parser.add_argument("--plot", metavar="PATH.png",
                        help="save a market chart (requires matplotlib)")
    args = parser.parse_args(argv)

    if args.scenario == "flash-crash":
        exchange = flash_crash(seed=args.seed, steps=args.steps,
                               crash_at=args.crash_at, outage=args.outage)
    else:
        exchange = default_market(seed=args.seed)
        exchange.run(args.steps)
    tape = exchange.tape

    mids = np.array(tape.mid)
    facts = stylized_facts(tape.returns())
    print(f"\n{BOLD}{CYAN}exchange-sim{RESET} — {args.scenario} · "
          f"{len(exchange.agents)} agents · {args.steps:,} steps · seed {args.seed}\n")
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
    print(f"{BOLD}Accounting{RESET} {DIM}(must be invariant){RESET}")
    print(f"  total cash       {exchange.total_cash():,.2f}")
    print(f"  total shares     {exchange.total_shares():,.0f}\n")

    if args.plot:
        from exchange_sim.plot import plot_market
        crash_kwargs = {}
        if args.scenario == "flash-crash":
            crash_kwargs = {"crash_at": args.crash_at, "outage": args.outage}
        plot_market(tape, args.plot,
                    title=f"exchange-sim — {args.scenario} (seed {args.seed})",
                    **crash_kwargs)
        print(f"chart saved to {args.plot}\n")


if __name__ == "__main__":
    main()
