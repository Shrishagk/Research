#!/usr/bin/env python
"""Single entry-point for the IoT blind-spot simulation sweep.

Usage
-----
    python run_sweep.py                       # defaults
    python run_sweep.py --seeds 5 --out results/quick
    python run_sweep.py --linear-only --no-intercept --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure `src` is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.experiments import run_sweep, run_single, summarise_sweep
from src.plots import (
    plot_absorption_heatmap,
    plot_time_series,
    plot_detection,
    plot_adaptation_rate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoT blind-spot simulation: sweep over attack slopes and RLS forgetting factors.",
    )
    parser.add_argument("--n-steps", type=int, default=20_000, help="Simulation length (steps, 1 step = 1 min)")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds per cell")
    parser.add_argument("--sigma", type=float, default=0.1, help="Measurement noise std")
    parser.add_argument("--out", type=str, default="results", help="Output directory for CSV and PNG files")
    parser.add_argument("--linear-only", action="store_true", help="Omit interaction term in RLS features")
    parser.add_argument("--no-intercept", action="store_true", help="Omit intercept term in RLS features")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Parameter grid ---
    slopes = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    lambdas = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    seeds = list(range(args.seeds))

    print(f"Sweep: {len(slopes)} slopes x {len(lambdas)} lambdas x {len(seeds)} seeds "
          f"= {len(slopes) * len(lambdas) * len(seeds)} runs")
    print(f"  n_steps={args.n_steps}, sigma={args.sigma}")
    print(f"  linear_only={args.linear_only}, no_intercept={args.no_intercept}")
    print(f"  Output -> {out_dir.resolve()}\n")

    t0 = time.time()
    raw_df = run_sweep(
        slopes=slopes,
        lambdas=lambdas,
        seeds=seeds,
        n_steps=args.n_steps,
        sigma=args.sigma,
        linear_only=args.linear_only,
        no_intercept=args.no_intercept,
    )
    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.1f} s")

    # --- Save raw + summary ---
    raw_df.to_csv(out_dir / "sweep_raw.csv", index=False)
    summary = summarise_sweep(raw_df)
    summary.to_csv(out_dir / "sweep.csv", index=False)
    print(f"Saved sweep_raw.csv ({len(raw_df)} rows) and sweep.csv ({len(summary)} rows)")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_absorption_heatmap(summary, out_dir)
    plot_detection(summary, out_dir)

    # Illustrative time-series for a "bad" setting
    demo_slope = 0.1
    demo_lam = 0.99
    demo_seed = 0
    r = run_single(
        slope=demo_slope, lam=demo_lam, seed=demo_seed,
        n_steps=args.n_steps, sigma=args.sigma,
        linear_only=args.linear_only, no_intercept=args.no_intercept,
        store_traces=True,
    )
    if r.df is not None and r.comp is not None:
        plot_time_series(r.df, r.comp, demo_lam, demo_slope, out_dir)
    if r.adapt_rate is not None:
        plot_adaptation_rate(r.adapt_rate, demo_lam, demo_slope, args.n_steps, out_dir)

    # --- Text summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Where does the compensator absorb > 50 %?
    high_abs = summary[summary["absorbed_fraction_mean"] > 0.50]
    if len(high_abs) > 0:
        print(f"\nCompensator absorbs >50 % of bias in {len(high_abs)}/{len(summary)} (slope, lam) cells:")
        for _, row in high_abs.iterrows():
            print(f"  slope={row['slope']:.4f}  lam={row['lambda']:.3f}  "
                  f"absorbed={row['absorbed_fraction_mean']:.2f} +/- {row['absorbed_fraction_std']:.2f}")
    else:
        print("\nCompensator does NOT absorb >50 % of bias in any cell.")

    # Where do detectors fail on compensated output?
    print("\nDetector failure rate (detection rate < 0.5) on COMPENSATED output:")
    for det_name in ("cusum", "page_hinkley"):
        col = f"det_rate_compensated_{det_name}"
        fails = summary[summary[col] < 0.50]
        if len(fails):
            print(f"  {det_name}: {len(fails)}/{len(summary)} cells")
            for _, row in fails.iterrows():
                print(f"    slope={row['slope']:.4f}  lam={row['lambda']:.3f}  det_rate={row[col]:.2f}")
        else:
            print(f"  {det_name}: detectors succeed in all cells.")

    print(f"\nAll outputs in {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
