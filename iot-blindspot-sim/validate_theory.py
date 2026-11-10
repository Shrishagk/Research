#!/usr/bin/env python
"""Validate the scalar RLS ramp law and report monitor-view detectability.

This controlled experiment deliberately removes environmental-regressor and
true-signal confounding.  It validates the analytical RLS special case before
the same ideas are interpreted in the full synthetic sensor model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.theory import absorbed_fraction, minimum_detectable_slope, output_ramp_gain


def scalar_rls_output(slope: float, lam: float, sigma: float, horizon: int, rng: np.random.Generator) -> np.ndarray:
    """Steady-gain scalar RLS, matching the pre-update output convention."""
    theta = 0.0
    out = np.empty(horizon)
    for i in range(horizon):
        y = slope * (i + 1) + rng.normal(0.0, sigma)
        out[i] = y - theta
        theta += (1.0 - lam) * out[i]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--out", type=Path, default=Path("results/theory"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A fixed nonzero slope makes the Monte Carlo check numerically stable.
    slope = 0.01
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(20261109)
    for lam in (0.90, 0.95, 0.98, 0.99, 0.995):
        traces = np.vstack([scalar_rls_output(slope, lam, args.sigma, args.horizon, rng) for _ in range(args.trials)])
        empirical = float(traces[:, -200:].mean() / (slope * np.arange(args.horizon - 199, args.horizon + 1).mean()))
        theory = float(output_ramp_gain(lam, args.horizon)[-200:].mean() /
                       np.arange(args.horizon - 199, args.horizon + 1).mean())
        row = {
            "lambda": lam,
            "empirical_visible_fraction": empirical,
            "theory_visible_fraction": theory,
            "absolute_error": abs(empirical - theory),
            "theory_absorbed_fraction_at_horizon": float(absorbed_fraction(lam, args.horizon)),
        }
        for view in ("output", "state", "peer"):
            row[f"s_min_{view}_per_sample"] = minimum_detectable_slope(
                view, lam, args.sigma, args.horizon,
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(args.out / "theory_validation.csv", index=False)
    max_error = result["absolute_error"].max()
    with (args.out / "theory_validation.md").open("w", encoding="utf-8") as f:
        f.write("# Scalar RLS ramp-law validation\n\n")
        f.write(f"Monte Carlo trials: {args.trials}; noise sigma: {args.sigma}; horizon: {args.horizon} samples.\n\n")
        f.write("The output response uses the simulator's pre-update prediction. The maximum empirical/theory visible-fraction error is ")
        f.write(f"{max_error:.5f}. Minimum slopes are one-sided matched-filter thresholds at alpha=0.05 and power=0.80.\n\n")
        f.write(result.to_markdown(index=False, floatfmt=".6g"))
        f.write("\n")
    print(result.to_string(index=False))
    print(f"\nWrote {args.out / 'theory_validation.csv'} and theory_validation.md")


if __name__ == "__main__":
    main()
