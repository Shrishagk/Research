"""Experiment sweeps and metrics.

Runs the sensor → RLS → detector pipeline across parameter grids and
collects absorption metrics and detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sensor import SensorConfig, generate_sensor_data
from .attacks import ramp_attack, no_attack
from .rls import RLSConfig, run_rls
from .detectors import run_detector


# ---------------------------------------------------------------------------
# Absorption metric
# ---------------------------------------------------------------------------

@dataclass
class AbsorptionMetrics:
    """Key metrics for one experiment run."""
    absorbed_fraction: float          # 1 - visible/injected  (final 20 %)
    steady_state_visible_ratio: float # visible/injected (final 20 %)
    lag: float                        # injected - visible bias (final 20 %)
    mean_visible_bias: float          # mean residual bias in final 20 %
    mean_injected_bias: float         # mean attack in final 20 %


def compute_absorption(
    comp: np.ndarray,
    true_signal: np.ndarray,
    attack: np.ndarray,
    attack_start: int,
) -> AbsorptionMetrics:
    """Compute how much of the injected bias the compensator absorbed.

    Measured over the final 20 % of the attack window.
    """
    n = len(comp)
    attack_len = n - attack_start
    tail_start = attack_start + int(0.8 * attack_len)

    residual_bias = comp[tail_start:] - true_signal[tail_start:]
    injected = attack[tail_start:]

    mean_vis = float(np.mean(residual_bias))
    mean_inj = float(np.mean(injected))

    if abs(mean_inj) < 1e-12:
        af = 0.0
        vr = 1.0
        lag = 0.0
    else:
        vr = mean_vis / mean_inj
        af = 1.0 - vr
        lag = mean_inj - mean_vis

    return AbsorptionMetrics(
        absorbed_fraction=af,
        steady_state_visible_ratio=vr,
        lag=lag,
        mean_visible_bias=mean_vis,
        mean_injected_bias=mean_inj,
    )


# ---------------------------------------------------------------------------
# Single-run pipeline
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Full result of a single experiment run."""
    seed: int
    slope: float
    lam: float
    absorption: AbsorptionMetrics
    absorption_static: Optional[AbsorptionMetrics] = None
    # Detection delays: dict[stream_name, dict[detector_name, delay|None]]
    detection: Dict[str, Dict[str, Optional[int]]] = None
    # For plotting (only stored when requested)
    df: Optional[pd.DataFrame] = None
    comp: Optional[np.ndarray] = None
    y_hat: Optional[np.ndarray] = None
    theta_history: Optional[np.ndarray] = None
    adapt_rate: Optional[np.ndarray] = None


def run_single(
    slope: float,
    lam: float,
    seed: int,
    n_steps: int = 20_000,
    sigma: float = 0.1,
    linear_only: bool = False,
    no_intercept: bool = False,
    freeze_static: bool = False,
    store_traces: bool = False,
) -> RunResult:
    """Execute one pipeline run with given parameters.

    Parameters
    ----------
    slope : float
        Ramp slope in sensor units per hour (0 → no attack).
    lam : float
        RLS forgetting factor.
    seed : int
        Random seed for reproducibility.
    n_steps : int
        Number of simulation steps.
    sigma : float
        Measurement noise std.
    linear_only : bool
        If True, omit the interaction term in RLS features.
    no_intercept : bool
        If True, omit the intercept term in RLS features.
    freeze_static : bool
        If True, freeze the compensator after the warm-up window.
    store_traces : bool
        If True, keep full time-series in the result (for plotting).

    Returns
    -------
    RunResult
    """
    warmup_frac = 0.20
    attack_start = int(warmup_frac * n_steps)

    # -- Sensor config --
    sensor_cfg = SensorConfig(
        n_steps=n_steps,
        sigma=sigma,
        seed=seed,
    )

    # -- Attack --
    if slope > 0:
        atk = ramp_attack(n_steps, slope, attack_start, sensor_cfg.dt_minutes)
    else:
        atk = no_attack(n_steps)

    # -- Generate data --
    df = generate_sensor_data(sensor_cfg, attack=atk)

    # -- RLS --
    rls_cfg = RLSConfig(
        lam=lam,
        include_interaction=not linear_only,
        include_intercept=not no_intercept,
        warmup_frac=warmup_frac,
        freeze_after_warmup=freeze_static,
    )
    comp, y_hat, theta_history, adapt_rate = run_rls(
        df, rls_cfg, T0=sensor_cfg.T0, H0=sensor_cfg.H0,
    )

    # -- Absorption --
    absorption = compute_absorption(
        comp, df["true_signal"].values, df["attack"].values, attack_start,
    )

    # -- Detection --
    # We also need a clean run (same seed, no attack) for calibration
    df_clean = generate_sensor_data(sensor_cfg, attack=no_attack(n_steps))
    rls_cfg_clean = replace(rls_cfg)
    comp_clean, y_hat_clean, _, _ = run_rls(
        df_clean, rls_cfg_clean, T0=sensor_cfg.T0, H0=sensor_cfg.H0,
    )

    # Static calibration baseline: freeze compensator after warmup
    rls_cfg_static = replace(rls_cfg, freeze_after_warmup=True)
    comp_static, y_hat_static, _, _ = run_rls(
        df, rls_cfg_static, T0=sensor_cfg.T0, H0=sensor_cfg.H0,
    )
    comp_static_clean, _, _, _ = run_rls(
        df_clean, replace(rls_cfg_static), T0=sensor_cfg.T0, H0=sensor_cfg.H0,
    )

    # Streams to test:
    # (a) compensated output, (b) raw residual (raw - y_hat), (c) static comp
    residual = df["raw"].values - y_hat  # same as comp (= raw - y_hat)
    residual_clean = df_clean["raw"].values - y_hat_clean

    streams = {
        "compensated": (comp_clean, comp),
        "residual": (residual_clean, residual),
        "static_comp": (comp_static_clean, comp_static),
    }

    detection: Dict[str, Dict[str, Optional[int]]] = {}
    for stream_name, (s_clean, s_attack) in streams.items():
        detection[stream_name] = {}
        for det_name in ("cusum", "page_hinkley"):
            delay, _ = run_detector(
                s_clean,
                s_attack,
                attack_start,
                detector=det_name,
                target_fa_interval=5000,
            )
            detection[stream_name][det_name] = delay

    # Static baseline absorption
    absorption_static = compute_absorption(
        comp_static, df["true_signal"].values, df["attack"].values, attack_start,
    )

    result = RunResult(
        seed=seed,
        slope=slope,
        lam=lam,
        absorption=absorption,
        absorption_static=absorption_static,
        detection=detection,
    )

    if store_traces:
        result.df = df
        result.comp = comp
        result.y_hat = y_hat
        result.theta_history = theta_history
        result.adapt_rate = adapt_rate

    return result


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(
    slopes: List[float],
    lambdas: List[float],
    seeds: List[int],
    n_steps: int = 20_000,
    sigma: float = 0.1,
    linear_only: bool = False,
    no_intercept: bool = False,
) -> pd.DataFrame:
    """Sweep over (slope, lambda, seed) and collect metrics.

    Returns
    -------
    pd.DataFrame
        One row per (slope, lambda, seed).
    """
    rows: list[dict] = []
    total = len(slopes) * len(lambdas) * len(seeds)
    done = 0

    for s in slopes:
        for l in lambdas:
            for seed in seeds:
                r = run_single(
                    slope=s, lam=l, seed=seed,
                    n_steps=n_steps, sigma=sigma,
                    linear_only=linear_only,
                    no_intercept=no_intercept,
                )
                row = {
                    "slope": s,
                    "lambda": l,
                    "seed": seed,
                    "absorbed_fraction": r.absorption.absorbed_fraction,
                    "steady_state_visible_ratio": r.absorption.steady_state_visible_ratio,
                    "lag": r.absorption.lag,
                    "mean_visible_bias": r.absorption.mean_visible_bias,
                    "mean_injected_bias": r.absorption.mean_injected_bias,
                    "absorbed_fraction_static": (
                        r.absorption_static.absorbed_fraction
                        if r.absorption_static is not None
                        else 0.0
                    ),
                }
                for stream_name, det_dict in r.detection.items():
                    for det_name, delay in det_dict.items():
                        row[f"delay_{stream_name}_{det_name}"] = delay
                rows.append(row)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}]")

    return pd.DataFrame(rows)


def summarise_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sweep results: mean, std, and 95 % CI over seeds.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sweep output from ``run_sweep``.

    Returns
    -------
    pd.DataFrame
        Grouped by (slope, lambda) with aggregated columns.
    """
    group_cols = ["slope", "lambda"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["seed"]]

    agg_rows = []
    for (s, l), grp in df.groupby(group_cols):
        row: dict = {"slope": s, "lambda": l, "n_seeds": len(grp)}
        for col in metric_cols:
            vals = grp[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{col}_std"] = vals.std() if len(vals) else np.nan
            if len(vals) >= 2:
                se = vals.std() / np.sqrt(len(vals))
                row[f"{col}_ci95_lo"] = vals.mean() - 1.96 * se
                row[f"{col}_ci95_hi"] = vals.mean() + 1.96 * se
            else:
                row[f"{col}_ci95_lo"] = np.nan
                row[f"{col}_ci95_hi"] = np.nan
        # Detection rate: fraction of seeds where detector fired
        for stream_name in ("compensated", "residual", "static_comp"):
            for det_name in ("cusum", "page_hinkley"):
                col = f"delay_{stream_name}_{det_name}"
                detected = grp[col].notna().sum()
                row[f"det_rate_{stream_name}_{det_name}"] = detected / len(grp)
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)

