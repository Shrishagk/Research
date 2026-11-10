"""Plotting functions for the blind-spot simulation.

All plots are saved to the specified output directory as 300 dpi PNGs.
Uses a colorblind-safe palette (tab10 / viridis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Colorblind-safe qualitative palette (Wong, 2011 / tableau10)
COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# -----------------------------------------------------------------------
# 1. Heatmap: absorbed fraction vs (slope, lambda)
# -----------------------------------------------------------------------

def plot_absorption_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Heatmap of absorbed fraction vs ramp slope and forgetting factor."""
    pivot = summary.pivot(
        index="slope", columns="lambda", values="absorbed_fraction_mean",
    )
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="viridis", origin="lower",
        vmin=0, vmax=1,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.4f}" for v in pivot.index])
    ax.set_xlabel(r"Forgetting factor $\lambda$")
    ax.set_ylabel("Ramp slope (sensor units / hour)")
    ax.set_title("Absorbed fraction of injected bias")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absorbed fraction")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    _savefig(fig, out_dir / "absorption_heatmap.png")


# -----------------------------------------------------------------------
# 2. Illustrative time series
# -----------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    comp: np.ndarray,
    lam: float,
    slope: float,
    out_dir: Path,
) -> None:
    """Time series of raw, compensated, true signal, and injected attack."""
    t_hours = df["t"].values / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1 = axes[0]
    ax1.plot(t_hours, df["raw"].values, alpha=0.4, lw=0.5, label="Raw", color=COLORS[0])
    ax1.plot(t_hours, comp, lw=0.8, label="Compensated", color=COLORS[1])
    ax1.plot(t_hours, df["true_signal"].values, lw=1.0, ls="--", label="True signal", color=COLORS[2])
    ax1.axvline(t_hours[int(0.20 * len(t_hours))], color="grey", ls=":", label="Attack start")
    ax1.set_ylabel("Sensor units")
    ax1.set_title(f"Time series  (lam={lam}, slope={slope} units/h)")
    ax1.legend(fontsize=8, loc="upper left")

    ax2 = axes[1]
    ax2.plot(t_hours, df["attack"].values, lw=1.0, label="Injected attack", color=COLORS[3])
    residual_bias = comp - df["true_signal"].values
    ax2.plot(t_hours, residual_bias, lw=0.8, label="Visible bias (comp - true)", color=COLORS[4])
    ax2.axvline(t_hours[int(0.20 * len(t_hours))], color="grey", ls=":")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Sensor units")
    ax2.set_title("Injected bias vs. visible bias")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    _savefig(fig, out_dir / "time_series.png")


# -----------------------------------------------------------------------
# 3. Detection delay / detection rate vs slope
# -----------------------------------------------------------------------

def plot_detection(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Detection rate vs ramp slope, one line per (lambda, stream)."""
    lambdas = sorted(summary["lambda"].unique())
    slopes = sorted(summary["slope"].unique())

    for det_name in ("cusum", "page_hinkley"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax_i, stream in enumerate(("compensated", "residual", "static_comp")):
            ax = axes[ax_i]
            col = f"det_rate_{stream}_{det_name}"
            for k, l in enumerate(lambdas):
                sub = summary[summary["lambda"] == l].sort_values("slope")
                ax.plot(sub["slope"], sub[col], marker="o", ms=4,
                        label=f"lam={l:.3f}", color=COLORS[k % len(COLORS)])
            ax.set_xscale("log")
            ax.set_xlabel("Ramp slope (units/h)")
            ax.set_ylabel("Detection rate")
            ax.set_title(f"{det_name.upper()} on {stream.replace('_', ' ')}")
            ax.set_ylim(-0.05, 1.05)
            if ax_i == 0:
                ax.legend(fontsize=6, ncol=2, loc="lower right")
        fig.suptitle(f"Detection rate vs. ramp slope  ({det_name.upper()})", y=1.02)
        fig.tight_layout()
        _savefig(fig, out_dir / f"detection_rate_{det_name}.png")

        # Also plot detection delay
        fig_d, axes_d = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax_i, stream in enumerate(("compensated", "residual", "static_comp")):
            ax = axes_d[ax_i]
            col = f"delay_{stream}_{det_name}_mean"
            for k, l in enumerate(lambdas):
                sub = summary[summary["lambda"] == l].sort_values("slope")
                ax.plot(sub["slope"], sub[col], marker="o", ms=4,
                        label=f"lam={l:.3f}", color=COLORS[k % len(COLORS)])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Ramp slope (units/h)")
            ax.set_ylabel("Detection delay (steps)")
            ax.set_title(f"Delay: {det_name.upper()} on {stream.replace('_', ' ')}")
            if ax_i == 0:
                ax.legend(fontsize=6, ncol=2, loc="upper right")
        fig_d.suptitle(f"Detection delay vs. ramp slope  ({det_name.upper()})", y=1.02)
        fig_d.tight_layout()
        _savefig(fig_d, out_dir / f"detection_delay_{det_name}.png")


# -----------------------------------------------------------------------
# 4. Adaptation rate over time
# -----------------------------------------------------------------------

def plot_adaptation_rate(
    adapt_rate: np.ndarray,
    lam: float,
    slope: float,
    n_steps: int,
    out_dir: Path,
) -> None:
    """||Δθ|| over time during an attack."""
    t_hours = np.arange(len(adapt_rate)) / 60.0
    attack_start_h = t_hours[int(0.20 * n_steps)]

    fig, ax = plt.subplots(figsize=(10, 4))
    # Smooth for visibility
    window = min(200, len(adapt_rate) // 20)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(adapt_rate, kernel, mode="same")
    else:
        smoothed = adapt_rate
    ax.plot(t_hours, smoothed, lw=0.8, color=COLORS[0])
    ax.axvline(attack_start_h, color="grey", ls=":", label="Attack start")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("||delta theta||  (smoothed)")
    ax.set_title(f"Adaptation rate  (lam={lam}, slope={slope} units/h)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_dir / "adaptation_rate.png")
