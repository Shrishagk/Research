"""Forgetting-factor Recursive Least Squares (RLS) compensator.

Estimates and removes the environmental effect from the raw sensor reading
using an online adaptive filter with tunable forgetting factor lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class RLSConfig:
    """Configuration for the RLS compensator."""

    lam: float = 0.99          # forgetting factor (0.9 – 1.0)
    delta: float = 1000.0      # initial P = delta * I
    include_interaction: bool = True   # include temp*hum feature
    include_intercept: bool = True     # include bias/intercept term
    warmup_frac: float = 0.20  # fraction of data used for warm-up (no attack)
    freeze_after_warmup: bool = False  # if True, stop adapting after warm-up


def _build_phi(
    temp: float,
    hum: float,
    T0: float,
    H0: float,
    include_interaction: bool,
    include_intercept: bool,
) -> np.ndarray:
    """Construct the feature vector for one time step."""
    dt = temp - T0
    dh = hum - H0
    parts: list[float] = []
    if include_intercept:
        parts.append(1.0)
    parts.append(dt)
    parts.append(dh)
    if include_interaction:
        parts.append(dt * dh)
    return np.array(parts, dtype=np.float64)


def run_rls(
    df: pd.DataFrame,
    cfg: RLSConfig,
    T0: float = 25.0,
    H0: float = 55.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the RLS compensator over a sensor DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``raw``, ``temp``, ``hum``.
    cfg : RLSConfig
        Compensator settings.
    T0, H0 : float
        Reference temperature / humidity for feature construction.

    Returns
    -------
    comp : np.ndarray
        Compensated output (estimate of true_signal).
    y_hat : np.ndarray
        Predicted environmental effect at each step.
    theta_history : np.ndarray
        ``(n_steps, p)`` array of weight vectors over time.
    adapt_rate : np.ndarray
        ``||theta_t - theta_{t-1}||`` at each step.
    """
    n = len(df)
    raw = df["raw"].values
    temp = df["temp"].values
    hum = df["hum"].values

    # Determine feature dimension
    p = (1 if cfg.include_intercept else 0) + 2 + (1 if cfg.include_interaction else 0)

    # Initialise
    theta = np.zeros(p)
    P = cfg.delta * np.eye(p)
    lam = cfg.lam

    warmup_end = int(cfg.warmup_frac * n)

    comp = np.empty(n)
    y_hat = np.empty(n)
    theta_history = np.empty((n, p))
    adapt_rate = np.empty(n)

    for t in range(n):
        phi = _build_phi(
            temp[t], hum[t], T0, H0,
            cfg.include_interaction, cfg.include_intercept,
        )

        # Prediction
        y_hat_t = phi @ theta
        y_hat[t] = y_hat_t
        comp[t] = raw[t] - y_hat_t

        # Innovation
        e = raw[t] - y_hat_t

        # Decide whether to update
        frozen = cfg.freeze_after_warmup and t >= warmup_end

        if not frozen:
            # RLS gain
            Pphi = P @ phi
            denom = lam + phi @ Pphi
            K = Pphi / denom

            theta_prev = theta.copy()
            theta = theta + K * e

            # Update covariance, symmetrise for numerical stability
            P = (P - np.outer(K, phi @ P)) / lam
            P = 0.5 * (P + P.T)

            # Guard against blow-up: cap diagonal elements
            max_diag = 1e12
            diag = np.diag(P)
            if np.any(diag > max_diag):
                scale = np.minimum(max_diag / np.maximum(diag, 1e-30), 1.0)
                P = P * np.sqrt(np.outer(scale, scale))

            adapt_rate[t] = np.linalg.norm(theta - theta_prev)
        else:
            adapt_rate[t] = 0.0

        theta_history[t] = theta.copy()

    return comp, y_hat, theta_history, adapt_rate

