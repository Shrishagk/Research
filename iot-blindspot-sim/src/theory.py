"""Closed-form benchmark for the intercept-dominated RLS blind spot.

The full simulator has time-varying environmental regressors.  This module
isolates the mechanism by which its intercept can learn a slow bias.  It is
therefore both a testable special case and a lower-complexity theory baseline,
not a claim that every multivariate RLS configuration has identical dynamics.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def output_ramp_gain(lam: float, horizon: int) -> np.ndarray:
    """Mean compensated-output response per unit *per-sample* ramp slope.

    For ``b_t = s t`` and the steady-state scalar RLS/EWMA update
    ``theta_t = lam theta_(t-1) + (1-lam) y_t``, the simulator's pre-update
    output is ``y_t - theta_(t-1)``.  Its attack mean is

        s (1 - lam**t) / (1 - lam).

    ``t`` starts at one because the attack value at onset is zero.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("this steady-state result requires lam in (0, 1)")
    t = np.arange(1, horizon + 1, dtype=float)
    return -np.expm1(t * np.log(lam)) / (1.0 - lam)


def absorbed_fraction(lam: float, t: np.ndarray | float) -> np.ndarray:
    """Expected fraction of instantaneous ramp magnitude absorbed by RLS.

    At sample ``t`` after onset, ``A_t = 1 - e_t/(s t)``.  Hence

        A_t = 1 - (1-lam**t) / ((1-lam)t).

    Noise has zero effect on this expectation; it affects the uncertainty and
    detectability through :func:`minimum_detectable_slope`.
    """
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr <= 0):
        raise ValueError("t must be positive")
    return 1.0 - output_ramp_gain(lam, int(np.max(t_arr)))[t_arr.astype(int) - 1] / t_arr


def output_noise_covariance(lam: float, sigma: float, horizon: int) -> np.ndarray:
    """Exact finite-horizon covariance of the pre-update compensated output.

    The result assumes independent measurement noise with standard deviation
    ``sigma`` and a scalar RLS filter already at its steady gain ``1-lam``.
    It retains the output autocorrelation induced by adaptation.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    A = np.eye(horizon)
    for row in range(horizon):
        for col in range(row):
            A[row, col] = -(1.0 - lam) * lam ** (row - 1 - col)
    return sigma**2 * (A @ A.T)


def view_model(
    view: str, lam: float, sigma: float, horizon: int, peer_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(g, Sigma)`` for ``z = slope*g + N(0, Sigma)``.

    Views are: ``output`` (compensated stream), ``state`` (increments of the
    RLS intercept), and ``peer`` (raw sensor minus independent reference).
    """
    output_g = output_ramp_gain(lam, horizon)
    output_cov = output_noise_covariance(lam, sigma, horizon)
    if view == "output":
        return output_g, output_cov
    if view == "state":
        # Delta theta_t = (1-lam) times the pre-update innovation.
        return (1.0 - lam) * output_g, (1.0 - lam) ** 2 * output_cov
    if view == "peer":
        t = np.arange(1, horizon + 1, dtype=float)
        ref_sigma = sigma if peer_sigma is None else peer_sigma
        return t, (sigma**2 + ref_sigma**2) * np.eye(horizon)
    raise ValueError("view must be one of: output, state, peer")


def minimum_detectable_slope(
    view: str,
    lam: float,
    sigma: float,
    horizon: int,
    alpha: float = 0.05,
    power: float = 0.80,
    peer_sigma: float | None = None,
) -> float:
    """One-sided Gaussian matched-filter slope threshold (units/sample).

    For a known positive ramp shape, the uniformly most powerful linear test
    has non-centrality ``s sqrt(g' Sigma^-1 g)``.  Requiring false-alarm rate
    ``alpha`` and power ``power`` gives

        s_min = [Phi^-1(1-alpha)+Phi^-1(power)] / sqrt(g' Sigma^-1 g).
    """
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    g, cov = view_model(view, lam, sigma, horizon, peer_sigma)
    info = float(g @ np.linalg.solve(cov, g))
    z = NormalDist().inv_cdf(1.0 - alpha) + NormalDist().inv_cdf(power)
    return z / np.sqrt(info)
