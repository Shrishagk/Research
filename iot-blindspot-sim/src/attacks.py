"""Attack signal generators.

Each function returns an array of length ``n_steps`` representing the additive
attack signal injected into the raw sensor reading.
"""

from __future__ import annotations

import numpy as np


def no_attack(n_steps: int) -> np.ndarray:
    """No attack — baseline."""
    return np.zeros(n_steps)


def ramp_attack(
    n_steps: int,
    slope_per_hour: float,
    start_idx: int,
    dt_minutes: float = 1.0,
) -> np.ndarray:
    """Linearly ramping bias starting at *start_idx*.

    Parameters
    ----------
    n_steps : int
        Total number of time steps.
    slope_per_hour : float
        Bias increase per hour in sensor units.
    start_idx : int
        Step index at which the attack begins.
    dt_minutes : float
        Sampling interval in minutes (to convert slope to per-step).

    Returns
    -------
    np.ndarray
        Attack signal of shape ``(n_steps,)``.
    """
    attack = np.zeros(n_steps)
    slope_per_step = slope_per_hour * dt_minutes / 60.0
    idx = np.arange(n_steps)
    mask = idx >= start_idx
    attack[mask] = slope_per_step * (idx[mask] - start_idx)
    return attack


def abrupt_attack(
    n_steps: int,
    step_size: float,
    start_idx: int,
) -> np.ndarray:
    """Abrupt (step) bias starting at *start_idx*.

    Parameters
    ----------
    n_steps : int
        Total number of time steps.
    step_size : float
        Magnitude of the step in sensor units.
    start_idx : int
        Step index at which the step occurs.

    Returns
    -------
    np.ndarray
        Attack signal of shape ``(n_steps,)``.
    """
    attack = np.zeros(n_steps)
    attack[start_idx:] = step_size
    return attack

