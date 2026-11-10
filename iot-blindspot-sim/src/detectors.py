"""Change-point detectors: two-sided CUSUM and Page-Hinkley.

Each detector is calibrated on attack-free data to achieve a target
false-alarm rate, then run on attack data to measure detection delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CUSUM (two-sided)
# ---------------------------------------------------------------------------

@dataclass
class CUSUMResult:
    """Output of a CUSUM run."""
    alarm_idx: Optional[int]   # first alarm step (None if never triggered)
    S_pos: np.ndarray          # upper cumulative sum
    S_neg: np.ndarray          # lower cumulative sum


def cusum(
    x: np.ndarray,
    threshold: float,
    drift: float = 0.0,
) -> CUSUMResult:
    """Two-sided CUSUM on *x* (assumed zero-mean under H0).

    Parameters
    ----------
    x : np.ndarray
        Input stream (typically residuals).
    threshold : float
        Decision threshold *h*.
    drift : float
        Allowance / slack parameter *k* (often half the expected shift).

    Returns
    -------
    CUSUMResult
    """
    n = len(x)
    S_pos = np.zeros(n)
    S_neg = np.zeros(n)
    alarm_idx: Optional[int] = None

    for t in range(1, n):
        S_pos[t] = max(0.0, S_pos[t - 1] + x[t] - drift)
        S_neg[t] = max(0.0, S_neg[t - 1] - x[t] - drift)
        if alarm_idx is None and (S_pos[t] > threshold or S_neg[t] > threshold):
            alarm_idx = t

    return CUSUMResult(alarm_idx=alarm_idx, S_pos=S_pos, S_neg=S_neg)


# ---------------------------------------------------------------------------
# Page-Hinkley
# ---------------------------------------------------------------------------

@dataclass
class PageHinkleyResult:
    """Output of a Page-Hinkley run."""
    alarm_idx: Optional[int]
    PH: np.ndarray


def page_hinkley(
    x: np.ndarray,
    threshold: float,
    delta: float = 0.005,
) -> PageHinkleyResult:
    """Page-Hinkley test for upward shift.

    Parameters
    ----------
    x : np.ndarray
        Input stream.
    threshold : float
        Decision threshold *lambda*.
    delta : float
        Detection sensitivity parameter.

    Returns
    -------
    PageHinkleyResult
    """
    n = len(x)
    m_t = 0.0          # running mean accumulator
    sum_x = 0.0        # cumulative sum of x
    U_t = 0.0          # cumulative sum with allowance
    PH = np.zeros(n)
    min_U = 0.0
    alarm_idx: Optional[int] = None

    for t in range(n):
        sum_x += x[t]
        m_t = sum_x / (t + 1)
        U_t += x[t] - m_t - delta
        min_U = min(min_U, U_t)
        PH[t] = U_t - min_U
        if alarm_idx is None and PH[t] > threshold:
            alarm_idx = t

    return PageHinkleyResult(alarm_idx=alarm_idx, PH=PH)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    x_clean: np.ndarray,
    detector: str,
    target_fa_interval: int = 5000,
    drift: Optional[float] = None,
    delta: float = 0.005,
    tol: float = 0.05,
    max_iter: int = 80,
) -> float:
    """Binary-search for a threshold that yields roughly one false alarm
    per *target_fa_interval* steps on clean data *x_clean*.

    Parameters
    ----------
    x_clean : np.ndarray
        Attack-free residual stream for calibration.
    detector : str
        ``"cusum"`` or ``"page_hinkley"``.
    target_fa_interval : int
        Desired mean steps between false alarms.
    drift : float, optional
        CUSUM slack parameter. Defaults to 0.5 * std(x_clean) if None.
    delta : float
        Page-Hinkley detection sensitivity parameter.
    tol : float
        Acceptable relative error on the FA rate.
    max_iter : int
        Max binary search iterations.

    Returns
    -------
    float
        Calibrated threshold.
    """
    n = len(x_clean)
    if drift is None:
        drift = float(0.5 * np.std(x_clean))

    def _count_alarms(h: float) -> int:
        """Count alarms by running the detector on successive windows."""
        alarms = 0
        pos = 0
        while pos < n:
            seg = x_clean[pos:]
            if detector == "cusum":
                res = cusum(seg, threshold=h, drift=drift)
                idx = res.alarm_idx
            else:
                res = page_hinkley(seg, threshold=h, delta=delta)
                idx = res.alarm_idx
            if idx is None:
                break
            alarms += 1
            pos += idx + 1  # restart after alarm
        return alarms

    # Target number of alarms
    target_alarms = max(1, n // target_fa_interval)

    lo, hi = 0.001, 10.0 * np.std(x_clean) * np.sqrt(n)
    if hi < 1.0:
        hi = 10.0

    # Ensure hi is large enough to yield 0 alarms
    for _ in range(15):
        if _count_alarms(hi) == 0:
            break
        hi *= 2.0

    best_h = hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        a = _count_alarms(mid)
        if a > target_alarms:
            lo = mid
        elif a < target_alarms:
            hi = mid
        else:
            best_h = mid
            break
        best_h = mid

    return best_h


# ---------------------------------------------------------------------------
# Convenience: run detector with calibration
# ---------------------------------------------------------------------------

def run_detector(
    x_clean: np.ndarray,
    x_attack: np.ndarray,
    attack_start_idx: int,
    detector: str = "cusum",
    target_fa_interval: int = 5000,
    drift: Optional[float] = None,
    delta_ph: float = 0.005,
    max_horizon: Optional[int] = None,
) -> Tuple[Optional[int], float]:
    """Calibrate on clean data, then run on attack data starting at attack onset.

    Parameters
    ----------
    x_clean : np.ndarray
        Clean baseline stream.
    x_attack : np.ndarray
        Full stream with attack.
    attack_start_idx : int
        Step at which the attack begins.
    detector : str
        ``"cusum"`` or ``"page_hinkley"``.
    target_fa_interval : int
        Target interval between false alarms.
    drift : float, optional
        CUSUM drift/slack parameter.
    delta_ph : float
        Page-Hinkley slack parameter.
    max_horizon : int, optional
        Maximum allowed delay to count as successful detection.
        Defaults to target_fa_interval.

    Returns
    -------
    detection_delay : int or None
        Steps from ``attack_start_idx`` to alarm (None if never triggered
        or triggered after max_horizon).
    threshold : float
        The calibrated threshold used.
    """
    if max_horizon is None:
        max_horizon = target_fa_interval

    cal = x_clean[attack_start_idx:]
    mu_cal = float(np.mean(cal))
    std_cal = float(np.std(cal))
    if drift is None:
        drift_val = 0.5 * (std_cal if std_cal > 1e-6 else 1.0)
    else:
        drift_val = drift

    # Calibrate on clean post-warmup segment
    if detector == "cusum":
        h = calibrate_threshold(
            cal - mu_cal, detector,
            target_fa_interval=target_fa_interval,
            drift=drift_val,
        )
        # Test post-attack segment
        x_eval = x_attack[attack_start_idx:] - mu_cal
        res = cusum(x_eval, threshold=h, drift=drift_val)
    else:
        h = calibrate_threshold(
            cal, detector,
            target_fa_interval=target_fa_interval,
            delta=delta_ph,
        )
        x_eval = x_attack[attack_start_idx:]
        res = page_hinkley(x_eval, threshold=h, delta=delta_ph)

    alarm = res.alarm_idx
    if alarm is not None and alarm <= max_horizon:
        delay = alarm
    else:
        delay = None

    return delay, h

