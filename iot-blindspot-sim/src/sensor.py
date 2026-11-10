"""Synthetic IoT sensor data generator.

Produces a DataFrame of environmental variables, a true signal of interest,
environmental coupling, noise, and an (optional) attack component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SensorConfig:
    """All tunables for the synthetic sensor."""

    n_steps: int = 20_000
    dt_minutes: float = 1.0  # sample interval

    # -- Temperature --
    temp_mean: float = 25.0           # °C baseline
    temp_daily_amp: float = 8.0       # daily sinusoid amplitude
    temp_seasonal_amp: float = 3.0    # slow seasonal drift amplitude
    temp_ar1_phi: float = 0.98        # AR(1) coefficient
    temp_ar1_sigma: float = 0.3       # AR(1) innovation std

    # -- Humidity --
    hum_mean: float = 55.0            # %RH baseline
    hum_temp_coupling: float = -0.6   # negative correlation w/ temp deviation
    hum_noise_sigma: float = 2.0      # additive noise std

    # -- True signal (quantity of interest) --
    signal_type: str = "random_walk"  # "random_walk" or "sinusoid"
    signal_rw_sigma: float = 0.005    # random-walk innovation std
    signal_sin_period_h: float = 48.0 # period in hours (sinusoid mode)
    signal_sin_amp: float = 1.0       # amplitude (sinusoid mode)

    # -- Environmental coupling coefficients --
    a1: float = 0.10   # linear temp
    a2: float = 0.05   # linear hum
    a3: float = 0.002  # interaction temp*hum  (mild nonlinearity)
    T0: float = 25.0   # reference temp
    H0: float = 55.0   # reference humidity

    # -- Measurement noise --
    sigma: float = 0.1  # Gaussian noise std on raw reading

    # -- RNG seed --
    seed: int = 42


def generate_sensor_data(
    cfg: SensorConfig,
    attack: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Generate synthetic sensor data.

    Parameters
    ----------
    cfg : SensorConfig
        Configuration dataclass.
    attack : array-like, optional
        Length-``cfg.n_steps`` additive attack signal.  Defaults to zeros.

    Returns
    -------
    pd.DataFrame
        Columns: t, temp, hum, true_signal, env_effect, noise, attack, raw.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps

    t = np.arange(n, dtype=np.float64)  # step indices (1-min each)
    t_hours = t * cfg.dt_minutes / 60.0

    # --- Temperature ---
    daily = cfg.temp_daily_amp * np.sin(2 * np.pi * t_hours / 24.0)
    seasonal = cfg.temp_seasonal_amp * np.sin(
        2 * np.pi * t_hours / (365.25 * 24.0)
    )
    ar1 = np.empty(n)
    ar1[0] = 0.0
    innovations = rng.normal(0, cfg.temp_ar1_sigma, size=n)
    for i in range(1, n):
        ar1[i] = cfg.temp_ar1_phi * ar1[i - 1] + innovations[i]
    temp = cfg.temp_mean + daily + seasonal + ar1

    # --- Humidity (negatively correlated with temp) ---
    hum_base = cfg.hum_mean + cfg.hum_temp_coupling * (temp - cfg.temp_mean)
    hum = hum_base + rng.normal(0, cfg.hum_noise_sigma, size=n)
    hum = np.clip(hum, 0.0, 100.0)

    # --- True signal ---
    if cfg.signal_type == "random_walk":
        true_signal = np.cumsum(rng.normal(0, cfg.signal_rw_sigma, size=n))
    elif cfg.signal_type == "sinusoid":
        true_signal = cfg.signal_sin_amp * np.sin(
            2 * np.pi * t_hours / cfg.signal_sin_period_h
        )
    else:
        raise ValueError(f"Unknown signal_type: {cfg.signal_type}")

    # --- Environmental effect ---
    dt = temp - cfg.T0
    dh = hum - cfg.H0
    env_effect = cfg.a1 * dt + cfg.a2 * dh + cfg.a3 * dt * dh

    # --- Noise ---
    noise = rng.normal(0, cfg.sigma, size=n)

    # --- Attack ---
    if attack is None:
        attack = np.zeros(n)
    else:
        attack = np.asarray(attack, dtype=np.float64)
        if attack.shape[0] != n:
            raise ValueError(
                f"Attack length {attack.shape[0]} != n_steps {n}"
            )

    # --- Raw reading ---
    raw = true_signal + env_effect + noise + attack

    return pd.DataFrame({
        "t": t,
        "temp": temp,
        "hum": hum,
        "true_signal": true_signal,
        "env_effect": env_effect,
        "noise": noise,
        "attack": attack,
        "raw": raw,
    })

