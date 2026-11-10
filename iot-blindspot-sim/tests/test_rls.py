"""Unit tests for the RLS compensator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rls import RLSConfig, run_rls
from src.sensor import SensorConfig, generate_sensor_data
from src.attacks import no_attack


class TestRLSConvergence:
    """On noise-free, attack-free data with λ=1, θ must converge to the true coefficients."""

    def test_converges_to_true_coefficients(self) -> None:
        """With σ=0 and no attack, RLS(λ=1) should recover the true env coefficients."""
        cfg = SensorConfig(
            n_steps=10_000,
            sigma=0.0,          # no measurement noise
            seed=123,
            a1=0.10,
            a2=0.05,
            a3=0.002,
            T0=25.0,
            H0=55.0,
        )
        df = generate_sensor_data(cfg, attack=no_attack(cfg.n_steps))

        rls_cfg = RLSConfig(
            lam=1.0,           # no forgetting → converge to LS solution
            delta=1000.0,
            include_interaction=True,
            include_intercept=True,
            warmup_frac=0.0,   # use all data for learning
            freeze_after_warmup=False,
        )

        comp, y_hat, theta_history, _ = run_rls(
            df, rls_cfg, T0=cfg.T0, H0=cfg.H0,
        )

        # Final theta: [intercept, a1, a2, a3]
        final_theta = theta_history[-1]
        # The intercept should be close to 0 (no constant env offset)
        # a1, a2, a3 should match cfg
        # Note: intercept may absorb some signal mean; check env coefficients
        assert final_theta[1] == pytest.approx(cfg.a1, abs=0.02), \
            f"a1: expected {cfg.a1}, got {final_theta[1]}"
        assert final_theta[2] == pytest.approx(cfg.a2, abs=0.02), \
            f"a2: expected {cfg.a2}, got {final_theta[2]}"
        assert final_theta[3] == pytest.approx(cfg.a3, abs=0.005), \
            f"a3: expected {cfg.a3}, got {final_theta[3]}"

    def test_compensated_output_close_to_true_signal(self) -> None:
        """Compensated output should approximate the true signal when noise is zero."""
        cfg = SensorConfig(
            n_steps=10_000,
            sigma=0.0,
            seed=456,
        )
        df = generate_sensor_data(cfg, attack=no_attack(cfg.n_steps))

        rls_cfg = RLSConfig(
            lam=1.0,
            delta=1000.0,
            include_interaction=True,
            include_intercept=True,
            warmup_frac=0.0,
            freeze_after_warmup=False,
        )

        comp, _, _, _ = run_rls(df, rls_cfg, T0=cfg.T0, H0=cfg.H0)

        # After initial transient (first 2000 steps), RMSE should be tiny
        tail = slice(2000, None)
        rmse = np.sqrt(np.mean((comp[tail] - df["true_signal"].values[tail]) ** 2))
        # RLS removes the env effect; residual includes any signal drift the
        # intercept partially absorbs, so allow a moderate tolerance.
        assert rmse < 0.15, f"RMSE after convergence = {rmse:.4f}, expected < 0.15"

    def test_linear_only_mode(self) -> None:
        """Linear-only mode should still work (but may not recover a3)."""
        cfg = SensorConfig(n_steps=5000, sigma=0.0, seed=789, a3=0.0)
        df = generate_sensor_data(cfg, attack=no_attack(cfg.n_steps))

        rls_cfg = RLSConfig(
            lam=1.0,
            include_interaction=False,
            include_intercept=True,
            warmup_frac=0.0,
        )
        comp, _, theta_history, _ = run_rls(df, rls_cfg, T0=cfg.T0, H0=cfg.H0)

        final_theta = theta_history[-1]
        # [intercept, a1, a2]
        assert final_theta[1] == pytest.approx(cfg.a1, abs=0.02)
        assert final_theta[2] == pytest.approx(cfg.a2, abs=0.02)

    def test_no_intercept_mode(self) -> None:
        """No-intercept mode: feature vector has no bias term."""
        cfg = SensorConfig(n_steps=5000, sigma=0.0, seed=321)
        df = generate_sensor_data(cfg, attack=no_attack(cfg.n_steps))

        rls_cfg = RLSConfig(
            lam=1.0,
            include_interaction=True,
            include_intercept=False,
            warmup_frac=0.0,
        )
        comp, _, theta_history, _ = run_rls(df, rls_cfg, T0=cfg.T0, H0=cfg.H0)

        # theta is [a1, a2, a3] — no intercept
        final_theta = theta_history[-1]
        assert len(final_theta) == 3

    def test_frozen_compensator_stops_adapting(self) -> None:
        """After warm-up, freeze_after_warmup=True should yield zero adaptation rate."""
        cfg = SensorConfig(n_steps=5000, sigma=0.1, seed=111)
        df = generate_sensor_data(cfg, attack=no_attack(cfg.n_steps))

        rls_cfg = RLSConfig(
            lam=0.99,
            warmup_frac=0.20,
            freeze_after_warmup=True,
        )
        _, _, theta_history, adapt_rate = run_rls(df, rls_cfg, T0=cfg.T0, H0=cfg.H0)

        warmup_end = int(0.20 * 5000)
        assert np.all(adapt_rate[warmup_end:] == 0.0)
        # Theta should be constant after warm-up
        assert np.allclose(
            theta_history[warmup_end:],
            theta_history[warmup_end],
        )
