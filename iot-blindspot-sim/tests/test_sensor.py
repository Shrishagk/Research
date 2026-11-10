"""Unit tests for the synthetic sensor data generator."""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensor import SensorConfig, generate_sensor_data
from src.attacks import no_attack, ramp_attack, abrupt_attack


class TestSensorDeterminism:
    """Sensor output must be deterministic given the same seed."""

    def test_same_seed_produces_same_data(self) -> None:
        cfg = SensorConfig(n_steps=1000, seed=42)
        df1 = generate_sensor_data(cfg)
        df2 = generate_sensor_data(cfg)
        for col in ("temp", "hum", "true_signal", "env_effect", "noise", "raw"):
            np.testing.assert_array_equal(df1[col].values, df2[col].values,
                                          err_msg=f"Column {col} differs between runs")

    def test_different_seeds_differ(self) -> None:
        df1 = generate_sensor_data(SensorConfig(n_steps=1000, seed=1))
        df2 = generate_sensor_data(SensorConfig(n_steps=1000, seed=2))
        assert not np.allclose(df1["raw"].values, df2["raw"].values)


class TestSensorShape:
    """Output DataFrame shape and columns."""

    def test_correct_columns(self) -> None:
        df = generate_sensor_data(SensorConfig(n_steps=100, seed=0))
        expected = {"t", "temp", "hum", "true_signal", "env_effect", "noise", "attack", "raw"}
        assert set(df.columns) == expected

    def test_correct_length(self) -> None:
        for n in (100, 500, 2000):
            df = generate_sensor_data(SensorConfig(n_steps=n, seed=0))
            assert len(df) == n

    def test_humidity_clipped(self) -> None:
        df = generate_sensor_data(SensorConfig(n_steps=10_000, seed=7))
        assert df["hum"].min() >= 0.0
        assert df["hum"].max() <= 100.0


class TestSensorPhysics:
    """Sanity checks on the generated signals."""

    def test_raw_equals_sum(self) -> None:
        """raw = true_signal + env_effect + noise + attack."""
        cfg = SensorConfig(n_steps=500, seed=99)
        atk = ramp_attack(500, 0.1, 100)
        df = generate_sensor_data(cfg, attack=atk)
        expected_raw = df["true_signal"] + df["env_effect"] + df["noise"] + df["attack"]
        np.testing.assert_allclose(df["raw"].values, expected_raw.values, atol=1e-12)

    def test_attack_array_length_mismatch(self) -> None:
        cfg = SensorConfig(n_steps=100, seed=0)
        with pytest.raises(ValueError, match="Attack length"):
            generate_sensor_data(cfg, attack=np.zeros(50))


class TestAttacks:
    """Unit tests for attack generators."""

    def test_no_attack_all_zeros(self) -> None:
        a = no_attack(100)
        assert np.all(a == 0.0)

    def test_ramp_before_start_is_zero(self) -> None:
        a = ramp_attack(100, slope_per_hour=1.0, start_idx=50)
        assert np.all(a[:50] == 0.0)
        assert a[50] == 0.0  # exactly at start
        assert a[51] > 0.0

    def test_ramp_slope(self) -> None:
        """Check the per-step slope is correct."""
        a = ramp_attack(200, slope_per_hour=6.0, start_idx=0, dt_minutes=1.0)
        # 6 units/hour = 0.1 units/step (1 min steps)
        expected = 0.1 * np.arange(200)
        np.testing.assert_allclose(a, expected, atol=1e-12)

    def test_abrupt_shape(self) -> None:
        a = abrupt_attack(100, step_size=5.0, start_idx=30)
        assert np.all(a[:30] == 0.0)
        assert np.all(a[30:] == 5.0)

