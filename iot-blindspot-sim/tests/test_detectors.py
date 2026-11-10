"""Unit tests for change-point detectors (CUSUM and Page-Hinkley)."""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors import cusum, page_hinkley, calibrate_threshold, run_detector


class TestDetectors:
    def test_cusum_detects_step(self) -> None:
        """CUSUM should trigger quickly on a large positive step."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 0.1, size=n)
        # Inject step at 500
        x[500:] += 1.0

        res = cusum(x, threshold=2.0, drift=0.05)
        assert res.alarm_idx is not None
        assert 500 <= res.alarm_idx <= 520

    def test_page_hinkley_detects_ramp(self) -> None:
        """Page-Hinkley should detect a ramping drift."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(0, 0.1, size=n)
        # Inject ramp from 500
        x[500:] += 0.01 * np.arange(1500)

        res = page_hinkley(x, threshold=5.0, delta=0.005)
        assert res.alarm_idx is not None
        assert res.alarm_idx >= 500

    def test_calibration_avoids_immediate_false_alarms(self) -> None:
        """Calibrated threshold should not alarm on the first few hundred steps of clean noise."""
        rng = np.random.default_rng(123)
        clean = rng.normal(0, 0.1, size=10_000)
        h = calibrate_threshold(clean, "cusum", target_fa_interval=5000, drift=0.05)
        res = cusum(clean, threshold=h, drift=0.05)
        # Should not alarm immediately
        if res.alarm_idx is not None:
            assert res.alarm_idx > 500

    def test_run_detector_convenience(self) -> None:
        """run_detector returns delay from attack onset."""
        rng = np.random.default_rng(999)
        n = 6000
        attack_start = 1000
        clean = rng.normal(0, 0.1, size=n)
        attack = clean.copy()
        attack[attack_start:] += 1.0  # huge step

        delay, h = run_detector(clean, attack, attack_start, detector="cusum")
        assert delay is not None
        assert delay < 50

