from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.theory import absorbed_fraction, minimum_detectable_slope, output_ramp_gain


def test_ramp_response_has_expected_plateau() -> None:
    gain = output_ramp_gain(0.99, 5000)
    np.testing.assert_allclose(gain[-1], 100.0, rtol=1e-12)


def test_absorption_increases_over_time() -> None:
    values = absorbed_fraction(0.99, np.array([1, 10, 100, 1000]))
    assert np.all(np.diff(values) > 0)
    assert values[-1] > 0.89


def test_state_view_has_same_information_as_output_when_noise_is_retained() -> None:
    out = minimum_detectable_slope("output", 0.99, 0.1, 500)
    state = minimum_detectable_slope("state", 0.99, 0.1, 500)
    np.testing.assert_allclose(state, out, rtol=1e-10)
