# IoT Blind-Spot Simulator

**Hypothesis:** An adaptive (forgetting-factor RLS) environmental compensator
deployed on a low-cost IoT sensor can gradually absorb a slowly ramping
malicious bias into its learned parameters.  If a change-point detector
monitors only the compensated output, it may fail to raise an alarm because
the compensator masks the drift.  This simulator tests whether — and under
what conditions — this "blind spot" emerges.

> **Note:** This is an exploratory research tool.  The blind spot is stated as
> a hypothesis, not an established fact.  The sweep results may show that the
> effect is strong, weak, or absent depending on parameter choices.  Interpret
> the plots honestly.

---

## Repository structure

```
iot-blindspot-sim/
  README.md              ← you are here
  requirements.txt       ← numpy, scipy, matplotlib, pandas, pytest
  run_sweep.py           ← single entry point (argparse, seeds)
  src/
    sensor.py            ← synthetic data generator (SensorConfig dataclass)
    rls.py               ← forgetting-factor RLS compensator
    attacks.py           ← attack generators (ramp, abrupt, none)
    detectors.py         ← CUSUM and Page-Hinkley detectors with calibration
    experiments.py       ← sweep logic, absorption metrics, single-run pipeline
    plots.py             ← all plot functions (heatmap, time series, detection, adaptation)
  tests/
    test_rls.py          ← RLS convergence and correctness tests
    test_sensor.py       ← sensor determinism, shape, and physics tests
  results/               ← CSV + PNG outputs (created at runtime)
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run the full sweep (default: 20 seeds × 56 cells ≈ few minutes)
python run_sweep.py

# Quick test run
python run_sweep.py --seeds 3 --n-steps 5000

# All CLI options
python run_sweep.py --help
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-steps` | 20000 | Simulation length (1 step = 1 minute) |
| `--seeds` | 20 | Random seeds per (slope, lambda) cell |
| `--sigma` | 0.1 | Measurement noise std |
| `--out` | `results` | Output directory |
| `--linear-only` | off | Omit temp×hum interaction in RLS features |
| `--no-intercept` | off | Omit intercept term in RLS features |

## What each module does

### sensor.py
Generates synthetic IoT sensor data with:
- **Temperature:** daily sinusoid + seasonal drift + AR(1) noise
- **Humidity:** negatively correlated with temperature, clipped to [0, 100]
- **True signal:** random walk or low-frequency sinusoid
- **Environmental coupling:** `f(T, H) = a1*(T-T0) + a2*(H-H0) + a3*(T-T0)*(H-H0)`
- **Noise:** Gaussian with configurable std
- **Attack:** additive, injected after warm-up

Raw reading = true_signal + env_effect + noise + attack

### rls.py
Recursive Least Squares with forgetting factor λ ∈ [0.9, 1.0]:
- Features: `[1, ΔT, ΔH, ΔT·ΔH]` (intercept and interaction are optional)
- Online update of weights θ and covariance P
- Numerically stabilised (P symmetrised, diagonal capping)
- 20% warm-up window with no attack for initial calibration
- Can be frozen after warm-up (static compensator baseline)

### attacks.py
- **ramp:** linear drift starting at attack onset, slope in sensor units/hour
- **abrupt:** step change at attack onset
- **none:** zero baseline

### detectors.py
- **CUSUM (two-sided):** cumulative sum detector
- **Page-Hinkley:** change detection for gradual shifts
- Threshold auto-calibrated on attack-free data to target ~1 false alarm per 5000 steps
- Run on three streams: (a) compensated output, (b) residual, (c) static-compensator baseline

### experiments.py
- `run_single()`: full pipeline for one (slope, λ, seed) triple
- `run_sweep()`: parameter grid sweep
- **Absorption metric** over final 20% of attack window:
  - `absorbed_fraction = 1 - mean(visible_bias) / mean(injected_bias)`
- Detection delay: steps from attack start to first alarm

## How to read the plots

### 1. `absorption_heatmap.png`
- **Axes:** ramp slope (y) vs forgetting factor λ (x)
- **Color:** fraction of injected bias absorbed by the RLS compensator
- Values near 1.0 (yellow) = the compensator hides most of the attack
- Values near 0.0 (dark) = the attack passes through to the detector
- **Key insight:** if a region shows high absorption AND low detection rate, that is the hypothesised "blind spot"

### 2. `time_series.png`
- Top panel: raw sensor reading, compensated output, and true signal over time
- Bottom panel: injected attack vs visible bias (comp − true_signal)
- The gap between the injected and visible curves is the absorbed portion
- Vertical line marks attack onset (after 20% warm-up)

### 3. `detection_rate_cusum.png` / `detection_rate_page_hinkley.png`
- Detection rate (fraction of seeds where alarm fired) vs ramp slope
- Three subplots: compensated output, residual, static-compensator baseline
- One line per λ value
- If the "compensated" subplot shows low detection rate where the "static" subplot shows high detection rate, the adaptive compensator is indeed masking the attack

### 4. `adaptation_rate.png`
- `||Δθ||` (smoothed) over time — shows how fast the RLS weights change
- An uptick after attack onset suggests the compensator is actively absorbing the bias
- A flat line with `freeze_after_warmup=True` confirms the static baseline works

## Reproducing results

All runs are fully deterministic given a seed.  The default sweep uses seeds 0–19.
To reproduce exactly:

```bash
python run_sweep.py --seeds 20 --n-steps 20000 --sigma 0.1 --out results
```

### Theory check: scalar RLS ramp law

The full simulator is deliberately realistic enough that environmental and
true-signal variability confound an analytical claim.  The theory contribution
is therefore validated first in its exact special case: an intercept-only,
steady-gain forgetting-factor RLS (`0 < lambda < 1`) tracking a ramp `b_t = s t`.

For the pre-update output used by `run_rls`, after `t` attack samples,

```
E[comp_t - true_t] = s (1 - lambda^t)/(1 - lambda)
A_t = 1 - (1 - lambda^t)/((1 - lambda)t)
```

where `A_t` is the instantaneous absorbed fraction.  There is no finite
time-independent absorbed *fraction* for an unbounded ramp: the visible bias
plateaus at `s/(1-lambda)`, while the injected ramp grows indefinitely, so
`A_t -> 1`.  Measurement noise leaves the mean law unchanged but governs the
minimum detectable slope through the noise covariance.

Run the controlled validation and three-view detectability calculation with:

```bash
python validate_theory.py --sigma 0.1 --horizon 1000 --trials 2000 --out results/theory
```

For each view it reports the one-sided Gaussian matched-filter threshold
`s_min = [Phi^-1(1-alpha)+Phi^-1(power)] / sqrt(g' Sigma^-1 g)` at
`alpha=0.05`, power `0.80`: `output` is the compensated stream, `state` is
the RLS-intercept increment, and `peer` is the raw stream minus an independent
reference.  In the ideal scalar Gaussian model state and output contain the
same information (one is a nonsingular scaling of the other); state becomes
useful in the full model when it is monitored jointly with model structure or
protected from the same data path.  The peer/reference has a distinct,
unabsorbed ramp shape.

## Interpreting the summary

The script prints a text summary at the end identifying:
1. Which (slope, λ) cells have >50% absorption
2. Where detectors fail (detection rate < 50%) on the compensated output

Cross-referencing these two lists reveals the blind-spot region — if it exists.
A strong effect would show: slow slopes + aggressive forgetting (λ ≈ 0.95–0.99)
absorbing most of the bias while detectors on the compensated stream fail to fire,
even though detectors on the static baseline or raw residual do fire.

If the effect is weak or absent, that is a valid and informative null result.

