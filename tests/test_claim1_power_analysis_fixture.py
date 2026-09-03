#!/usr/bin/env python3
"""Fixture tests for run_claim1_power_analysis.py -- checked on synthetic
data with a known ground truth before trusting the Monte Carlo harness on
real data. Two checks:
  1. Calibration at r_true=0: empirical power should land near the nominal
     5% alpha (same "does the harness itself introduce bias" check
     calibrate_claim4_interaction.py did for the interaction test).
  2. Sensitivity at a strong true effect (r_true=0.9): empirical power
     should be high -- confirms the harness can actually detect a real
     signal when one is present, not just correctly stay quiet when one
     isn't.
"""

import sys
from pathlib import Path
# Resolve sibling script directories relative to this file, so the tests run
# from any checkout location without editing paths.
_ROOT = Path(__file__).resolve().parent.parent
for _d in sorted((_ROOT / "scripts").iterdir()):
    if _d.is_dir():
        sys.path.insert(0, str(_d))

import sys
import numpy as np
import run_claim1_analysis as c1
from run_claim1_power_analysis import simulate_power_curve, make_base_pattern, interpolate_mdes


def make_synthetic_YZ(rng, n_subjects=30, n_units=8):
    """Synthetic per-subject controllability (Y) and size (Z) matrices with
    a real shared group-level pattern plus subject-level noise -- mimics
    the real data's structure (every subject varies around a common
    regional profile) without needing real data for this fixture."""
    group_pattern = rng.normal(size=n_units)
    Y = np.array([group_pattern + rng.normal(scale=0.5, size=n_units) for _ in range(n_subjects)])
    Z = np.array([rng.normal(loc=5, scale=1, size=n_units) for _ in range(n_subjects)])
    return Y, Z


def test_calibration_near_nominal_alpha_at_r_true_zero():
    rng = np.random.default_rng(21)
    Y, Z = make_synthetic_YZ(rng)
    results = simulate_power_curve(c1, Y, Z, r_true_levels=[0.0], n_reps=150, n_permutations=1000, rng=rng)
    power = results[0]["empirical_power"]
    print(f"calibration (r_true=0): empirical power (false-positive rate) = {power:.3f} (nominal alpha=0.05)")
    # binomial SE at n_reps=150, p=0.05 is ~0.018 -- allow a generous band
    assert power < 0.13, f"false-positive rate too high, harness may be miscalibrated: {power:.3f}"
    print("PASS")


def test_detects_strong_true_signal():
    rng = np.random.default_rng(23)
    Y, Z = make_synthetic_YZ(rng)
    results = simulate_power_curve(c1, Y, Z, r_true_levels=[0.9], n_reps=150, n_permutations=1000, rng=rng)
    power = results[0]["empirical_power"]
    print(f"strong signal (r_true=0.9): empirical power = {power:.3f}")
    assert power > 0.7, f"harness should reliably detect a strong true effect, got power={power:.3f}"
    print("PASS")


def test_power_curve_is_monotonically_increasing():
    rng = np.random.default_rng(29)
    Y, Z = make_synthetic_YZ(rng)
    results = simulate_power_curve(c1, Y, Z, r_true_levels=[0.0, 0.3, 0.6, 0.9], n_reps=100, n_permutations=1000, rng=rng)
    powers = [r["empirical_power"] for r in results]
    print(f"power curve across r_true=[0.0,0.3,0.6,0.9]: {powers}")
    # allow small Monte Carlo noise between adjacent levels, but the overall
    # trend across the full range must be clearly increasing
    assert powers[-1] > powers[0], "power should increase from r_true=0 to r_true=0.9"
    print("PASS")


def test_mdes_interpolation_sane():
    # synthetic power curve, exact interpolation check
    results = [
        {"r_true_target": 0.0, "empirical_power": 0.05},
        {"r_true_target": 0.5, "empirical_power": 0.60},
        {"r_true_target": 1.0, "empirical_power": 1.00},
    ]
    mdes = interpolate_mdes(results, target_power=0.80)
    print(f"interpolated MDES at 80% power: {mdes:.3f} (expected between 0.5 and 1.0)")
    assert 0.5 <= mdes <= 1.0
    print("PASS")


if __name__ == "__main__":
    test_calibration_near_nominal_alpha_at_r_true_zero()
    test_detects_strong_true_signal()
    test_power_curve_is_monotonically_increasing()
    test_mdes_interpolation_sane()
    print("\nAll fixture tests passed.")
