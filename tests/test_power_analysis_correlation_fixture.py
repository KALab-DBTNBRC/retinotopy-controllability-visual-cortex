#!/usr/bin/env python3
"""Fixture tests for power_analysis_correlation.py -- checked against
Cohen's (1988) own published power tables before trusting this on real
project data, not just internal consistency."""
import sys
sys.path.insert(0, ".")
from power_analysis_correlation import power_for_r, min_detectable_r, required_n


def test_matches_cohen_benchmark_r30_n84():
    # Cohen 1988, Table 3.4.1: r=.30, two-tailed alpha=.05 -> n=84 gives
    # power ~=.80 (the commonly-cited "medium effect" sample-size rule).
    power = power_for_r(0.30, 84, alpha=0.05)
    print(f"power_for_r(0.30, n=84): {power:.4f} (Cohen's benchmark: ~0.80)")
    assert 0.78 <= power <= 0.82, f"expected ~0.80, got {power:.4f}"
    print("PASS")


def test_matches_cohen_benchmark_r50_n28():
    # Cohen 1988: r=.50, two-tailed alpha=.05 -> n~=28 gives power ~=.80
    power = power_for_r(0.50, 28, alpha=0.05)
    print(f"power_for_r(0.50, n=28): {power:.4f} (Cohen's benchmark: ~0.80)")
    assert 0.75 <= power <= 0.85, f"expected ~0.80, got {power:.4f}"
    print("PASS")


def test_required_n_matches_min_detectable_r_inverse():
    # required_n and min_detectable_r should be inverse functions of each
    # other at a fixed power/alpha -- internal consistency check.
    n = required_n(0.30, power=0.80, alpha=0.05)
    r_back = min_detectable_r(n, power=0.80, alpha=0.05)
    print(f"required_n(r=0.30, power=0.80) = {n}; min_detectable_r(n={n}) = {r_back:.4f}")
    assert abs(r_back - 0.30) < 0.01
    print("PASS")


def test_power_increases_with_n_and_r():
    assert power_for_r(0.30, 100) > power_for_r(0.30, 30)
    assert power_for_r(0.50, 46) > power_for_r(0.10, 46)
    print("power monotonicity checks PASS")


def test_low_power_at_this_projects_real_n_for_small_effects():
    # sanity check specific to this project: with n=46 (Claim 3/personality's
    # real sample size), power to detect a SMALL effect should be low --
    # this is the whole point of running the analysis, confirm the numbers
    # are in a sane, expected range before reporting them.
    p_small = power_for_r(0.10, 46)
    p_medium = power_for_r(0.30, 46)
    print(f"n=46: power(small r=.10)={p_small:.3f}, power(medium r=.30)={p_medium:.3f}")
    assert p_small < 0.15, "power to detect a small effect at n=46 should be low (near alpha)"
    assert 0.4 < p_medium < 0.6, "power to detect a medium effect at n=46 should be moderate, ~50%"
    print("PASS")


if __name__ == "__main__":
    test_matches_cohen_benchmark_r30_n84()
    test_matches_cohen_benchmark_r50_n28()
    test_required_n_matches_min_detectable_r_inverse()
    test_power_increases_with_n_and_r()
    test_low_power_at_this_projects_real_n_for_small_effects()
    print("\nAll fixture tests passed.")
