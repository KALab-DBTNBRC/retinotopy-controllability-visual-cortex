#!/usr/bin/env python3
"""
Fixture test for aggregate_prf_to_backbone.py's circular_mean_deg() (2026-08-13
extension, Claim 2 Phase B3/C2). Confirms the circular-mean logic handles the
0/360 degree wraparound correctly and that its concentration statistic (r)
correctly discriminates tightly-clustered vs. evenly-scattered angle sets.
Run before trusting mean_polar_angle_deg on real data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/path/to/project")
from aggregate_prf_to_backbone import circular_mean_deg


def test_wraparound():
    # [350, 5, 10] deg == [-10, 5, 10] deg -> true circular mean = 5/3 = 1.667 deg.
    # A naive arithmetic mean gives 121.67 deg -- wildly wrong, the exact
    # failure mode this function exists to avoid.
    m, r = circular_mean_deg([350, 5, 10])
    naive = np.mean([350, 5, 10])
    assert abs(m - 1.667) < 0.1, f"expected ~1.667, got {m}"
    assert r > 0.9, "tightly clustered angles should give high concentration"
    assert abs(naive - 121.67) < 0.1


def test_scattered_low_concentration():
    # angles spread evenly around the full circle -> the "mean" is not a
    # meaningful summary, and concentration r should correctly signal that.
    m, r = circular_mean_deg([0, 90, 180, 270])
    assert r < 0.01, "evenly-scattered angles should give near-zero concentration"


def test_simple_no_wraparound():
    m, r = circular_mean_deg([40, 50, 60])
    assert abs(m - 50.0) < 1.0
    assert r > 0.9


if __name__ == "__main__":
    test_wraparound()
    test_scattered_low_concentration()
    test_simple_no_wraparound()
    print("All aggregate_prf_to_backbone circular-mean fixture tests passed.")
