#!/usr/bin/env python3
"""Fixture tests for receptor_behavior_direct.py's new statistical piece
(the raw, no-covariate spatial permutation test) before trusting it on real
data. The rest of the script reuses already-verified functions (Claim 1's
UNIT_ORDER/load_region_controllability/pearson_rowwise/bh_fdr, Claim 3's
load_behavioral/run_one_pair), so only the new function needs its own
fixture here."""
import numpy as np
from scipy import stats
import run_claim1_analysis as c1
from receptor_behavior_direct import test_receptor_vs_size

rng = np.random.default_rng(7)


def test_detects_real_signal():
    n_subjects = 46
    true_pattern = rng.normal(size=8)
    receptor_x = true_pattern + rng.normal(scale=0.1, size=8)  # receptor tracks the true pattern closely
    subj_size = {}
    for i in range(n_subjects):
        y = true_pattern + rng.normal(scale=0.5, size=8)  # each subject's own size noisily tracks the same true pattern
        subj_size[f"s{i}"] = {u: float(v) for u, v in zip(c1.UNIT_ORDER, y)}
    subjects = list(subj_size.keys())
    stat, per_subj = test_receptor_vs_size(receptor_x, subj_size, subjects, 2000, rng)
    print(f"test_detects_real_signal: mean_r={stat['mean_raw_r']:.3f}, perm_p={stat['perm_p_value']:.4g} -- "
          f"{'PASS' if stat['perm_p_value'] < 0.01 and stat['mean_raw_r'] > 0.3 else 'FAIL'}")
    assert stat["perm_p_value"] < 0.01
    assert stat["mean_raw_r"] > 0.3


def test_null_calibration():
    n_reps = 300
    sig_count = 0
    local_rng = np.random.default_rng(11)
    for _ in range(n_reps):
        receptor_x = local_rng.normal(size=8)
        subj_size = {}
        for i in range(20):
            y = local_rng.normal(size=8)  # fully independent of receptor_x
            subj_size[f"s{i}"] = {u: float(v) for u, v in zip(c1.UNIT_ORDER, y)}
        subjects = list(subj_size.keys())
        stat, _ = test_receptor_vs_size(receptor_x, subj_size, subjects, 500, local_rng)
        if stat.get("perm_p_value") is not None and stat["perm_p_value"] < 0.05:
            sig_count += 1
    rate = sig_count / n_reps
    print(f"test_null_calibration: false-positive rate = {rate:.3f} (expected ~0.05) -- "
          f"{'PASS' if 0.01 < rate < 0.12 else 'FAIL'}")
    assert 0.01 < rate < 0.12


def test_missing_subject_dropped_not_guessed():
    receptor_x = rng.normal(size=8)
    subj_size = {
        "complete": {u: float(v) for u, v in zip(c1.UNIT_ORDER, rng.normal(size=8))},
        "incomplete": {u: float(v) for u, v in zip(c1.UNIT_ORDER[:6], rng.normal(size=6))},  # missing 2 units
    }
    stat, per_subj = test_receptor_vs_size(receptor_x, subj_size, ["complete", "incomplete"], 200, rng)
    print(f"test_missing_subject_dropped_not_guessed: n_valid={stat['n_subjects_valid']} -- "
          f"{'PASS' if stat['n_subjects_valid'] == 1 and 'incomplete' not in per_subj else 'FAIL'}")
    assert stat["n_subjects_valid"] == 1
    assert "incomplete" not in per_subj


if __name__ == "__main__":
    test_detects_real_signal()
    test_null_calibration()
    test_missing_subject_dropped_not_guessed()
    print("\nAll fixture tests passed.")
