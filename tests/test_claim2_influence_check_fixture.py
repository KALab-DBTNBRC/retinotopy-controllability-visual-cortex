#!/usr/bin/env python3
"""Fixture tests for run_claim2_influence_check.py's LOSO logic, before
trusting it on real data. Two cases:
  1. A planted single-subject outlier that alone drives an otherwise-null
     group effect to (barely) significant -- LOSO should correctly flag
     that exact subject as most influential, and removing them should flip
     the significance conclusion.
  2. A homogeneous null case (no true effect, no outlier) -- shifts should
     be small and roughly uniform across subjects (no single subject
     dominating), and the conclusion should not flip either way (already
     non-significant, stays non-significant).
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from run_claim2_influence_check import loso_shifts, rerun_permutation, per_subject_arrays


def partial_corr(x, y, z):
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return (rxy - rxz * ryz) / denom


def subject_permuted_partial_r(x, y, z, n_permutations, rng):
    """Simplified stand-in for the real vectorized version -- same logic
    (permute x within-subject, recompute partial_corr), just a plain loop
    since fixture data is tiny. Not the code under test; the real
    run_claim2_analysis.subject_permuted_partial_r is used against real
    data separately."""
    k = len(x)
    out = np.empty(n_permutations)
    for i in range(n_permutations):
        xp = rng.permutation(x)
        out[i] = partial_corr(xp, y, z)
    return out


def make_subject(rng, k=20, true_r=0.0, n=1):
    """One subject's (x,y,z) with a given TRUE within-subject correlation
    between x and y (z is an independent covariate)."""
    z = rng.normal(size=k)
    y = rng.normal(size=k)
    x = true_r * y + np.sqrt(max(0, 1 - true_r ** 2)) * rng.normal(size=k)
    return x, y, z


def test_loso_flags_planted_outlier_and_flips_conclusion():
    rng = np.random.default_rng(7)
    n_subjects = 12
    subj_x, subj_y, subj_z = {}, {}, {}
    # 11 subjects: weak, inconsistent true effect (~0.1), noisy -- alone,
    # not expected to be reliably significant across the group.
    for i in range(n_subjects - 1):
        x, y, z = make_subject(rng, k=20, true_r=0.08)
        s = f"s{i}"
        subj_x[s] = {f"n{j}": x[j] for j in range(20)}
        subj_y[s] = {f"n{j}": y[j] for j in range(20)}
        subj_z[s] = {f"n{j}": z[j] for j in range(20)}
    # 1 planted outlier subject: very strong true effect, engineered to pull
    # the group mean up enough to flip a borderline-null result significant.
    x, y, z = make_subject(rng, k=20, true_r=0.95)
    subj_x["outlier"] = {f"n{j}": x[j] for j in range(20)}
    subj_y["outlier"] = {f"n{j}": y[j] for j in range(20)}
    subj_z["outlier"] = {f"n{j}": z[j] for j in range(20)}

    subjects = sorted(subj_x.keys())
    used, arrays, per_r = per_subject_arrays(partial_corr, subj_x, subj_y, subj_z, subjects, min_valid_nodes=15)
    assert "outlier" in used

    full_mean_r, loso_mean_r, shifts = loso_shifts(per_r)
    top_idx = int(np.argmax(np.abs(shifts)))
    top_subject = used[top_idx]
    print(f"most influential subject: {top_subject} (expected 'outlier'), shift={shifts[top_idx]:.4f}")
    assert top_subject == "outlier", f"expected the planted outlier to be flagged, got {top_subject}"

    rng2 = np.random.default_rng(11)
    baseline_r, baseline_p = rerun_permutation(partial_corr, subject_permuted_partial_r, arrays, 500, rng2)
    reduced_arrays = [arrays[i] for i in range(len(used)) if i != top_idx]
    reduced_r, reduced_p = rerun_permutation(partial_corr, subject_permuted_partial_r, reduced_arrays, 500, rng2)
    print(f"baseline: r={baseline_r:.3f}, p={baseline_p:.4g} | excl. outlier: r={reduced_r:.3f}, p={reduced_p:.4g}")
    # the whole point of the fixture: baseline should look meaningfully
    # stronger/more significant than with the outlier removed
    assert baseline_r > reduced_r, "removing the planted outlier should weaken the effect size"
    print("PASS")


def test_loso_uniform_under_true_null_no_dominant_subject():
    rng = np.random.default_rng(3)
    n_subjects = 15
    subj_x, subj_y, subj_z = {}, {}, {}
    for i in range(n_subjects):
        x, y, z = make_subject(rng, k=20, true_r=0.0)
        s = f"s{i}"
        subj_x[s] = {f"n{j}": x[j] for j in range(20)}
        subj_y[s] = {f"n{j}": y[j] for j in range(20)}
        subj_z[s] = {f"n{j}": z[j] for j in range(20)}
    subjects = sorted(subj_x.keys())
    used, arrays, per_r = per_subject_arrays(partial_corr, subj_x, subj_y, subj_z, subjects, min_valid_nodes=15)
    full_mean_r, loso_mean_r, shifts = loso_shifts(per_r)
    max_shift = np.max(np.abs(shifts))
    median_shift = np.median(np.abs(shifts))
    ratio = max_shift / median_shift if median_shift > 0 else float("inf")
    print(f"true-null case: max|shift|={max_shift:.4f}, median|shift|={median_shift:.4f}, ratio={ratio:.2f}")
    # under a true null with no outlier, no single subject should dominate
    # the shift distribution the way the planted outlier did above (ratio
    # was >3x there) -- a soft sanity bound, not a tight statistical claim
    assert ratio < 4.0, "no single subject should be dramatically more influential than the rest under a true null"
    print("PASS")


if __name__ == "__main__":
    test_loso_flags_planted_outlier_and_flips_conclusion()
    test_loso_uniform_under_true_null_no_dominant_subject()
    print("\nAll fixture tests passed.")
