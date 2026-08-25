#!/usr/bin/env python3
"""
Claim 1 -- Receptor-informed controllability in a sensory system.

Tests whether node-level receptor density (Julich-Brain, hOc1/hOc2/hOc3v/hOc3d
only -- the 4 of 7 target regions with real receptor coverage, see todo.md
decisions log) predicts average/modal controllability of those same regions,
over and above region size.

Statistical design (per-subject spatial correlation, matching the actual
methodology of the two papers Claim 1 is positioned against -- Singleton et
al. 2022 and Braun et al. 2021, both receptor-informed NCT studies, neither
of which pools subjects into one mixed model for this kind of test):
  1. Receptor density is GROUP-level (one fixed value per region/hemisphere,
     from the Julich-Brain post-mortem histological atlas -- NOT per-subject,
     there is no "this living subject's own receptor density").
  2. Controllability is PER-SUBJECT (from Phase 5/6, weighted onto each hOc
     region via build_hoc_region_controllability.py).
  3. For each subject, correlate their own 8-length controllability vector
     (hOc1/hOc2/hOc3v/hOc3d x left/right, kept separate rather than
     hemisphere-averaged so real bilateral asymmetry isn't averaged away)
     against the fixed 8-length receptor-density vector, ACROSS REGIONS.
     Primary statistic is the PARTIAL correlation controlling for each
     subject's own region vertex count (n_hoc_vertices) -- per this
     project's own methodological note (lit-review.docx Section 5):
     "because average controllability is known to scale with region
     size/streamline count, any comparison across Julich parcels of unequal
     size should include size as a covariate." Raw (non-partial) Pearson r
     is also reported for transparency/comparison, not as the primary claim.
  4. Each subject's partial-r is Fisher z-transformed; a one-sample t-test
     (primary) and Wilcoxon signed-rank test (nonparametric cross-check)
     ask whether the mean z is reliably different from 0 across the 46
     subjects. This tests "is this receptor's spatial relationship with
     controllability consistent across the cohort," not "is any single
     subject's correlation significant" (N=46 subjects is the real sample
     size here, not N=8 regions).
  5. Benjamini-Hochberg FDR applied across the full family of tests run
     (n_receptors x 2 controllability metrics) -- user decision, 2026-08-11.

average_controllability is analyzed in LOG-SPACE (2026-08-12 fix): a real,
systematic right>left asymmetry (up to ~9x whole-brain) was traced via
diagnose_connectome_hemisphere_asymmetry.py to be visual-cortex-specific and
NOT present in the raw pre-controllability connectome at any level checked
(whole-brain or Vis-only) -- concluded to be a genuine, known property of
average controllability's heavy-tailed, eigenstructure-sensitive
distribution (field-standard practice is to log-transform it before linear
statistics), not a pipeline bug. This script now reads
`weighted_avg_controllability_log` (build_hoc_region_controllability.py's
weighted mean of log(avg_controllability) across overlapping backbone nodes,
2026-08-12) for the "average_controllability" arm instead of the raw column,
so the partial correlation isn't dominated by a handful of extreme-valued
regions/subjects. modal_controllability is unaffected (no asymmetry found at
any point in that investigation) and is still analyzed on its raw scale.

IMPORTANT -- with only 8 spatial units (4 regions x 2 hemispheres), the
parametric one-sample t-test above is NOT trustworthy on its own, confirmed
via a synthetic fixture test before this script was trusted on real data
(2026-08-11): two receptors with densities generated as pure noise, entirely
independent of the true controllability-driving signal, still came back
"significant" (p<1e-16) under the parametric test. Root cause is real, not a
fixture quirk: receptor density is a FIXED 8-length vector (group-level,
identical for every subject, exactly like real Julich-Brain data), so if it
happens to align even moderately with controllability's true spatial pattern
purely by chance at n=8, that SAME chance alignment appears in every
subject's data (since controllability's regional pattern is itself fairly
stable across a healthy cohort) -- inflating the cross-subject t-test into
apparent high significance for what is really just one coincidental n=8
correlation, not 46 independent confirmations of it. This is a real risk on
real data too, not just this fixture.

Fix: a spatial permutation test is the PRIMARY significance test. The 8
receptor-density values are randomly reassigned across the 8 region labels
(breaking the true region correspondence while keeping every subject's real
controllability data untouched), the same group-level statistic (mean Fisher
z across subjects) is recomputed, and this repeats --n-permutations times
(default 10000) to build an empirical null distribution specific to this
exact 8-unit design. The permutation p-value (fraction of |null stat| >=
|observed stat|) is what FDR-correction and the "significant results" summary
use -- NOT the parametric p-value, which is reported alongside for standard-
practice comparison only. Re-verified against the same synthetic fixture
after adding this: REAL_SIGNAL (the true effect) remained significant,
NOISE1/NOISE2 (the two false positives above) both came back non-significant
under permutation, confirming this fixes the exact failure mode found.

Usage:
    python3 run_claim1_analysis.py \
        --region-controllability-csv /path/to/hoc_region_controllability/schaefer200.csv \
        --receptor-csv /path/to/receptor-densities-deduped.csv \
        --out-csv /path/to/claim1_results_schaefer200.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REGION_ORDER = ["hOc1", "hOc2", "hOc3v", "hOc3d"]
HEMI_ORDER = ["left", "right"]
UNIT_ORDER = [f"{r}_{h}" for r in REGION_ORDER for h in HEMI_ORDER]  # 8 spatial units, fixed order


def load_region_controllability(path):
    """Returns:
      subj_avg[subject][hoc_name] = weighted_avg_controllability_log (LOG-SPACE,
        see module docstring re: the 2026-08-12 asymmetry investigation --
        NOT the raw weighted_avg_controllability column)
      subj_modal[subject][hoc_name] = weighted_modal_controllability (raw, unaffected)
      subj_size[subject][hoc_name] = n_hoc_vertices
    """
    subj_avg = defaultdict(dict)
    subj_modal = defaultdict(dict)
    subj_size = defaultdict(dict)
    n_missing_log = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "weighted_avg_controllability_log" not in reader.fieldnames:
            raise ValueError(
                f"{path}: no 'weighted_avg_controllability_log' column found -- this file was "
                f"built with an older version of build_hoc_region_controllability.py (pre-2026-08-12 "
                f"log-space fix). Rebuild it before running this script, otherwise the "
                f"average_controllability arm would silently fall back to the unreliable raw scale."
            )
        for row in reader:
            s, h = row["subject"], row["hoc_name"]
            # modal_controllability and size are independent of the log-space fix -- always
            # populate them regardless of whether this row's log value is usable.
            subj_modal[s][h] = float(row["weighted_modal_controllability"])
            subj_size[s][h] = float(row["n_hoc_vertices"])

            log_val = row.get("weighted_avg_controllability_log", "")
            if log_val == "" or log_val is None:
                n_missing_log += 1
                continue  # e.g. a nonpositive raw value excluded it -- don't guess, drop this unit
            subj_avg[s][h] = float(log_val)
    if n_missing_log:
        print(f"WARNING: {n_missing_log} rows had no weighted_avg_controllability_log value "
              f"(see build_hoc_region_controllability.py output) and were dropped from the "
              f"average_controllability arm only -- modal_controllability for those subjects/regions "
              f"is unaffected (populated independently above).")
    return subj_avg, subj_modal, subj_size


def load_receptor_vectors(path):
    """Returns {receptor: {hoc_name: mean_density}}."""
    out = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hoc_name = f"{row['region']}_{row['hemisphere']}"
            out[row["receptor"]][hoc_name] = float(row["mean_density"])
    return out


def partial_corr(x, y, z):
    """First-order partial correlation of x,y controlling for z. Returns NaN
    if any pairwise correlation is undefined (e.g. zero variance)."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return (rxy - rxz * ryz) / denom


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns q-values in the same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (standard BH step-up correction)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def pearson_rowwise(A, B):
    """Row-wise Pearson correlation between two (n, k) matrices (or one
    broadcastable to the other) -- vectorized equivalent of calling
    scipy.stats.pearsonr(A[i], B[i]) for every row i, used so the spatial
    permutation test (thousands of reshuffles) doesn't pay per-call scipy
    overhead. Verified to match stats.pearsonr row-for-row against a
    synthetic fixture (2026-08-11)."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def vectorized_mean_fisher_z(x_vec, Y_mat, Z_mat):
    """For a single fixed 8-length receptor vector x_vec and per-subject
    controllability/size matrices Y_mat, Z_mat (n_subjects x 8), computes
    each subject's partial correlation (x vs y controlling for z) and
    returns the mean Fisher z across subjects (NaN-safe). This is the group-
    level statistic the permutation null distribution is built from."""
    n = Y_mat.shape[0]
    X_mat = np.tile(x_vec, (n, 1))
    rxy = pearson_rowwise(X_mat, Y_mat)
    rxz = pearson_rowwise(X_mat, Z_mat)
    ryz = pearson_rowwise(Y_mat, Z_mat)
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = (rxy - rxz * ryz) / denom
    valid = np.isfinite(rp)
    if not np.any(valid):
        return np.nan
    z = np.arctanh(np.clip(rp[valid], -0.999999, 0.999999))
    return float(np.mean(z))


def spatial_permutation_test(receptor_x, Y_mat, Z_mat, observed_stat, n_permutations, rng):
    """PRIMARY significance test (see module docstring). Randomly reassigns
    the 8 receptor-density values across the 8 region labels (breaking the
    true region correspondence, leaving every subject's real controllability
    data untouched), recomputes the same group-level statistic
    (mean_fisher_z) for each reshuffle, and returns the resulting null
    distribution plus a two-sided empirical p-value with add-one smoothing
    (standard practice, avoids a p=0 from finite permutation counts)."""
    n_units = len(receptor_x)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_units)
        null_stats[i] = vectorized_mean_fisher_z(receptor_x[perm], Y_mat, Z_mat)
    null_stats = null_stats[np.isfinite(null_stats)]
    if len(null_stats) == 0 or not np.isfinite(observed_stat):
        return np.nan, null_stats
    perm_p = (np.sum(np.abs(null_stats) >= abs(observed_stat)) + 1) / (len(null_stats) + 1)
    return float(perm_p), null_stats


def run_one_receptor_metric(subj_controllability, subj_size, receptor_vec, subjects,
                             n_permutations=10000, rng=None):
    """Returns dict of summary stats, or a note dict if too few valid subjects."""
    receptor_x = np.array([receptor_vec[u] for u in UNIT_ORDER])

    per_subject_r = []
    per_subject_partial_r = []
    used_subjects = []
    y_rows, z_rows = [], []

    for s in subjects:
        c_vec = subj_controllability.get(s, {})
        size_vec = subj_size.get(s, {})
        if not all(u in c_vec for u in UNIT_ORDER) or not all(u in size_vec for u in UNIT_ORDER):
            continue  # this subject is missing at least one of the 8 spatial units -- drop, don't guess
        y = np.array([c_vec[u] for u in UNIT_ORDER])
        z = np.array([size_vec[u] for u in UNIT_ORDER])

        if np.std(y) == 0:
            continue
        r_raw = stats.pearsonr(receptor_x, y)[0]
        r_partial = partial_corr(receptor_x, y, z)
        if not np.isfinite(r_partial):
            continue

        per_subject_r.append(r_raw)
        per_subject_partial_r.append(r_partial)
        used_subjects.append(s)
        y_rows.append(y)
        z_rows.append(z)

    n_valid = len(used_subjects)
    if n_valid < 5:
        return {"n_subjects_valid": n_valid, "note": "too few valid subjects, skipped"}

    z_vals = np.arctanh(np.clip(per_subject_partial_r, -0.999999, 0.999999))
    t_stat, p_t = stats.ttest_1samp(z_vals, 0.0)
    try:
        w_stat, p_w = stats.wilcoxon(per_subject_partial_r)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    observed_mean_fisher_z = float(np.mean(z_vals))
    Y_mat = np.array(y_rows)
    Z_mat = np.array(z_rows)
    if rng is None:
        rng = np.random.default_rng()
    perm_p, null_stats = spatial_permutation_test(
        receptor_x, Y_mat, Z_mat, observed_mean_fisher_z, n_permutations, rng
    )

    return {
        "n_subjects_valid": n_valid,
        "mean_raw_r": float(np.mean(per_subject_r)),
        "mean_partial_r": float(np.mean(per_subject_partial_r)),
        "mean_fisher_z": observed_mean_fisher_z,
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else None,
        "p_wilcoxon": float(p_w) if np.isfinite(p_w) else None,
        "perm_p_value": perm_p,
        "n_permutations_used": int(len(null_stats)),
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-controllability-csv", required=True)
    ap.add_argument("--receptor-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000,
                     help="spatial permutation count per receptor x metric test -- this is the PRIMARY "
                          "significance test given only 8 spatial units, see module docstring")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility across runs")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    subj_avg, subj_modal, subj_size = load_region_controllability(args.region_controllability_csv)
    receptor_vectors = load_receptor_vectors(args.receptor_csv)
    subjects = sorted(subj_avg.keys())

    print(f"{len(subjects)} subjects with region-controllability data")
    print(f"{len(receptor_vectors)} receptors found: {sorted(receptor_vectors.keys())}")
    print(f"Spatial permutation test: {args.n_permutations} permutations per receptor x metric, seed={args.seed}")

    missing_units = {u for u in UNIT_ORDER if not all(u in receptor_vectors[r] for r in receptor_vectors)}
    if missing_units:
        print(f"WARNING: some spatial units missing from the receptor CSV entirely: {missing_units} "
              f"-- results for affected receptors will be dropped, not silently zero-filled")

    results = []
    for receptor, vec in sorted(receptor_vectors.items()):
        if not all(u in vec for u in UNIT_ORDER):
            print(f"receptor={receptor}: SKIPPED -- missing one of {UNIT_ORDER} in receptor data")
            continue
        for metric_name, subj_c in [("average_controllability_log", subj_avg), ("modal_controllability", subj_modal)]:
            stat = run_one_receptor_metric(subj_c, subj_size, vec, subjects,
                                            n_permutations=args.n_permutations, rng=rng)
            row = {"receptor": receptor, "metric": metric_name}
            row.update(stat)
            results.append(row)

    # PRIMARY: FDR across the permutation p-values (see module docstring -- the
    # parametric p_ttest is reported for standard-practice comparison only and
    # is NOT what significance is judged on).
    perm_p_values = [r["perm_p_value"] for r in results if "perm_p_value" in r and np.isfinite(r["perm_p_value"])]
    q_values = bh_fdr(perm_p_values) if perm_p_values else []
    qi = 0
    for r in results:
        if "perm_p_value" in r and np.isfinite(r["perm_p_value"]):
            r["q_value_fdr_perm"] = float(q_values[qi])
            qi += 1
        else:
            r["q_value_fdr_perm"] = None

    fieldnames = ["receptor", "metric", "n_subjects_valid", "mean_raw_r", "mean_partial_r",
                  "mean_fisher_z", "perm_p_value", "n_permutations_used", "q_value_fdr_perm",
                  "t_stat", "p_ttest", "wilcoxon_stat", "p_wilcoxon", "note"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n=== Results (permutation FDR q < 0.05 -- PRIMARY test) ===")
    sig = [r for r in results if r.get("q_value_fdr_perm") is not None and r["q_value_fdr_perm"] < 0.05]
    if sig:
        for r in sorted(sig, key=lambda x: x["q_value_fdr_perm"]):
            print(f"  {r['receptor']} / {r['metric']}: partial_r={r['mean_partial_r']:.3f}, "
                  f"perm_p={r['perm_p_value']:.4g}, q={r['q_value_fdr_perm']:.4g}, "
                  f"(parametric p_ttest={r['p_ttest']:.4g} for comparison), n={r['n_subjects_valid']}")
    else:
        print("  (none)")

    print(f"\nFull results ({len(results)} rows) written to {out_csv}")


if __name__ == "__main__":
    main()
