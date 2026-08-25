#!/usr/bin/env python3
"""
Claim 2, node-strength control arm
==================================

WHY THIS EXISTS
---------------
Tu, Rocha, Corbetta, Zampieri, Zorzi & Suweis (2018, NeuroImage 176:83-91)
showed that average and modal controllability are very strongly correlated
with weighted node degree, and that random null models with no biological
resemblance to connectome architecture reproduce the same degree-
controllability relationship reported by Gu et al. (2015). Pasqualetti,
Gu & Bassett (2019, NeuroImage) replied, but the empirical core of the
critique stands and is live in THIS project's own data. A quick check on
our real Schaefer-200 connectomes, restricted to the 29 Vis-network nodes
that Claim 2 actually analyses, gave (46 subjects, within-subject Spearman):

    weighted_degree vs modal_controllability    median rho = -0.918
    weighted_degree vs average_controllability  median rho = +0.801

At that level of collinearity a reviewer is entitled to ask whether
"eccentricity predicts controllability" is really just "eccentricity
predicts node degree." This script answers that question properly.

A preliminary, non-permutation check (rank-based partial correlation,
Wilcoxon across subjects) suggested the effect survives:

    average (size covariate only)      median r = +0.305, p = 5.1e-11
    average (size + degree covariate)  median r = +0.254, p = 3.1e-11
    modal   (size covariate only)      median r = -0.166, p = 7.3e-07
    modal   (size + degree covariate)  median r = -0.134, p = 4.2e-03

That check is NOT what this project reports. It used a rank transform and a
Wilcoxon, not the subject-level spatial permutation test that is the primary
significance measure for Claim 2 everywhere else. This script re-runs the
question through the real machinery so the answer is directly comparable to
the already-reported Claim 2 numbers. Expect the point estimates above to be
close but not identical (rank vs. raw), and expect the p-values to differ
(permutation vs. Wilcoxon).

WHAT IT DOES
------------
For each (atlas x metric) cell it computes, per subject, the partial
correlation between pRF eccentricity and controllability across that
subject's valid Vis-network nodes, under two covariate configurations:

    size-only        z = [n_vertices_total]              (reproduces the
                                                          already-reported
                                                          Claim 2 result --
                                                          this is the
                                                          regression check)
    size + strength  z = [n_vertices_total, weighted_degree]

Significance is the subject-level spatial permutation test used throughout
Claim 2: each subject's eccentricity vector is reshuffled across that
subject's OWN nodes (controllability and both covariates keep their real
node identity), the group mean Fisher-z is recomputed per permutation, and
the observed group mean Fisher-z is compared against that empirical null.
BH-FDR is applied across the 8-test family (2 atlases x 2 metrics x 2
covariate configs).

The size-only arm is included deliberately as a regression check: if it does
not reproduce the already-reported Claim 2 numbers, something is wrong with
this script and the size+strength arm should not be trusted either.

IMPORTANT: node strength is per-subject per-node (each subject has their own
connectome), unlike node size, which is fixed by the atlas. The covariate
matrix is therefore assembled per subject, not once globally.

VERIFY BEFORE YOU TRUST IT
--------------------------
    python run_claim2_nodestrength_control.py --self-test

runs five fixture checks with no real data:
    1. multi-covariate partial r reduces EXACTLY to run_claim2_analysis.py's
       own verified first-order partial_corr() when given a single covariate
    2. a planted true signal is recovered as significant
    3. a planted pure null is correctly non-significant
    4. a planted signal that is ENTIRELY mediated by the covariate is
       correctly killed once that covariate is included -- i.e. the script
       can actually detect the failure mode it was written to test for
    5. the vendored helpers reproduce the host run_claim2_analysis.py's
       equivalents exactly, when a compatible copy is importable

USAGE
-----
    # self-test first
    python run_claim2_nodestrength_control.py --self-test

    # Schaefer-200
    python run_claim2_nodestrength_control.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_schaefer200.csv \
        --node-strength-csv node_strength/all_subjects_node_strength_schaefer200.csv \
        --atlas schaefer200 \
        --out-csv claim2_nodestrength_control_schaefer200.csv

    # Glasser MMP1.0
    python run_claim2_nodestrength_control.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --node-strength-csv node_strength/all_subjects_node_strength_glasser.csv \
        --atlas glasser \
        --out-csv claim2_nodestrength_control_glasser.csv

COVARIATE CONFIGURATIONS
------------------------
--base-covariate size    (default) z = [n_vertices_total] then
                         [n_vertices_total, weighted_degree]. Regression
                         check: reproduces run_claim2_analysis.py's numbers.
--base-covariate rfsize  z = [mean_rfsize_deg] then
                         [mean_rfsize_deg, weighted_degree]. This is the
                         cortical-magnification proxy validated in Phase B3
                         and the covariate the manuscript's headline Claim 2
                         numbers use. Following run_claim2_rfsize_permutation.py,
                         rfsize REPLACES size rather than joining it.
                         Regression check: reproduces that script's
                         'rfsize_only' rows.
Run BOTH. The modal-controllability arm behaves differently under the two,
which is itself a reportable result.

Defaults (min-valid-nodes=15, min-vertices-per-node=5, n-permutations=10000,
seed=42) match run_claim2_analysis.py exactly. Runtime is dominated by the
permutation step: roughly comparable to a standard Claim 2 run.
"""

import argparse
import csv
import inspect
import json
import math
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# SELF-CONTAINED HELPERS
#
# An earlier version of this script imported these from run_claim2_analysis.py
# to avoid duplicating verified logic. That broke on the analysis server, which
# carries a PRE-2026-08-13 copy of run_claim2_analysis.py whose
# load_prf_backbone() takes (path, min_vertices_per_node) with no atlas
# argument -- it predates the --atlas support added for the Claim 5 Glasser
# rerun. The failure was loud (TypeError) rather than silent, but a stale copy
# that differed in BEHAVIOUR rather than signature would not have been, and
# this project has a documented history of stale-copy bugs (the 2026-08-08
# extract_prf_solutions.py incident).
#
# These functions are therefore vendored here verbatim from the canonical
# 2026-08-13 run_claim2_analysis.py, so this script's result does not depend
# on which vintage of that file happens to be sitting next to it. When a
# compatible copy IS importable, cross_check_against_host() runs both
# implementations on the real data and asserts they agree exactly.
# --------------------------------------------------------------------------

GLASSER_VIS_AREAS = {
    "V1",
    "V2", "V3", "V4",
    "V3A", "V3B", "V6", "V6A", "V7", "IPS1",
    "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",
    "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t",
}
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def is_glasser_vis_node(node_label):
    m = GLASSER_LABEL_RE.match(node_label)
    return bool(m) and m.group(1) in GLASSER_VIS_AREAS


def load_prf_backbone(path, min_vertices_per_node, atlas):
    """subj_ecc[subject][node] = mean_eccentricity_deg; node_size[node] = n_vertices_total."""
    subj_ecc = defaultdict(dict)
    node_size = {}
    n_dropped_thin = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["atlas"] != atlas:
                continue
            if atlas == "schaefer200":
                is_vis = row["is_vis_network"] == "True"
            else:
                is_vis = is_glasser_vis_node(row["backbone_label"])
            if not is_vis:
                continue
            node_size[row["backbone_label"]] = int(row["n_vertices_total"])
            if row["mean_eccentricity_deg"] == "":
                continue
            if int(row["n_vertices_valid_R2gt30"]) < min_vertices_per_node:
                n_dropped_thin += 1
                continue
            subj_ecc[row["subject"]][row["backbone_label"]] = float(row["mean_eccentricity_deg"])
    print(f"Dropped {n_dropped_thin} (subject, node) entries with < {min_vertices_per_node} "
          f"R2>30 vertices (too thin to trust the mean eccentricity estimate)")
    return subj_ecc, node_size


def load_prf_rfsize(path, min_vertices_per_node, atlas):
    """subj_rfsize[subject][node] = mean_rfsize_deg.

    pRF size is the cortical-magnification proxy validated in Phase B3 and is
    the covariate the manuscript's headline Claim 2 numbers actually use, in
    place of raw vertex count. Unlike n_vertices_total (fixed by the atlas),
    rfsize is per-subject AND per-node, so it is loaded separately here.
    Row-inclusion rules match load_prf_backbone() exactly, with the additional
    requirement -- taken from run_claim2_rfsize_permutation.py -- that
    mean_rfsize_deg is itself non-missing.
    """
    subj_rfsize = defaultdict(dict)
    n_missing_rfsize = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["atlas"] != atlas:
                continue
            if atlas == "schaefer200":
                is_vis = row["is_vis_network"] == "True"
            else:
                is_vis = is_glasser_vis_node(row["backbone_label"])
            if not is_vis:
                continue
            if row["mean_eccentricity_deg"] == "":
                continue
            if int(row["n_vertices_valid_R2gt30"]) < min_vertices_per_node:
                continue
            if row["mean_rfsize_deg"] == "":
                n_missing_rfsize += 1
                continue
            subj_rfsize[row["subject"]][row["backbone_label"]] = float(row["mean_rfsize_deg"])
    if n_missing_rfsize:
        print(f"Dropped {n_missing_rfsize} (subject, node) entries that had eccentricity "
              f"but no mean_rfsize_deg")
    return subj_rfsize


def load_controllability(path):
    """subj_avg_log[subject][node] = log(average_controllability); subj_modal likewise."""
    subj_avg_log = defaultdict(dict)
    subj_modal = defaultdict(dict)
    n_nonpositive = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s, n = row["subject"], row["node_label"]
            avg_c = float(row["average_controllability"])
            subj_modal[s][n] = float(row["modal_controllability"])
            if avg_c > 0:
                subj_avg_log[s][n] = math.log(avg_c)
            else:
                n_nonpositive += 1
    if n_nonpositive:
        print(f"WARNING: {n_nonpositive} rows had non-positive average_controllability -- "
              f"excluded from the log-space metric only, investigate if unexpected")
    return subj_avg_log, subj_modal


def partial_corr(x, y, z):
    """First-order partial correlation of x,y controlling for z."""
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
    """Benjamini-Hochberg FDR; q-values in input order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    q_sorted = pvals[order] * n / (np.arange(n) + 1)
    q_sorted = np.clip(np.minimum.accumulate(q_sorted[::-1])[::-1], 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def pearson_rowwise(A, B):
    """Row-wise Pearson correlation between two (n, k) matrices."""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def check_inputs_intersect(subj_ecc, node_size, subj_avg_log, subj_modal,
                           subj_strength, atlas):
    """Fail loudly, with specifics, when the three input files do not refer to
    the same atlas / node-label vocabulary.

    Motivation: a real 2026-08-22 server run returned 'too few valid subjects,
    skipped' (n=0) for all four Glasser cells. n=0 in the size_only arm cannot
    be a node-strength problem, because size_only never touches node strength
    -- so the collapse had to be upstream, in the eccentricity / controllability
    / node-size intersection. That is exactly the signature of passing a
    controllability or node-strength file from the OTHER atlas, whose node
    labels ('7Networks_LH_Vis_1') share nothing with Glasser's ('L_V1_ROI').
    Rather than report an empty result that looks like a finding, diagnose it.
    """
    ecc_nodes = {n for d in subj_ecc.values() for n in d}
    ctrl_nodes = {n for d in subj_avg_log.values() for n in d} | \
                 {n for d in subj_modal.values() for n in d}
    str_nodes = {n for d in subj_strength.values() for n in d}

    print(f"\nInput compatibility check (atlas = {atlas}):")
    print(f"  pRF backbone      : {len(subj_ecc):3d} subjects, {len(ecc_nodes):4d} vis nodes "
          f"with data, {len(node_size):4d} nodes in atlas")
    print(f"  controllability   : {len(subj_avg_log):3d} subjects, {len(ctrl_nodes):4d} nodes")
    print(f"  node strength     : {len(subj_strength):3d} subjects, {len(str_nodes):4d} nodes")
    print(f"  node-label overlap: ecc n ctrl = {len(ecc_nodes & ctrl_nodes):4d} | "
          f"ecc n strength = {len(ecc_nodes & str_nodes):4d}")

    def _example(s):
        return sorted(s)[:3] if s else []

    problems = []
    if not ecc_nodes:
        problems.append(
            f"The pRF backbone file yielded ZERO {atlas} Vis-network nodes with usable "
            f"eccentricity. Check that this file actually contains rows with "
            f"atlas == '{atlas}' (it must contain both atlases, or you must pass the "
            f"matching file).")
    if ecc_nodes and not (ecc_nodes & ctrl_nodes):
        problems.append(
            f"ZERO node-label overlap between the pRF backbone and the controllability "
            f"file. This almost always means the controllability CSV is from the OTHER "
            f"atlas.\n      pRF backbone labels look like : {_example(ecc_nodes)}\n"
            f"      controllability labels look like: {_example(ctrl_nodes)}")
    if ecc_nodes and not (ecc_nodes & str_nodes):
        problems.append(
            f"ZERO node-label overlap between the pRF backbone and the node-strength "
            f"file. This almost always means the node-strength CSV is from the OTHER "
            f"atlas.\n      pRF backbone labels look like : {_example(ecc_nodes)}\n"
            f"      node-strength labels look like : {_example(str_nodes)}")

    if problems:
        print("\n*** INPUT ERROR -- refusing to produce an empty result that could be "
              "mistaken for a finding ***")
        for p in problems:
            print(f"  - {p}")
        print(f"\n  For --atlas {atlas}, all three files must use the same node-label "
              f"vocabulary. Expected pattern: "
              f"{'7Networks_..._Vis_N' if atlas == 'schaefer200' else 'L_<AREA>_ROI / R_<AREA>_ROI'}")
        sys.exit(2)
    print("  OK -- all three inputs share a node-label vocabulary.\n")


def cross_check_against_host(prf_path, min_vertices_per_node, atlas):
    """If a COMPATIBLE run_claim2_analysis.py is importable next to this
    script, run its loader on the same real file and assert the vendored
    copy above reproduces it exactly. Skips (with a clear message) when the
    host copy is absent or is the stale pre-2026-08-13 vintage."""
    try:
        import run_claim2_analysis as host
    except ImportError:
        print("Cross-check: run_claim2_analysis.py not importable here -- skipped. "
              "Vendored loaders used (this is fine; results do not depend on it).")
        return
    try:
        n_params = len(inspect.signature(host.load_prf_backbone).parameters)
    except (TypeError, ValueError):
        n_params = -1
    if n_params != 3:
        print(f"Cross-check: SKIPPED -- the run_claim2_analysis.py next to this script is a "
              f"STALE pre-2026-08-13 copy (load_prf_backbone takes {n_params} args, expected 3; "
              f"it predates --atlas support). Vendored loaders used instead, so this run is "
              f"still correct. RECOMMEND updating that server copy before it is used for "
              f"anything else.")
        return
    h_ecc, h_size = host.load_prf_backbone(prf_path, min_vertices_per_node, atlas)
    v_ecc, v_size = load_prf_backbone(prf_path, min_vertices_per_node, atlas)
    same = (h_size == v_size
            and set(h_ecc) == set(v_ecc)
            and all(h_ecc[s] == v_ecc[s] for s in h_ecc))
    print("Cross-check against host run_claim2_analysis.py:",
          "PASSED (vendored loader reproduces it exactly)" if same
          else "*** FAILED -- vendored loader disagrees with host copy, STOP AND INVESTIGATE ***")
    if not same:
        sys.exit(1)


# --------------------------------------------------------------------------
# node strength loader (per-subject, per-node -- unlike node size)
# --------------------------------------------------------------------------

def load_node_strength(path):
    """Returns subj_strength[subject][node_label] = weighted_degree.

    Expects the real column names produced by compute_node_strength.py:
    subject, node_label, weighted_degree. Fails loudly on a schema mismatch
    rather than silently returning an empty dict -- this project has a
    history of clean-exit-but-empty-output bugs.
    """
    subj_strength = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"subject", "node_label", "weighted_degree"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )
        n_rows = 0
        for row in reader:
            subj_strength[row["subject"]][row["node_label"]] = float(row["weighted_degree"])
            n_rows += 1
    if n_rows == 0:
        raise ValueError(f"{path} contained no data rows.")
    print(f"Loaded node strength: {n_rows} rows, {len(subj_strength)} subjects")
    return subj_strength


# --------------------------------------------------------------------------
# multi-covariate partial correlation, by residualization
# --------------------------------------------------------------------------

def _residualize(v, Z):
    """Residuals of v after least-squares regression on Z (intercept added)."""
    X = np.column_stack([np.ones(len(v)), Z])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_corr_multi(x, y, Z):
    """Partial correlation of x and y controlling for every column of Z.

    With a single covariate column this is algebraically identical to
    run_claim2_analysis.py's first-order partial_corr(); self-test 1
    checks that numerically rather than asserting it.

    Returns NaN when the correlation is undefined (zero residual variance,
    or fewer observations than covariates + 2).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    if len(x) < Z.shape[1] + 3:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    rx = _residualize(x, Z)
    ry = _residualize(y, Z)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def subject_permuted_partial_r_multi(x, y, Z, n_permutations, rng):
    """Null distribution of one subject's partial r under within-subject
    reshuffles of x only (y and Z keep their real node identity).

    Mirrors run_claim2_analysis.py's subject_permuted_partial_r() but for an
    arbitrary number of covariates. Vectorized: y and Z are residualized once
    (they do not change across permutations), x is residualized per
    permutation, then correlations are computed row-wise in one pass.
    """
    x = np.asarray(x, dtype=float)
    k = len(x)
    ry = _residualize(np.asarray(y, dtype=float), Z)

    keys = np.random.default_rng(rng.integers(0, 2**63 - 1)).random((n_permutations, k))
    perm_idx = np.argsort(keys, axis=1)
    X_perm = x[perm_idx]                                  # (n_perm, k)

    # residualize every permuted x against the same fixed design matrix
    X_design = np.column_stack([np.ones(k), np.asarray(Z, dtype=float)])
    # hat matrix applied on the right: resid = X_perm - X_perm @ H.T
    H = X_design @ np.linalg.pinv(X_design)               # (k, k), symmetric
    RX = X_perm - X_perm @ H.T                            # (n_perm, k)

    RY = np.tile(ry, (n_permutations, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return pearson_rowwise(RX, RY)


# --------------------------------------------------------------------------
# one (metric x covariate-config) cell
# --------------------------------------------------------------------------

def run_one_cell(subj_ecc, subj_metric, subj_base, subj_strength, subjects,
                 use_strength, min_valid_nodes, n_permutations, rng):
    """subj_base is dict[subject][node] -> base covariate value. For the raw
    vertex-count covariate this is the same atlas-fixed value broadcast to
    every subject; for the rfsize covariate it genuinely varies by subject."""
    per_subject_partial_r = []
    subject_arrays = []
    used_subjects = []

    for s in subjects:
        ecc_vec = subj_ecc.get(s, {})
        c_vec = subj_metric.get(s, {})
        str_vec = subj_strength.get(s, {})
        base_vec = subj_base.get(s, {})

        common = set(ecc_vec) & set(c_vec) & set(base_vec)
        if use_strength:
            common &= set(str_vec)
        common = sorted(common)
        if len(common) < min_valid_nodes:
            continue

        x = np.array([ecc_vec[n] for n in common], dtype=float)
        y = np.array([c_vec[n] for n in common], dtype=float)
        cols = [np.array([base_vec[n] for n in common], dtype=float)]
        if use_strength:
            cols.append(np.array([str_vec[n] for n in common], dtype=float))
        Z = np.column_stack(cols)

        r_partial = partial_corr_multi(x, y, Z)
        if not np.isfinite(r_partial):
            continue

        per_subject_partial_r.append(r_partial)
        subject_arrays.append((x, y, Z))
        used_subjects.append(s)

    n_valid = len(used_subjects)
    if n_valid < 5:
        return {"n_subjects_valid": n_valid, "note": "too few valid subjects, skipped"}

    z_vals = np.arctanh(np.clip(per_subject_partial_r, -0.999999, 0.999999))
    observed_mean_fisher_z = float(np.mean(z_vals))
    t_stat, p_t = stats.ttest_1samp(z_vals, 0.0)
    try:
        w_stat, p_w = stats.wilcoxon(per_subject_partial_r)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    # PRIMARY: subject-level spatial permutation test
    null_rows = []
    for (x, y, Z) in subject_arrays:
        rp = subject_permuted_partial_r_multi(x, y, Z, n_permutations, rng)
        null_rows.append(np.arctanh(np.clip(rp, -0.999999, 0.999999)))
    null_mean_z = np.nanmean(np.vstack(null_rows), axis=0)
    null_mean_z = null_mean_z[np.isfinite(null_mean_z)]
    if len(null_mean_z) == 0:
        perm_p = np.nan
    else:
        perm_p = (np.sum(np.abs(null_mean_z) >= abs(observed_mean_fisher_z)) + 1) / (len(null_mean_z) + 1)

    return {
        "n_subjects_valid": n_valid,
        "median_n_nodes_used": float(np.median([len(x) for (x, _, _) in subject_arrays])),
        "mean_partial_r": float(np.mean(per_subject_partial_r)),
        "median_partial_r": float(np.median(per_subject_partial_r)),
        "mean_fisher_z": observed_mean_fisher_z,
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else None,
        "p_wilcoxon": float(p_w) if np.isfinite(p_w) else None,
        "perm_p_value": float(perm_p) if np.isfinite(perm_p) else None,
        "n_permutations_used": n_permutations,
        "note": "",
    }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def self_test():
    rng = np.random.default_rng(20260822)
    ok = True

    # 1. reduces to the project's own verified first-order partial_corr
    print("[1] multi-covariate partial r vs. run_claim2_analysis.partial_corr ...")
    worst = 0.0
    for _ in range(200):
        n = rng.integers(15, 40)
        x, y, z = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        a, b = partial_corr(x, y, z), partial_corr_multi(x, y, z[:, None])
        worst = max(worst, abs(a - b))
    print(f"    max abs difference over 200 random draws: {worst:.2e}")
    if worst > 1e-10:
        print("    FAIL"); ok = False
    else:
        print("    PASS")

    # helper: build a synthetic cohort and run one cell through the real code path
    def synth(effect, mediated_by_strength, n_subjects=40, n_nodes=25):
        subj_ecc, subj_metric, subj_strength, node_size = {}, {}, {}, {}
        nodes = [f"node_{i}" for i in range(n_nodes)]
        for i, n in enumerate(nodes):
            node_size[n] = 100 + 10 * i
        for s in range(n_subjects):
            sid = f"sub-{s:03d}"
            ecc = rng.normal(size=n_nodes)
            strength = rng.normal(size=n_nodes)
            if mediated_by_strength:
                # eccentricity drives strength; strength alone drives the metric.
                # Controlling for strength MUST kill this.
                strength = 0.9 * ecc + 0.44 * rng.normal(size=n_nodes)
                metric = 1.0 * strength + 0.30 * rng.normal(size=n_nodes)
            else:
                metric = effect * ecc + math.sqrt(max(1e-9, 1 - effect**2)) * rng.normal(size=n_nodes)
            subj_ecc[sid] = dict(zip(nodes, ecc))
            subj_metric[sid] = dict(zip(nodes, metric))
            subj_strength[sid] = dict(zip(nodes, strength))
        return subj_ecc, subj_metric, subj_strength, node_size, sorted(subj_ecc)

    def cell(use_strength, **kw):
        se, sm, ss, ns, subs = synth(**kw)
        ns = {s: ns for s in se}   # broadcast atlas-fixed covariate to per-subject form
        return run_one_cell(se, sm, ns, ss, subs, use_strength,
                            min_valid_nodes=15, n_permutations=2000,
                            rng=np.random.default_rng(7))

    print("[2] planted true signal is recovered ...")
    r = cell(True, effect=0.45, mediated_by_strength=False)
    print(f"    mean partial r = {r['mean_partial_r']:+.3f}, perm p = {r['perm_p_value']:.4f}")
    if r["perm_p_value"] < 0.01 and r["mean_partial_r"] > 0.2:
        print("    PASS")
    else:
        print("    FAIL"); ok = False

    print("[3] planted pure null is correctly non-significant ...")
    r = cell(True, effect=0.0, mediated_by_strength=False)
    print(f"    mean partial r = {r['mean_partial_r']:+.3f}, perm p = {r['perm_p_value']:.4f}")
    if r["perm_p_value"] > 0.05:
        print("    PASS")
    else:
        print("    FAIL"); ok = False

    print("[4] strength-mediated signal survives WITHOUT the covariate, dies WITH it ...")
    r_no = cell(False, effect=0.0, mediated_by_strength=True)
    r_yes = cell(True, effect=0.0, mediated_by_strength=True)
    print(f"    size-only       : r = {r_no['mean_partial_r']:+.3f}, perm p = {r_no['perm_p_value']:.4f}")
    print(f"    size + strength : r = {r_yes['mean_partial_r']:+.3f}, perm p = {r_yes['perm_p_value']:.4f}")
    if r_no["perm_p_value"] < 0.01 and r_yes["perm_p_value"] > 0.05:
        print("    PASS  (the script can detect the failure mode it was written to test for)")
    else:
        print("    FAIL"); ok = False

    # 5. vendored helpers agree with the host copy of run_claim2_analysis.py,
    #    when a compatible one is importable
    print("[5] vendored helpers vs. host run_claim2_analysis.py ...")
    try:
        import run_claim2_analysis as host
        n_params = len(inspect.signature(host.load_prf_backbone).parameters)
        if n_params != 3:
            print(f"    SKIP -- host copy is the stale pre-2026-08-13 vintage "
                  f"(load_prf_backbone takes {n_params} args, expected 3). "
                  f"Vendored copies will be used; that is safe.")
        else:
            worst_pc = 0.0
            for _ in range(100):
                n = rng.integers(15, 40)
                x, y, z = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
                worst_pc = max(worst_pc, abs(host.partial_corr(x, y, z) - partial_corr(x, y, z)))
            p = rng.random(12)
            worst_q = float(np.max(np.abs(host.bh_fdr(p) - bh_fdr(p))))
            A, B = rng.normal(size=(20, 30)), rng.normal(size=(20, 30))
            worst_rw = float(np.max(np.abs(host.pearson_rowwise(A, B) - pearson_rowwise(A, B))))
            print(f"    partial_corr max diff {worst_pc:.2e} | bh_fdr {worst_q:.2e} "
                  f"| pearson_rowwise {worst_rw:.2e}")
            if max(worst_pc, worst_q, worst_rw) > 1e-10:
                print("    FAIL"); ok = False
            else:
                print("    PASS")
    except ImportError:
        print("    SKIP -- run_claim2_analysis.py not importable from here (safe; "
              "vendored copies are self-contained)")

    print()
    print("SELF-TEST:", "ALL PASSED" if ok else "FAILURES ABOVE -- do not trust real output")
    return 0 if ok else 1


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run fixture checks with no real data and exit")
    ap.add_argument("--prf-backbone-csv")
    ap.add_argument("--controllability-csv")
    ap.add_argument("--node-strength-csv")
    ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="schaefer200", choices=("schaefer200", "glasser"))
    ap.add_argument("--base-covariate", default="size", choices=("size", "rfsize"),
                    help="Covariate that node strength is added ON TOP OF. "
                         "'size' = n_vertices_total (raw vertex count, the original "
                         "Claim 2 covariate). 'rfsize' = mean_rfsize_deg, the "
                         "cortical-magnification proxy validated in Phase B3 and used "
                         "for the manuscript's headline Claim 2 numbers. Following "
                         "run_claim2_rfsize_permutation.py, rfsize REPLACES size rather "
                         "than being added alongside it. Default: size.")
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    for req in ("prf_backbone_csv", "controllability_csv", "node_strength_csv", "out_csv"):
        if not getattr(args, req):
            ap.error(f"--{req.replace('_', '-')} is required (or use --self-test)")

    rng = np.random.default_rng(args.seed)

    cross_check_against_host(args.prf_backbone_csv, args.min_vertices_per_node, args.atlas)

    subj_ecc, node_size = load_prf_backbone(args.prf_backbone_csv,
                                            args.min_vertices_per_node, args.atlas)
    subj_avg_log, subj_modal = load_controllability(args.controllability_csv)
    subj_strength = load_node_strength(args.node_strength_csv)
    subjects = sorted(subj_ecc)
    print(f"Atlas: {args.atlas} | subjects with eccentricity data: {len(subjects)} | "
          f"Vis-network nodes in atlas: {len(node_size)}")

    check_inputs_intersect(subj_ecc, node_size, subj_avg_log, subj_modal,
                           subj_strength, args.atlas)

    # Assemble the base covariate as dict[subject][node]. Vertex count is fixed
    # by the atlas, so it is broadcast to every subject; rfsize is genuinely
    # per-subject and is loaded from the same pRF file.
    if args.base_covariate == "rfsize":
        subj_base = load_prf_rfsize(args.prf_backbone_csv,
                                    args.min_vertices_per_node, args.atlas)
        n_cov_nodes = len({n for d in subj_base.values() for n in d})
        print(f"Base covariate: mean_rfsize_deg (per-subject) -- "
              f"{len(subj_base)} subjects, {n_cov_nodes} nodes")
        if not subj_base:
            sys.exit("*** INPUT ERROR: --base-covariate rfsize requested but no usable "
                     "mean_rfsize_deg values were found in the pRF backbone file. ***")
    else:
        subj_base = {s: node_size for s in subj_ecc}
        print(f"Base covariate: n_vertices_total (atlas-fixed, broadcast to all subjects)")

    # report the collinearity that motivates this script, on the real data
    print("\nObserved controllability-vs-strength collinearity (within-subject Spearman, "
          "analysis node set):")
    for label, metric in (("average (log)", subj_avg_log), ("modal", subj_modal)):
        rhos = []
        for s in subjects:
            common = sorted(set(metric.get(s, {})) & set(subj_strength.get(s, {})) & set(node_size))
            if len(common) < args.min_valid_nodes:
                continue
            m = [metric[s][n] for n in common]
            w = [subj_strength[s][n] for n in common]
            rhos.append(stats.spearmanr(w, m).statistic)
        if rhos:
            print(f"  {label:14s} median rho = {np.median(rhos):+.3f}  "
                  f"[{np.min(rhos):+.3f}, {np.max(rhos):+.3f}]  (n={len(rhos)})")

    rows = []
    for metric_name, metric in (("average_controllability_log", subj_avg_log),
                                ("modal_controllability", subj_modal)):
        for use_strength in (False, True):
            base = args.base_covariate
            cfg = f"{base}_plus_strength" if use_strength else f"{base}_only"
            print(f"\n=== {args.atlas} | {metric_name} | covariates: {cfg} ===")
            res = run_one_cell(subj_ecc, metric, subj_base, subj_strength, subjects,
                               use_strength, args.min_valid_nodes,
                               args.n_permutations, rng)
            res.update(atlas=args.atlas, metric=metric_name, covariate_config=cfg,
                       base_covariate=base)
            rows.append(res)
            print(json.dumps({k: v for k, v in res.items()
                              if k not in ("atlas", "metric", "covariate_config")}, indent=2))

    # BH-FDR across this atlas's 4-test family (combine the two atlases'
    # output files afterwards for the full 8-test family if reporting jointly)
    pvals = [r.get("perm_p_value") for r in rows]
    if all(p is not None and np.isfinite(p) for p in pvals):
        for r, q in zip(rows, bh_fdr(pvals)):
            r["q_value_perm_within_atlas"] = float(q)

    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.out_csv} ({len(rows)} rows)")

    base = args.base_covariate
    ref = ("run_claim2_analysis.py's reported Claim 2 numbers" if base == "size"
           else "claim2_rfsize_permutation_*.csv's 'rfsize_only' rows")
    print(f"\nREGRESSION CHECK: the two '{base}_only' rows above should reproduce "
          f"{ref} for this atlas. If they do not, this script is wrong and the "
          f"'{base}_plus_strength' rows should not be trusted.")


if __name__ == "__main__":
    main()
