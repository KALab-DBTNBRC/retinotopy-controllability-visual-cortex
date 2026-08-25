#!/usr/bin/env python3
"""
Claim 4 -- Sex x receptor x controllability interaction.

"The well-established sex difference in whole-brain controllability (higher
modal controllability in females, higher average controllability in males)
will be present in visual cortex and will be partly explained by
sex-invariant receptor density gradients interacting with sex-differentiated
connectivity" (lit-review.docx, Claim 4).

Decomposes into two genuinely different sub-questions, tested separately
with their own FDR families (user decision 2026-08-12 -- treating them as
one pooled family would dilute power for two unrelated hypotheses):

  (A) SEX DIFFERENCE (main effect): does the visual-cortex controllability
      sex difference replicate the whole-brain literature pattern? Simple
      two-group (M vs F) comparison of subject-level mean controllability.
      Reported under BOTH visual-system node-set definitions used elsewhere
      in this project (Claim 1's 8 hOc units, and Claim 2/3's 29 Vis-network
      Schaefer-200 nodes) -- user decision 2026-08-12, same dual-reporting
      approach as Claim 3. Welch's t-test (unequal-variance, the safer
      default for a possibly-unequal-N M/F split) + Mann-Whitney U.

  (B) SEX x RECEPTOR INTERACTION: does the receptor-density/controllability
      relationship found (or not found) in Claim 1 differ by sex? User
      explicitly requested a formal moderated regression here (2026-08-12),
      not a simpler between-group comparison of Claim 1's already-computed
      per-subject partial-r values. Necessarily restricted to the hOc
      8-unit definition -- receptor density has no meaning for the broader
      Vis-network node set (Julich-Brain receptor autoradiography only
      covers hOc1/hOc2/hOc3v/hOc3d, same limitation as Claim 1).

      Model: controllability ~ receptor_density + sex + receptor_density:sex
      + n_hoc_vertices, pooled across all (subject, hOc unit) rows. This
      pools up to 46 subjects x 8 units = up to 368 rows, which are NOT
      independent observations -- each subject contributes a correlated
      block of up to 8 rows (same non-independence problem Claim 1's
      per-subject-then-t-test design was built to respect). A naive pooled
      OLS p-value would be anti-conservative (pseudo-replication). Rather
      than introduce a new mixed-model dependency, this reuses the same
      permutation-test philosophy already validated in Claims 1/2: sex
      labels are reshuffled at the SUBJECT level (each subject's own
      8-row block of real receptor/controllability/size data stays intact,
      only the M/F label attached to that whole block is shuffled),
      preserving the real clustered/repeated-measures structure exactly,
      then the same pooled OLS is refit and the interaction coefficient
      recomputed, 10,000 times, to build an empirical null. This is the
      PRIMARY significance test for the interaction coefficient, same
      "empirical null over parametric assumption" reasoning as Claims 1/2.

average_controllability is used in log-space throughout (same rationale as
Claims 1-3: diagnose_connectome_hemisphere_asymmetry.py's finding that its
heavy-tailed distribution is concentrated in visual cortex).

IMPORTANT interpretive caveat, found via fixture testing (2026-08-12): the
sex x receptor interaction permutation test was independently calibration-
checked (calibrate_claim4_interaction.py, 300 true-null replicates with
freshly-redrawn receptor vectors each time) and is well-calibrated (3.3%
false-positive rate at alpha=0.05, KS-test p=0.52 against Uniform(0,1) --
no evidence of miscalibration). BUT a separate fixture test exposed a real,
expected consequence of only having 8 fixed spatial units: two
independently-drawn receptor-density vectors can coincidentally correlate
with each other at n=8 (a synthetic test case hit r=-0.32 between two
supposedly-unrelated receptors purely by chance) -- and when that happens, a
REAL sex-dependent interaction driven by one receptor's true spatial pattern
can partially "leak" into an apparent (and statistically well-calibrated,
not spuriously inflated -- the math is doing exactly what it should)
interaction result for the OTHER, uninvolved receptor, simply because their
density vectors are collinear. This is the same small-spatial-N limitation
already documented in Claim 1's docstring, showing up in a new form. If
multiple receptors come back significant here, check their real pairwise
density correlations at n=8 (a single line of code) before concluding they
are independent findings -- correlated receptors showing correlated
"interaction" results may reflect one true driver, not several.

Usage:
    python3 run_claim4_analysis.py \
        --master-table-xlsx /path/to/master-table.xlsx \
        --hoc-region-controllability-csv /path/to/hoc_region_controllability/schaefer200.csv \
        --receptor-csv /path/to/receptor-densities-deduped.csv \
        --controllability-csv /path/to/all_subjects_controllability_schaefer200.csv \
        --out-csv-sexdiff /path/to/claim4_sexdiff_schaefer200.csv \
        --out-csv-interaction /path/to/claim4_interaction_schaefer200.csv
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

REGION_ORDER = ["hOc1", "hOc2", "hOc3v", "hOc3d"]
HEMI_ORDER = ["left", "right"]
UNIT_ORDER = [f"{r}_{h}" for r in REGION_ORDER for h in HEMI_ORDER]


def load_sex(path):
    """Returns {subject: 0/1}, F=1, M=0 (arbitrary, documented here)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Subjects"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    if "Sex" not in idx or "Subject ID" not in idx:
        raise ValueError("master-table.xlsx Subjects sheet missing 'Sex' or 'Subject ID' column")
    out = {}
    for r in rows[1:]:
        if r[idx["Subject ID"]] is None or r[idx["Sex"]] is None:
            continue
        sex_raw = str(r[idx["Sex"]]).strip().upper()
        if sex_raw not in ("M", "F"):
            raise ValueError(f"Unexpected Sex value: {sex_raw!r} for subject {r[idx['Subject ID']]}")
        out[str(r[idx["Subject ID"]])] = 1 if sex_raw == "F" else 0
    return out


def load_hoc_region_controllability(path):
    """Returns subj_avg_log[subject][hoc_name], subj_modal[subject][hoc_name],
    subj_size[subject][hoc_name] -- same as run_claim1_analysis.py."""
    subj_avg_log = defaultdict(dict)
    subj_modal = defaultdict(dict)
    subj_size = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "weighted_avg_controllability_log" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing weighted_avg_controllability_log -- rebuild with current script.")
        for row in reader:
            s, h = row["subject"], row["hoc_name"]
            log_val = row.get("weighted_avg_controllability_log", "")
            if log_val not in ("", None):
                subj_avg_log[s][h] = float(log_val)
            subj_modal[s][h] = float(row["weighted_modal_controllability"])
            subj_size[s][h] = float(row["n_hoc_vertices"])
    return subj_avg_log, subj_modal, subj_size


def load_vis_summary(path, atlas="schaefer200"):
    """Returns {subject: (avg_log_mean, modal_mean)} across all Vis-network-
    equivalent nodes -- same as run_claim3_analysis.py's load_vis_summary.
    Atlas-aware (2026-08-13 fix, closing the Claim 4 Glasser gap): this was
    hardcoded to Schaefer's '_Vis_' label convention, so a real Glasser
    rerun silently returned 0 subjects for this arm ('too few subjects in
    one group, skipped') rather than erroring -- easy to miss. Reuses
    run_claim2_analysis.py's already-verified GLASSER_VIS_AREAS/
    is_glasser_vis_node (2026-08-13 decision, 52/52 real label matches
    confirmed there) for Glasser, exactly the same fix already applied to
    run_claim3_analysis.py's vis arm in Claim 5."""
    if atlas == "glasser":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_claim2_analysis as c2
        is_vis = lambda node_label: c2.is_glasser_vis_node(node_label)
    else:
        is_vis = lambda node_label: "_Vis_" in node_label

    per_subj_avg_log = defaultdict(list)
    per_subj_modal = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_vis(row["node_label"]):
                continue
            avg_c = float(row["average_controllability"])
            per_subj_modal[row["subject"]].append(float(row["modal_controllability"]))
            if avg_c > 0:
                per_subj_avg_log[row["subject"]].append(math.log(avg_c))
    out = {}
    for s in set(per_subj_avg_log) | set(per_subj_modal):
        avg_vals = per_subj_avg_log.get(s, [])
        modal_vals = per_subj_modal.get(s, [])
        out[s] = (
            float(np.mean(avg_vals)) if avg_vals else None,
            float(np.mean(modal_vals)) if modal_vals else None,
        )
    return out


def load_receptor_vectors(path):
    out = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hoc_name = f"{row['region']}_{row['hemisphere']}"
            out[row["receptor"]][hoc_name] = float(row["mean_density"])
    return out


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


# ---------- Sub-claim A: sex difference in visual-cortex controllability ----------

def run_sex_difference(subject_means, sex_map):
    """subject_means: {subject: value}. Returns dict of Welch's t-test +
    Mann-Whitney U stats comparing F (sex_map==1) vs M (sex_map==0)."""
    f_vals = [v for s, v in subject_means.items() if s in sex_map and sex_map[s] == 1]
    m_vals = [v for s, v in subject_means.items() if s in sex_map and sex_map[s] == 0]
    if len(f_vals) < 5 or len(m_vals) < 5:
        return {"n_f": len(f_vals), "n_m": len(m_vals), "note": "too few subjects in one group, skipped"}
    t_stat, p_t = stats.ttest_ind(f_vals, m_vals, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(f_vals, m_vals, alternative="two-sided")
    return {
        "n_f": len(f_vals), "n_m": len(m_vals),
        "mean_f": float(np.mean(f_vals)), "mean_m": float(np.mean(m_vals)),
        "t_stat": float(t_stat), "p_ttest": float(p_t),
        "u_stat": float(u_stat), "p_mannwhitney": float(p_u),
        "note": "",
    }


# ---------- Sub-claim B: sex x receptor moderated regression ----------

def build_design_matrix(rows):
    """rows: list of (receptor_density, sex, size, y). Returns X (n,5) with
    columns [intercept, receptor, sex, receptor:sex, size], y (n,)."""
    receptor = np.array([r[0] for r in rows])
    sex = np.array([r[1] for r in rows])
    size = np.array([r[2] for r in rows])
    y = np.array([r[3] for r in rows])
    X = np.column_stack([np.ones(len(rows)), receptor, sex, receptor * sex, size])
    return X, y


def ols_interaction_coef(X, y):
    """Closed-form OLS via lstsq. Returns the receptor:sex interaction
    coefficient (column index 3)."""
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coefs[3]


def run_sex_receptor_interaction(subject_rows, sex_map, n_permutations, rng):
    """subject_rows: {subject: [(receptor_density, size, y), ...]} (one entry
    per hOc unit that subject has valid data for). Builds the pooled design
    matrix using each subject's real sex label, fits OLS, gets the observed
    interaction coefficient, then permutes SEX LABELS AT THE SUBJECT LEVEL
    (each subject's own row-block stays intact) to build an empirical null.
    """
    subjects = [s for s in subject_rows if s in sex_map and len(subject_rows[s]) > 0]
    if len(subjects) < 10:
        return {"n_subjects": len(subjects), "note": "too few subjects, skipped"}

    def build_rows(sex_assignment):
        rows = []
        for s in subjects:
            sx = sex_assignment[s]
            for (receptor_d, size, y) in subject_rows[s]:
                rows.append((receptor_d, sx, size, y))
        return rows

    real_sex = {s: sex_map[s] for s in subjects}
    X, y = build_design_matrix(build_rows(real_sex))
    if np.std(X[:, 2]) == 0:  # sex has no variance (all one group) -- can't fit interaction
        return {"n_subjects": len(subjects), "note": "sex has no variance in this subset, skipped"}
    observed_coef = ols_interaction_coef(X, y)

    sex_labels = np.array([real_sex[s] for s in subjects])
    null_coefs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_labels = rng.permutation(sex_labels)
        perm_sex_assignment = {s: perm_labels[j] for j, s in enumerate(subjects)}
        Xp, yp = build_design_matrix(build_rows(perm_sex_assignment))
        null_coefs[i] = ols_interaction_coef(Xp, yp)

    perm_p = (np.sum(np.abs(null_coefs) >= abs(observed_coef)) + 1) / (n_permutations + 1)
    n_f = int(np.sum(sex_labels == 1))
    n_m = int(np.sum(sex_labels == 0))

    return {
        "n_subjects": len(subjects), "n_f": n_f, "n_m": n_m,
        "n_rows_pooled": len(y),
        "interaction_coef": float(observed_coef),
        "perm_p_value": float(perm_p),
        "n_permutations_used": n_permutations,
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-table-xlsx", required=True)
    ap.add_argument("--hoc-region-controllability-csv", required=True)
    ap.add_argument("--receptor-csv", required=True)
    ap.add_argument("--controllability-csv", required=True,
                     help="all_subjects_controllability_schaefer200.csv, for the Vis-network sex-difference arm")
    ap.add_argument("--atlas", default="schaefer200", choices=("schaefer200", "glasser"),
                     help="which atlas's Vis-network-equivalent node set to use for the vis arm "
                          "(2026-08-13 fix, see load_vis_summary docstring)")
    ap.add_argument("--out-csv-sexdiff", required=True)
    ap.add_argument("--out-csv-interaction", required=True)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_csv_sexdiff).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv_interaction).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    sex_map = load_sex(args.master_table_xlsx)
    subj_avg_log, subj_modal, subj_size = load_hoc_region_controllability(args.hoc_region_controllability_csv)
    vis_summary = load_vis_summary(args.controllability_csv, atlas=args.atlas)
    receptor_vectors = load_receptor_vectors(args.receptor_csv)

    n_f = sum(1 for v in sex_map.values() if v == 1)
    n_m = sum(1 for v in sex_map.values() if v == 0)
    print(f"{len(sex_map)} subjects with sex data ({n_f}F / {n_m}M)")

    # ---------- Sub-claim A ----------
    hoc_avg_mean = {s: float(np.mean(list(subj_avg_log[s].values()))) for s in subj_avg_log if subj_avg_log[s]}
    hoc_modal_mean = {s: float(np.mean(list(subj_modal[s].values()))) for s in subj_modal if subj_modal[s]}
    vis_avg_mean = {s: v[0] for s, v in vis_summary.items() if v[0] is not None}
    vis_modal_mean = {s: v[1] for s, v in vis_summary.items() if v[1] is not None}

    sexdiff_results = []
    for label, means in [("avg_controllability_log__hoc", hoc_avg_mean),
                          ("modal_controllability__hoc", hoc_modal_mean),
                          ("avg_controllability_log__vis", vis_avg_mean),
                          ("modal_controllability__vis", vis_modal_mean)]:
        stat = run_sex_difference(means, sex_map)
        row = {"controllability_summary": label}
        row.update(stat)
        sexdiff_results.append(row)

    p_values_a = [r["p_ttest"] for r in sexdiff_results if "p_ttest" in r]
    q_values_a = bh_fdr(p_values_a) if p_values_a else []
    qi = 0
    for r in sexdiff_results:
        if "p_ttest" in r:
            r["q_value_fdr"] = float(q_values_a[qi]); qi += 1
        else:
            r["q_value_fdr"] = None

    fieldnames_a = ["controllability_summary", "n_f", "n_m", "mean_f", "mean_m",
                     "t_stat", "p_ttest", "q_value_fdr", "u_stat", "p_mannwhitney", "note"]
    with open(args.out_csv_sexdiff, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_a)
        writer.writeheader()
        for r in sexdiff_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames_a})

    print(f"\n=== Sub-claim A: sex difference (FDR q < 0.05) ===")
    sig_a = [r for r in sexdiff_results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    for r in sig_a:
        print(f"  {r['controllability_summary']}: F mean={r['mean_f']:.3f}, M mean={r['mean_m']:.3f}, "
              f"p={r['p_ttest']:.4g}, q={r['q_value_fdr']:.4g}")
    if not sig_a:
        print("  (none)")

    # ---------- Sub-claim B ----------
    interaction_results = []
    for receptor, vec in sorted(receptor_vectors.items()):
        if not all(u in vec for u in UNIT_ORDER):
            print(f"receptor={receptor}: SKIPPED -- missing one of {UNIT_ORDER} in receptor data")
            continue
        for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                     ("modal_controllability", subj_modal)]:
            subject_rows = {}
            for s in subj_c:
                rows = []
                for u in UNIT_ORDER:
                    if u in subj_c[s] and u in subj_size[s]:
                        rows.append((vec[u], subj_size[s][u], subj_c[s][u]))
                if rows:
                    subject_rows[s] = rows
            stat = run_sex_receptor_interaction(subject_rows, sex_map, args.n_permutations, rng)
            row = {"receptor": receptor, "metric": metric_name}
            row.update(stat)
            interaction_results.append(row)

    perm_p_values = [r["perm_p_value"] for r in interaction_results if "perm_p_value" in r]
    q_values_b = bh_fdr(perm_p_values) if perm_p_values else []
    qi = 0
    for r in interaction_results:
        if "perm_p_value" in r:
            r["q_value_fdr_perm"] = float(q_values_b[qi]); qi += 1
        else:
            r["q_value_fdr_perm"] = None

    fieldnames_b = ["receptor", "metric", "n_subjects", "n_f", "n_m", "n_rows_pooled",
                     "interaction_coef", "perm_p_value", "n_permutations_used", "q_value_fdr_perm", "note"]
    with open(args.out_csv_interaction, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_b)
        writer.writeheader()
        for r in interaction_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames_b})

    print(f"\n=== Sub-claim B: sex x receptor interaction (permutation FDR q < 0.05) ===")
    sig_b = [r for r in interaction_results if r.get("q_value_fdr_perm") is not None and r["q_value_fdr_perm"] < 0.05]
    if sig_b:
        for r in sorted(sig_b, key=lambda x: x["q_value_fdr_perm"]):
            print(f"  {r['receptor']} / {r['metric']}: coef={r['interaction_coef']:.4g}, "
                  f"perm_p={r['perm_p_value']:.4g}, q={r['q_value_fdr_perm']:.4g}, n={r['n_subjects']}")
    else:
        print("  (none)")

    print(f"\nSub-claim A written to {args.out_csv_sexdiff}")
    print(f"Sub-claim B written to {args.out_csv_interaction}")


if __name__ == "__main__":
    main()
