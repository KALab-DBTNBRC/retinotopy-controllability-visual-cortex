#!/usr/bin/env python3
"""
Claim 2, Phase C2 -- does pRF polar angle (not just eccentricity) relate to
controllability? Motivated directly by the lit review (Griffis et al. 2015's
V1-fronto-parietal connectivity gradient isn't purely eccentricity-driven;
"A retinotopic wiring principle of the human brain" reconstructs visual
pathways as a function of BOTH eccentricity and polar angle).

Polar angle is CIRCULAR data (0/360 deg wraparound) with no natural
"low-to-high" ordering the way eccentricity has -- a standard Pearson
correlation is not meaningful. Uses the standard circular-linear correlation
coefficient (Mardia 1972; Fisher & Lee 1983):
    r_cl^2 = (r_xc^2 + r_xs^2 - 2*r_xc*r_xs*r_cs) / (1 - r_cs^2)
where c=cos(theta), s=sin(theta), x=the linear variable (controllability,
size-residualized first, same size confound as the main eccentricity
analysis). r_cl^2 is in [0,1] and has NO sign/direction (a variable can be
strongly linked to angle position without a single "increases with angle"
direction, e.g. higher at both 0 and 180 deg, lower at 90/270) -- this is a
real interpretive difference from the eccentricity analysis's signed r,
noted explicitly in the output.

Verified against a synthetic fixture before trusting on real data (see
test_circular_linear_corr_fixture.py): a variable that peaks at one angle
and troughs at the opposite (y = cos(theta - phi0) + noise) gives a high
r_cl^2; a variable independent of angle gives r_cl^2 near the noise floor.

Per-node circular_concentration_r (from aggregate_prf_to_backbone.py, mean
resultant length of the underlying per-vertex angles) is used as a real
per-entry quality filter: nodes/subjects whose own angle summary is
unreliable (scattered vertex-level angles, low concentration) are excluded,
same spirit as min_vertices_per_node already excludes thin eccentricity
estimates.

Group-level significance uses the SAME per-subject-independent permutation
design as run_claim2_analysis.py (each subject's own angle values reshuffled
across their own nodes, controllability/size keep their real node identity).

Usage:
    python3 run_claim2_polar_angle.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv all_subjects_controllability_glasser.csv \
        --atlas glasser --out-csv claim2_polar_angle_glasser.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def circular_linear_corr(theta_deg, x):
    """Mardia/Fisher-Lee circular-linear correlation. Returns r_cl^2 in [0,1]."""
    theta = np.radians(np.asarray(theta_deg, dtype=float))
    x = np.asarray(x, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    if np.std(c) == 0 or np.std(s) == 0 or np.std(x) == 0:
        return np.nan
    r_xc = stats.pearsonr(x, c)[0]
    r_xs = stats.pearsonr(x, s)[0]
    r_cs = stats.pearsonr(c, s)[0]
    denom = 1 - r_cs ** 2
    if denom <= 0:
        return np.nan
    r2 = (r_xc ** 2 + r_xs ** 2 - 2 * r_xc * r_xs * r_cs) / denom
    return float(np.clip(r2, 0, 1))


def size_residualize(y, z):
    Z = np.column_stack([np.ones_like(z), z])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ coef


def load_angle_backbone(path, min_vertices_per_node, min_concentration, atlas, is_vis_fn):
    subj_angle, subj_concentration, node_size = {}, {}, {}
    n_dropped_thin, n_dropped_scattered = 0, 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["atlas"] != atlas:
                continue
            is_vis = (row["is_vis_network"] == "True") if atlas == "schaefer200" else is_vis_fn(row["backbone_label"])
            if not is_vis:
                continue
            node_size[row["backbone_label"]] = int(row["n_vertices_total"])
            if row["mean_polar_angle_deg"] == "" or row["circular_concentration_r"] == "":
                continue
            n_valid = int(row["n_vertices_valid_R2gt30"])
            if n_valid < min_vertices_per_node:
                n_dropped_thin += 1
                continue
            conc = float(row["circular_concentration_r"])
            if conc < min_concentration:
                n_dropped_scattered += 1
                continue
            s = row["subject"]
            subj_angle.setdefault(s, {})[row["backbone_label"]] = float(row["mean_polar_angle_deg"])
            subj_concentration.setdefault(s, {})[row["backbone_label"]] = conc
    print(f"Dropped {n_dropped_thin} entries (< {min_vertices_per_node} R2>30 vertices), "
          f"{n_dropped_scattered} entries (circular_concentration_r < {min_concentration}, "
          f"unreliable angle summary)")
    return subj_angle, node_size


def permuted_circular_linear_r2_vectorized(theta_deg, y_resid, n_permutations, rng, pearson_rowwise):
    """Vectorized per-subject permutation null (same performance pattern as
    run_claim2_analysis.py's subject_permuted_partial_r / pearson_rowwise --
    a naive per-permutation scipy.stats.pearsonr loop was measured too slow
    at 10,000 permutations x ~40 subjects, see module change notes). reshuffle
    theta's node positions, y_resid keeps its real node identity, recompute
    r_cl^2 for every permutation at once via matrix row-wise correlation.

    Key insight: r_cs (correlation between cos(theta) and sin(theta) across
    nodes) is UNCHANGED by permuting theta's assignment to nodes, since a
    permutation reorders both cos and sin by the same index -- it doesn't
    change which cos value is paired with which sin value. So r_cs is
    computed once, not per-permutation."""
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    r_cs = np.corrcoef(c, s)[0, 1]
    denom = 1 - r_cs ** 2
    if denom <= 0:
        return np.full(n_permutations, np.nan)

    k = len(theta_deg)
    keys = rng.random((n_permutations, k))
    perm_idx = np.argsort(keys, axis=1)
    C_perm = c[perm_idx]  # (n_permutations, k)
    S_perm = s[perm_idx]
    Y_mat = np.tile(y_resid, (n_permutations, 1))

    r_xc = pearson_rowwise(C_perm, Y_mat)
    r_xs = pearson_rowwise(S_perm, Y_mat)
    r2 = (r_xc ** 2 + r_xs ** 2 - 2 * r_xc * r_xs * r_cs) / denom
    return np.clip(r2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--prf-backbone-csv", required=True)
    ap.add_argument("--controllability-csv", required=True)
    ap.add_argument("--atlas", required=True, choices=("schaefer200", "glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--min-concentration", type=float, default=0.5,
                     help="minimum circular_concentration_r to trust a node's angle summary")
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.script_dir)
    import run_claim2_analysis as c2

    rng = np.random.default_rng(args.seed)
    subj_angle, node_size = load_angle_backbone(args.prf_backbone_csv, args.min_vertices_per_node,
                                                 args.min_concentration, args.atlas, c2.is_glasser_vis_node)
    subj_avg_log, subj_modal = c2.load_controllability(args.controllability_csv)
    subjects = sorted(subj_angle.keys())
    print(f"{len(subjects)} subjects with polar-angle backbone data (post quality filters)")

    results = []
    for metric_name, subj_c in [("average_controllability_log", subj_avg_log),
                                 ("modal_controllability", subj_modal)]:
        per_subject_r2, used_subjects, subj_null_arrays = [], [], []
        for s in subjects:
            ang_vec = subj_angle.get(s, {})
            c_vec = subj_c.get(s, {})
            common_nodes = sorted(set(ang_vec) & set(c_vec) & set(node_size))
            if len(common_nodes) < args.min_valid_nodes:
                continue
            theta = np.array([ang_vec[n] for n in common_nodes])
            y = np.array([c_vec[n] for n in common_nodes])
            z = np.array([node_size[n] for n in common_nodes])
            if np.std(theta) == 0 or np.std(y) == 0:
                continue
            y_resid = size_residualize(y, z)
            r2 = circular_linear_corr(theta, y_resid)
            if not np.isfinite(r2):
                continue
            per_subject_r2.append(r2)
            used_subjects.append(s)
            null = permuted_circular_linear_r2_vectorized(theta, y_resid, args.n_permutations, rng,
                                                            c2.pearson_rowwise)
            subj_null_arrays.append(null)

        n_valid = len(used_subjects)
        if n_valid < 5:
            print(f"{metric_name}: too few valid subjects ({n_valid}), skipped")
            results.append({"metric": metric_name, "n_subjects_valid": n_valid, "note": "too few valid subjects"})
            continue

        observed_mean_r2 = float(np.mean(per_subject_r2))
        null_stack = np.vstack(subj_null_arrays)
        null_mean_r2 = np.nanmean(null_stack, axis=0)
        null_mean_r2 = null_mean_r2[np.isfinite(null_mean_r2)]
        # one-sided: r_cl^2 is non-negative by construction, larger = more relationship
        perm_p = (np.sum(null_mean_r2 >= observed_mean_r2) + 1) / (len(null_mean_r2) + 1)

        print(f"{metric_name}: n={n_valid}, mean_r_cl^2={observed_mean_r2:.4f}, perm_p={perm_p:.5g}")
        results.append({
            "metric": metric_name, "n_subjects_valid": n_valid,
            "mean_circular_linear_r2": observed_mean_r2, "perm_p_value": float(perm_p),
            "n_permutations_used": args.n_permutations, "note": "",
        })

    perm_p_values = [r["perm_p_value"] for r in results if r.get("perm_p_value") is not None]
    q_values = c2.bh_fdr(perm_p_values) if perm_p_values else []
    qi = 0
    for r in results:
        if r.get("perm_p_value") is not None:
            r["q_value_fdr_perm"] = float(q_values[qi])
            qi += 1
        else:
            r["q_value_fdr_perm"] = None

    out_csv = Path(args.out_csv)
    fieldnames = ["metric", "n_subjects_valid", "mean_circular_linear_r2", "perm_p_value",
                  "n_permutations_used", "q_value_fdr_perm", "note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Written to {out_csv}")


if __name__ == "__main__":
    main()
