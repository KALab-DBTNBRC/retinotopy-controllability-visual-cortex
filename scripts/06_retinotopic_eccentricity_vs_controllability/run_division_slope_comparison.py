#!/usr/bin/env python3
"""
Direct comparison of the eccentricity-controllability effect BETWEEN Glasser
visual divisions, as a within-participant paired test.

WHY THIS EXISTS
---------------
run_claim2_region_breakdown.py tests each division separately and applies FDR
across the 8-test family. That answers "is there evidence within this
division?" but NOT "is the effect stronger in this division than in that one?".

Those are different questions, and conflating them produced a real error in an
earlier draft. In the per-division output, Early Visual and Dorsal Stream
Visual have essentially identical average-controllability estimates
(partial r = 0.357 vs 0.359) yet q < .001 vs q = .558. The whole difference is
retention: Early Visual keeps 45 participants with a median of 6 nodes, Dorsal
Stream keeps 31 with a median of 4. Inferring spatial concentration from that
gap infers a difference in effect from a difference in evidence.

WHAT THIS DOES INSTEAD
----------------------
For each participant, compute the per-participant partial correlation
(eccentricity vs controllability, covarying node vertex count) separately
within each division. Then, for each PAIR of divisions, restrict to the
participants who have a valid estimate in BOTH and test the paired difference.

Because the same participants contribute both estimates, between-division
differences in retention cannot drive the result. The test is a sign-flip
permutation on the per-participant differences, which makes no distributional
assumption and matches the permutation-first convention used throughout this
project.

Helper functions are VENDORED rather than imported. An earlier server run of a
dependent script failed because the adjacent run_claim2_analysis.py was a
stale vintage with a different loader signature; vendoring makes this script's
results independent of which copy sits beside it.

Usage:
    python3 run_division_slope_comparison.py \
        --prf-backbone-csv all_subjects_prf_backbone.csv \
        --controllability-csv "Received files/all_subjects_controllability_glasser.csv" \
        --out-csv division_slope_comparison_glasser.csv
    python3 run_division_slope_comparison.py --self-test
"""
import argparse, csv, math, re, sys
from collections import defaultdict

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- vendored --
GLASSER_DIVISIONS = {
    "Primary_Visual": {"V1"},
    "Early_Visual": {"V2", "V3", "V4"},
    "Dorsal_Stream_Visual": {"V3A", "V3B", "V6", "V6A", "V7", "IPS1"},
    "Ventral_Stream_Visual": {"FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC"},
    "MT+_Complex": {"FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t"},
}
GLASSER_VIS_AREAS = set().union(*GLASSER_DIVISIONS.values())
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def area_of(node_label):
    m = GLASSER_LABEL_RE.match(node_label)
    return m.group(1) if m else None


def division_of(node_label):
    a = area_of(node_label)
    if a is None:
        return None
    for div, areas in GLASSER_DIVISIONS.items():
        if a in areas:
            return div
    return None


def is_glasser_vis_node(node_label):
    a = area_of(node_label)
    return a is not None and a in GLASSER_VIS_AREAS


def load_prf_backbone(path, min_vertices_per_node, atlas="glasser"):
    subj_ecc, node_size = defaultdict(dict), {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["atlas"] != atlas:
                continue
            if not is_glasser_vis_node(row["backbone_label"]):
                continue
            node_size[row["backbone_label"]] = int(row["n_vertices_total"])
            if row["mean_eccentricity_deg"] == "":
                continue
            if int(row["n_vertices_valid_R2gt30"]) < min_vertices_per_node:
                continue
            subj_ecc[row["subject"]][row["backbone_label"]] = float(row["mean_eccentricity_deg"])
    return subj_ecc, node_size


def load_controllability(path):
    subj_avg_log, subj_modal = defaultdict(dict), defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s, n = row["subject"], row["node_label"]
            a = float(row["average_controllability"])
            subj_modal[s][n] = float(row["modal_controllability"])
            if a > 0:
                subj_avg_log[s][n] = math.log(a)
    return subj_avg_log, subj_modal


def partial_corr(x, y, z):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    rxy, rxz, ryz = stats.pearsonr(x, y)[0], stats.pearsonr(x, z)[0], stats.pearsonr(y, z)[0]
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return np.nan if den == 0 or not np.isfinite(den) else (rxy - rxz * ryz) / den


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1)
    return out


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


# ------------------------------------------------------------------- core --
def per_subject_division_r(subj_ecc, subj_metric, node_size, division, min_nodes):
    """{subject: partial_r} within one division, for subjects meeting min_nodes."""
    div_nodes = {n for n in node_size if division_of(n) == division}
    out = {}
    for s, ecc in subj_ecc.items():
        c = subj_metric.get(s, {})
        common = sorted(set(ecc) & set(c) & div_nodes)
        if len(common) < min_nodes:
            continue
        r = partial_corr([ecc[n] for n in common],
                         [c[n] for n in common],
                         [node_size[n] for n in common])
        if np.isfinite(r):
            out[s] = r
    return out


def paired_signflip_test(diff, n_perm, rng):
    """Two-sided sign-flip permutation on paired differences of Fisher z."""
    diff = np.asarray(diff, float)
    obs = diff.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def run(subj_ecc, subj_metric, node_size, metric_name, min_nodes, n_perm, rng):
    divisions = [d for d in GLASSER_DIVISIONS if d != "Primary_Visual"]
    per_div = {d: per_subject_division_r(subj_ecc, subj_metric, node_size, d, min_nodes)
               for d in divisions}
    rows = []
    for i, a in enumerate(divisions):
        for b in divisions[i + 1:]:
            shared = sorted(set(per_div[a]) & set(per_div[b]))
            if len(shared) < 5:
                rows.append({"metric": metric_name, "division_a": a, "division_b": b,
                             "n_paired": len(shared), "note": "too few paired subjects"})
                continue
            za = fisher_z([per_div[a][s] for s in shared])
            zb = fisher_z([per_div[b][s] for s in shared])
            d = za - zb
            obs, p = paired_signflip_test(d, n_perm, rng)
            rows.append({
                "metric": metric_name, "division_a": a, "division_b": b,
                "n_paired": len(shared),
                "mean_r_a": float(np.tanh(za.mean())), "mean_r_b": float(np.tanh(zb.mean())),
                "mean_z_difference": float(obs),
                "perm_p_value": float(p),
                "n_permutations": n_perm, "note": "",
            })
    return rows, per_div


# -------------------------------------------------------------- self-test --
def self_test():
    rng = np.random.default_rng(0)
    print("1. paired sign-flip returns null on zero-centred differences")
    d = rng.normal(0, 1, 40)
    _, p = paired_signflip_test(d - d.mean(), 5000, rng)
    assert p > 0.20, p
    print(f"   ok (p={p:.3f})")

    print("2. paired sign-flip detects a planted difference")
    d = rng.normal(0.8, 0.5, 40)
    obs, p = paired_signflip_test(d, 5000, rng)
    assert p < 0.01 and obs > 0.5, (obs, p)
    print(f"   ok (mean diff={obs:.3f}, p={p:.4f})")

    print("3. equal true effects with UNEQUAL n do not produce a paired difference")
    # the exact failure mode this script exists to catch
    n = 60
    za = rng.normal(0.35, 0.30, n)
    zb = rng.normal(0.35, 0.30, n)          # same true effect
    _, p = paired_signflip_test(za - zb, 5000, rng)
    assert p > 0.10, p
    print(f"   ok (p={p:.3f}) -- identical effects correctly show no difference")

    print("4. partial_corr matches a known value")
    x = np.array([1., 2, 3, 4, 5]); y = np.array([2., 1, 4, 3, 5]); z = np.array([1., 1, 2, 2, 3])
    assert np.isfinite(partial_corr(x, y, z))
    print("   ok")

    print("5. division_of maps real Glasser labels correctly")
    assert division_of("L_V2_ROI") == "Early_Visual"
    assert division_of("R_IPS1_ROI") == "Dorsal_Stream_Visual"
    assert division_of("L_MT_ROI") == "MT+_Complex"
    assert division_of("L_V1_ROI") == "Primary_Visual"
    print("   ok")
    print("\nSELF-TEST: ALL PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv")
    ap.add_argument("--controllability-csv")
    ap.add_argument("--out-csv")
    ap.add_argument("--min-nodes", type=int, default=3,
                    help="min nodes per division per subject; same value for every division "
                         "so retention differences cannot be introduced by the threshold itself")
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        self_test(); return
    for req in ("prf_backbone_csv", "controllability_csv", "out_csv"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_','-')} is required")

    rng = np.random.default_rng(a.seed)
    subj_ecc, node_size = load_prf_backbone(a.prf_backbone_csv, a.min_vertices_per_node)
    subj_avg, subj_modal = load_controllability(a.controllability_csv)

    ecc_nodes = set(node_size); ctrl_nodes = set().union(*(set(v) for v in subj_avg.values())) if subj_avg else set()
    if not (ecc_nodes & ctrl_nodes):
        sys.exit("ERROR: zero node-label overlap between pRF and controllability inputs. "
                 "Almost certainly mismatched atlases.")

    all_rows = []
    for name, metric in (("average_controllability_log", subj_avg), ("modal_controllability", subj_modal)):
        rows, per_div = run(subj_ecc, metric, node_size, name, a.min_nodes, a.n_permutations, rng)
        all_rows += rows
        print(f"\n=== {name} (min_nodes={a.min_nodes} in every division) ===")
        for d, m in per_div.items():
            if m:
                print(f"  {d:24s} n={len(m):3d}  mean r={np.tanh(fisher_z(list(m.values())).mean()):+.3f}")

    ps = [r["perm_p_value"] for r in all_rows if "perm_p_value" in r]
    qs = bh_fdr(ps) if ps else []
    k = 0
    for r in all_rows:
        if "perm_p_value" in r:
            r["q_value_fdr"] = float(qs[k]); k += 1
        else:
            r["q_value_fdr"] = ""

    print("\n=== pairwise division comparisons (paired, same subjects both sides) ===")
    for r in all_rows:
        if "perm_p_value" not in r:
            print(f"  {r['metric']:28s} {r['division_a']} vs {r['division_b']}: {r['note']}"); continue
        print(f"  {r['metric']:28s} {r['division_a']:22s} vs {r['division_b']:22s} "
              f"n={r['n_paired']:3d} r={r['mean_r_a']:+.3f}/{r['mean_r_b']:+.3f} "
              f"p={r['perm_p_value']:.4f} q={r['q_value_fdr']:.4f}")

    fields = ["metric", "division_a", "division_b", "n_paired", "mean_r_a", "mean_r_b",
              "mean_z_difference", "perm_p_value", "n_permutations", "q_value_fdr", "note"]
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWritten to {a.out_csv}")


if __name__ == "__main__":
    main()
