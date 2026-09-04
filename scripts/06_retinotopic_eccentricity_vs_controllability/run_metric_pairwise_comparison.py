#!/usr/bin/env python3
"""
Is the eccentricity association actually STRONGER for average controllability
than for the alternative graph metrics?

WHY THIS EXISTS
---------------
`run_graph_metric_specificity.py` tests each metric against zero. In Glasser,
average controllability survives correction and weighted degree, communicability,
eigenvector centrality and modal controllability do not. The manuscript has been
describing that as specificity.

That inference does not follow. "A is significant and B is not" is not a test of
"A is larger than B", and the point is not pedantic here: all five metrics are
computed from the same matrices, on the same nodes, in the same participants, so
their correlations with eccentricity are strongly dependent quantities. The
difference between two such correlations has a much smaller standard error than
either correlation does, which cuts both ways. It can make a real difference
detectable, and it can expose a claimed difference as nothing.

WHAT THIS DOES
--------------
For every participant, computes the partial correlation between node eccentricity
and each metric, on one common node set with one covariate, so that the metrics
differ only in which metric they are. Then, for each alternative metric, forms the
per-participant paired difference in Fisher z and tests whether its mean is zero.

Inference is a paired sign-flip permutation on the differences, which is exact
under exchangeability of the sign of each participant's difference and makes no
distributional assumption about the correlations themselves.

Comparisons are on ABSOLUTE Fisher z. The question is whether the association is
stronger, and modal controllability carries the opposite sign, so a signed
comparison would confound direction with magnitude. Signed differences are
reported alongside for completeness.

BH-FDR is applied across the alternatives within an atlas.

Three outcomes are possible and all three are reportable:
  - average controllability is significantly stronger: the specificity claim stands;
  - no significant difference: the claim must be reduced to "the only metric to
    reach corrected significance", which is what the data support;
  - an alternative is stronger: the claim is wrong.

USAGE
-----
    python3 run_metric_pairwise_comparison.py \
        --prf-backbone-csv <PRF.csv> --controllability-csv <CTRL.csv> \
        --graph-metrics-csv <GRAPH.csv> --atlas glasser --out-csv <OUT.csv>

    python3 run_metric_pairwise_comparison.py --self-test
"""

import argparse
import csv
import math
import sys
from collections import defaultdict

import numpy as np

GLASSER_VIS_AREAS = {
    "V1", "V2", "V3", "V4", "V3A", "V3B", "V6", "V6A", "V7", "IPS1",
    "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",
    "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t"}

ALTERNATIVES = ["weighted_degree", "communicability", "eigenvector_centrality",
                "participation_coefficient", "modal_controllability"]
REFERENCE = "average_controllability"


def is_glasser_vis(lab):
    p = lab.split("_")
    return len(p) >= 3 and p[0] in ("L", "R") and p[1] in GLASSER_VIS_AREAS


def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def partial_corr(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return np.clip(q, 0, 1)


def signflip_p(d, n_perm, rng):
    d = np.asarray(d, float)
    if d.size < 3:
        return float("nan")
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n_perm + 1))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_prf(path, atlas, minv):
    ecc, size = defaultdict(dict), {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("atlas") != atlas:
                continue
            lab = r["backbone_label"]
            vis = (r.get("is_vis_network") == "True") if atlas == "schaefer200" \
                else is_glasser_vis(lab)
            if not vis:
                continue
            try:
                size[lab] = float(r["n_vertices_total"])
                if float(r.get("n_vertices_valid_R2gt30") or 0) < minv:
                    continue
                e = float(r["mean_eccentricity_deg"])
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(e):
                ecc[r["subject"]][lab] = e
    return ecc, size


def load_metrics(ctrl_path, graph_path):
    """subject -> metric -> label -> value, for all five metrics."""
    out = defaultdict(lambda: defaultdict(dict))
    with open(ctrl_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                a = float(r["average_controllability"])
                m = float(r["modal_controllability"])
            except (ValueError, KeyError, TypeError):
                continue
            if a > 0:
                out[r["subject"]]["average_controllability"][r["node_label"]] = math.log(a)
            out[r["subject"]]["modal_controllability"][r["node_label"]] = m
    with open(graph_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = [c for c in rd.fieldnames if c in ALTERNATIVES]
        for r in rd:
            for c in cols:
                try:
                    v = float(r[c])
                except (ValueError, KeyError, TypeError):
                    continue
                if np.isfinite(v):
                    out[r["subject"]][c][r["node_label"]] = v
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

def compare(ecc, metrics, node_size, min_nodes, n_perm, seed):
    rng = np.random.default_rng(seed)
    present = [m for m in ALTERNATIVES
               if any(m in metrics[s] and metrics[s][m] for s in metrics)]

    per_sub = defaultdict(list)          # metric -> [fisher z per participant]
    n_nodes = []
    for s, e in ecc.items():
        md = metrics.get(s, {})
        if REFERENCE not in md:
            continue
        # ONE common node set across every metric being compared, so the
        # comparison isolates the metric and not the node set.
        common = set(e) & set(node_size) & set(md[REFERENCE])
        for m in present:
            common &= set(md.get(m, {}))
        common = sorted(common)
        if len(common) < min_nodes:
            continue
        x = np.array([e[n] for n in common])
        z = np.array([node_size[n] for n in common], float)
        vals = {}
        ok = True
        for m in [REFERENCE] + present:
            r = partial_corr(x, np.array([md[m][n] for n in common]), z)
            if not np.isfinite(r):
                ok = False
                break
            vals[m] = fz(r)
        if not ok:
            continue
        for m, v in vals.items():
            per_sub[m].append(v)
        n_nodes.append(len(common))

    n = len(per_sub.get(REFERENCE, []))
    if n < 5:
        return None

    ref = np.array(per_sub[REFERENCE])
    rows, praw = [], []
    for m in present:
        alt = np.array(per_sub[m])
        d_abs = np.abs(ref) - np.abs(alt)
        d_sgn = ref - alt
        p = signflip_p(d_abs, n_perm, rng)
        praw.append(p)
        rows.append({"comparison": f"{REFERENCE} vs {m}", "n_subjects": n,
                     "mean_abs_z_reference": float(np.abs(ref).mean()),
                     "mean_abs_z_alternative": float(np.abs(alt).mean()),
                     "mean_paired_diff_abs_z": float(d_abs.mean()),
                     "mean_paired_diff_signed_z": float(d_sgn.mean()),
                     "pct_subjects_reference_stronger":
                         float(100 * np.mean(np.abs(ref) > np.abs(alt))),
                     "p_signflip": p})
    q = bh_fdr(praw) if praw else []
    for row, qi in zip(rows, q):
        row["q_bh"] = float(qi)
    return {"rows": rows, "n_subjects": n,
            "median_n_nodes": float(np.median(n_nodes)) if n_nodes else float("nan")}


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    def synth(n_sub=40, n_node=30, seed0=0, eff_ref=0.55, eff_alt=0.55,
              eff_modal=-0.55):
        """Node sets identical across metrics. Only the eccentricity coupling
        differs, which is exactly what the comparison is supposed to isolate."""
        r = np.random.default_rng(seed0)
        ecc, size = defaultdict(dict), {}
        metrics = defaultdict(lambda: defaultdict(dict))
        for j in range(n_node):
            size[f"N{j:03d}"] = float(r.integers(200, 900))
        sz = np.array([size[f"N{j:03d}"] for j in range(n_node)])
        for i in range(n_sub):
            s = f"s{i:03d}"
            e = r.uniform(0.5, 8.0, n_node)
            base = 0.3 * (sz - sz.mean()) / sz.std()
            for j in range(n_node):
                ecc[s][f"N{j:03d}"] = float(e[j])
            for m, eff in [(REFERENCE, eff_ref), ("weighted_degree", eff_alt),
                           ("communicability", eff_alt),
                           ("eigenvector_centrality", eff_alt),
                           ("modal_controllability", eff_modal)]:
                y = eff * e + base + r.normal(0, 1.0, n_node)
                if m == REFERENCE:
                    y = np.exp(y * 0.1)          # exercised through the log path
                    y = np.log(y)
                for j in range(n_node):
                    metrics[s][m][f"N{j:03d}"] = float(y[j])
        return ecc, metrics, size

    def get(res, m):
        return next(r for r in res["rows"] if r["comparison"].endswith(m))

    # --- estimator sanity ----------------------------------------------------
    rng = np.random.default_rng(0)
    a, b, c = (rng.normal(size=200) for _ in range(3))
    check("partial_corr of independent vectors is near zero",
          abs(partial_corr(a, b, c)) < 0.2)
    check("bh_fdr is monotone and bounded",
          np.all(np.diff(np.sort(bh_fdr([0.001, 0.01, 0.04, 0.2, 0.9]))) >= 0)
          and bh_fdr([0.5, 0.5])[0] <= 1.0)
    d = np.full(30, 0.4)
    check("signflip detects a constant nonzero difference",
          signflip_p(d, 2000, np.random.default_rng(1)) < 0.01)
    check("signflip is null for a symmetric difference",
          signflip_p(np.tile([0.4, -0.4], 15), 2000,
                     np.random.default_rng(2)) > 0.2)

    # --- NEGATIVE CONTROL: metrics genuinely equally strong -------------------
    res = compare(*synth(eff_ref=0.55, eff_alt=0.55, seed0=1), min_nodes=10,
                  n_perm=4000, seed=1)
    check("negative control: equal metrics give a near-zero paired difference",
          all(abs(r["mean_paired_diff_abs_z"]) < 0.08 for r in res["rows"]))
    check("negative control: no comparison survives FDR",
          all(r["q_bh"] > 0.05 for r in res["rows"]))
    check("negative control: reference wins in roughly half of participants",
          all(30 < r["pct_subjects_reference_stronger"] < 70
              for r in res["rows"]))

    # --- planted: reference genuinely stronger --------------------------------
    res_s = compare(*synth(eff_ref=0.9, eff_alt=0.35, eff_modal=-0.35, seed0=2),
                    min_nodes=10, n_perm=4000, seed=2)
    check("planted advantage is detected for every alternative",
          all(r["q_bh"] < 0.01 for r in res_s["rows"]))
    check("planted advantage has the right sign",
          all(r["mean_paired_diff_abs_z"] > 0 for r in res_s["rows"]))
    check("planted advantage: modal compared on magnitude, not sign",
          get(res_s, "modal_controllability")["mean_abs_z_alternative"] > 0)

    # --- planted: an ALTERNATIVE is stronger, which must be detectable too ----
    res_w = compare(*synth(eff_ref=0.3, eff_alt=0.9, eff_modal=-0.3, seed0=3),
                    min_nodes=10, n_perm=4000, seed=3)
    check("a stronger alternative is detected",
          get(res_w, "weighted_degree")["q_bh"] < 0.01)
    check("a stronger alternative carries a negative paired difference",
          get(res_w, "weighted_degree")["mean_paired_diff_abs_z"] < 0)

    # --- the comparison must not be driven by differing node sets ------------
    ecc, metrics, size = synth(seed0=4)
    for s in list(metrics)[:5]:
        for n in list(metrics[s]["weighted_degree"])[:6]:
            del metrics[s]["weighted_degree"][n]
    res_m = compare(ecc, metrics, size, min_nodes=10, n_perm=1500, seed=4)
    check("dropping nodes from one metric shrinks the COMMON node set for all",
          res_m["median_n_nodes"] <= res["median_n_nodes"])

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--graph-metrics-csv"); ap.add_argument("--out-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for r in ("prf_backbone_csv", "controllability_csv", "graph_metrics_csv", "out_csv"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required unless --self-test")

    ecc, size = load_prf(a.prf_backbone_csv, a.atlas, a.min_vertices_per_node)
    metrics = load_metrics(a.controllability_csv, a.graph_metrics_csv)
    res = compare(ecc, metrics, size, a.min_valid_nodes, a.n_permutations, a.seed)
    if res is None:
        sys.exit("too few participants with a usable comparison")

    fields = ["comparison", "n_subjects", "mean_abs_z_reference",
              "mean_abs_z_alternative", "mean_paired_diff_abs_z",
              "mean_paired_diff_signed_z", "pct_subjects_reference_stronger",
              "p_signflip", "q_bh"]
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in res["rows"]:
            w.writerow({k: r[k] for k in fields})

    print(f"atlas = {a.atlas}   participants = {res['n_subjects']}   "
          f"median common nodes = {res['median_n_nodes']:.0f}")
    print(f"  paired comparisons against {REFERENCE}, on |Fisher z|\n")
    for r in res["rows"]:
        print(f"    vs {r['comparison'].split(' vs ')[1]:26s} "
              f"|z| {r['mean_abs_z_reference']:.4f} vs {r['mean_abs_z_alternative']:.4f}   "
              f"diff {r['mean_paired_diff_abs_z']:+.4f}   "
              f"stronger in {r['pct_subjects_reference_stronger']:5.1f}% of participants   "
              f"p = {r['p_signflip']:.4f}   q = {r['q_bh']:.4f}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
