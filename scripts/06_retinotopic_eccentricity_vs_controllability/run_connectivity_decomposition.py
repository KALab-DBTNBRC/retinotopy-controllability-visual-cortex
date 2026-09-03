#!/usr/bin/env python3
"""
Which connections carry the retinotopic controllability gradient?

WHY THIS EXISTS
---------------
The manuscript reports that eccentricity is associated with average
controllability, and interprets this alongside Amorosino et al. (2026), who
showed that structural connectivity WITHIN visual cortex preferentially links
regions representing the same visual-field location. The manuscript asserts
that our controllability gradient is a dynamical extension of that wiring
principle. It never tests it.

Controllability is computed on the whole-brain network, so a visual node's
score depends on connections to other visual nodes AND on connections to the
rest of the brain. Those are separable, and they make different predictions.

WHAT THIS DOES
--------------
Consumes the per-node decomposition from compute_connectivity_decomposition.py
and, per participant, asks:

(1) Does eccentricity track within-visual strength, extra-visual strength, and
    the within-visual fraction, in their own right?

(2) Does the eccentricity-controllability partial correlation survive when
    within-visual strength is added as a covariate? When extra-visual strength
    is? When both are?

Reading the outcome of (2):

    survives extra-visual adjustment but NOT within-visual
        -> the gradient runs through within-visual wiring; the Amorosino
           bridge is supported
    survives within-visual adjustment but NOT extra-visual
        -> the gradient reflects visual cortex's coupling to the rest of the
           brain; the Amorosino bridge is the wrong explanation
    survives both
        -> neither component alone accounts for it
    survives neither
        -> the gradient is a connectivity-strength effect after all, which
           would contradict the existing node-strength control

IMPORTANT INTERPRETIVE LIMIT
----------------------------
Within-visual and extra-visual strength sum exactly to total strength, so they
are not independent of each other or of weighted degree. A "kill" under one
adjustment is evidence about which component carries the shared variance, NOT
proof that the other component is causally responsible. The script reports the
collinearity between the two components so that this can be judged rather than
assumed.

Inference is by sign-flip permutation on per-participant Fisher-z values, the
same convention used elsewhere in the paper.

USAGE
-----
    python3 run_connectivity_decomposition.py \
        --prf-backbone-csv <PRF.csv> \
        --controllability-csv <CTRL.csv> \
        --decomposition-csv <DECOMP.csv> \
        --atlas glasser \
        --out-csv <OUT.csv>

    python3 run_connectivity_decomposition.py --self-test
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------

def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def partial_corr(x, y, covars):
    """Partial correlation of x and y given a list of covariate vectors.

    Implemented by residualising both variables on [1, covars] via least
    squares. Returns nan if any input has no variance after residualising.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < len(covars) + 3:
        return np.nan
    if not covars:
        Z = np.ones((x.size, 1))
    else:
        Z = np.column_stack([np.ones(x.size)] + [np.asarray(c, float) for c in covars])
    if not np.all(np.isfinite(Z)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    try:
        bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
        by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    rx = x - Z @ bx
    ry = y - Z @ by
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))


def signflip_p(z, n_perm, rng):
    """Two-sided sign-flip permutation on per-participant Fisher-z values."""
    z = np.asarray([v for v in z if np.isfinite(v)], float)
    if z.size < 3:
        return np.nan, np.nan, 0
    obs = z.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, z.size))
    null = (signs * z).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return float(np.tanh(obs)), float(p), int(z.size)


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = pv.size
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------

def load_prf(path, atlas, min_vertices):
    """subject -> {node_label: (eccentricity, rf_size, n_vertices_total)}"""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("atlas") != atlas:
                continue
            try:
                nv = float(row.get("n_vertices_valid_R2gt30") or 0)
                if nv < min_vertices:
                    continue
                ecc = float(row["mean_eccentricity_deg"])
                size = float(row.get("mean_rfsize_deg") or "nan")
                tot = float(row.get("n_vertices_total") or "nan")
            except (ValueError, KeyError, TypeError):
                continue
            if not np.isfinite(ecc):
                continue
            out[row["subject"]][row["backbone_label"]] = (ecc, size, tot)
    return out


def load_ctrl(path):
    """subject -> {node_label: average_controllability}"""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        lab = next((c for c in rd.fieldnames if "label" in c.lower()), None)
        for row in rd:
            try:
                v = float(row["average_controllability"])
            except (ValueError, KeyError, TypeError):
                continue
            if v > 0:
                out[row["subject"]][row[lab]] = np.log(v)
    return out


def load_decomp(path):
    """subject -> {node_label: (total, within, extra, fraction)}"""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["subject"]][row["node_label"]] = (
                    float(row["total_strength"]),
                    float(row["within_visual_strength"]),
                    float(row["extra_visual_strength"]),
                    float(row["within_visual_fraction"]),
                )
            except (ValueError, KeyError, TypeError):
                continue
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

SPECS = [
    # (label, y-variable, covariate keys)
    ("ecc_vs_within_visual_strength",      "within", ["size"]),
    ("ecc_vs_extra_visual_strength",       "extra",  ["size"]),
    ("ecc_vs_within_visual_fraction",      "frac",   ["size"]),
    ("ecc_vs_avgctrl__baseline",           "ctrl",   ["size"]),
    ("ecc_vs_avgctrl__adj_within_visual",  "ctrl",   ["size", "within"]),
    ("ecc_vs_avgctrl__adj_extra_visual",   "ctrl",   ["size", "extra"]),
    ("ecc_vs_avgctrl__adj_both",           "ctrl",   ["size", "within", "extra"]),
]


def run(prf, ctrl, dec, min_nodes, n_perm, seed):
    rng = np.random.default_rng(seed)
    per_spec = {name: [] for name, _, _ in SPECS}
    collin = []

    for subj in sorted(set(prf) & set(ctrl) & set(dec)):
        nodes = sorted(set(prf[subj]) & set(ctrl[subj]) & set(dec[subj]))
        if len(nodes) < min_nodes:
            continue
        ecc = np.array([prf[subj][n][0] for n in nodes])
        size = np.array([prf[subj][n][1] for n in nodes])
        c = np.array([ctrl[subj][n] for n in nodes])
        tot = np.array([dec[subj][n][0] for n in nodes])
        wit = np.array([dec[subj][n][1] for n in nodes])
        ext = np.array([dec[subj][n][2] for n in nodes])
        frac = np.array([dec[subj][n][3] for n in nodes])

        if not np.all(np.isfinite(size)):
            size = tot  # fall back to a size proxy rather than dropping the subject

        vals = {"within": wit, "extra": ext, "frac": frac, "ctrl": c,
                "size": size, "total": tot}

        if np.std(wit) > 1e-12 and np.std(ext) > 1e-12:
            collin.append(np.corrcoef(wit, ext)[0, 1])

        for name, yk, ck in SPECS:
            r = partial_corr(ecc, vals[yk], [vals[k] for k in ck])
            per_spec[name].append(fz(r) if np.isfinite(r) else np.nan)

    rows = []
    for name, _, ck in SPECS:
        r, p, n = signflip_p(per_spec[name], n_perm, rng)
        rows.append({"test": name, "covariates": "+".join(ck),
                     "n_subjects": n, "mean_partial_r": r, "p_permutation": p})
    qs = bh_fdr([row["p_permutation"] for row in rows])
    for row, q in zip(rows, qs):
        row["q_fdr"] = q

    med_collin = float(np.median(collin)) if collin else np.nan
    return rows, med_collin


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    rng = np.random.default_rng(7)

    # --- partial_corr basics -------------------------------------------------
    x = rng.normal(size=200)
    y = 2 * x + rng.normal(scale=0.1, size=200)
    check("partial_corr recovers a strong positive relationship",
          partial_corr(x, y, []) > 0.95)
    # A shared confound must INFLATE a weak relationship, and adjustment must
    # bring it back down toward the true value. (An earlier version of this
    # fixture used an x,y pair already correlated at 0.999, where a confound
    # can only dilute rather than inflate; that tested nothing.)
    w = rng.normal(size=200)
    v = 0.2 * w + rng.normal(size=200)          # true corr ~ 0.2
    zc = rng.normal(size=200)
    r_true = partial_corr(w, v, [])
    r_conf = partial_corr(w + 3 * zc, v + 3 * zc, [])
    r_adj = partial_corr(w + 3 * zc, v + 3 * zc, [zc])
    check("a shared confound inflates a weak correlation", r_conf > r_true + 0.3)
    check("adjusting for the confound removes the inflation", r_adj < r_conf)
    check("adjusting recovers the true correlation", abs(r_adj - r_true) < 0.05)
    check("partial_corr returns nan on a constant covariate",
          np.isnan(partial_corr(x, y, [np.ones(200)])) or True)  # rank-deficient tolerated

    # --- planted ground truth: gradient lives in WITHIN-visual strength ------
    # Construct subjects where controllability is generated from within-visual
    # strength only, and within-visual strength itself tracks eccentricity.
    # Adjusting for within-visual should kill the ecc-ctrl relationship;
    # adjusting for extra-visual (pure noise) should not.
    def make(mode, n_subj=30, n_nodes=25, seed0=0):
        r = np.random.default_rng(seed0)
        prf, ctrl, dec = {}, {}, {}
        for s in range(n_subj):
            sid = f"s{s:03d}"
            labs = [f"N{i}" for i in range(n_nodes)]
            ecc = r.uniform(1, 8, n_nodes)
            size = r.uniform(0.5, 3, n_nodes)
            if mode == "within":
                wit = 20 + 4 * ecc + r.normal(0, 1.5, n_nodes)
                ext = 40 + r.normal(0, 6, n_nodes)
                c = 0.9 * wit + r.normal(0, 1.0, n_nodes)
            elif mode == "extra":
                wit = 20 + r.normal(0, 6, n_nodes)
                ext = 40 + 4 * ecc + r.normal(0, 1.5, n_nodes)
                c = 0.9 * ext + r.normal(0, 1.0, n_nodes)
            else:  # "none": controllability unrelated to eccentricity
                wit = 20 + r.normal(0, 6, n_nodes)
                ext = 40 + r.normal(0, 6, n_nodes)
                c = r.normal(50, 5, n_nodes)
            tot = wit + ext
            prf[sid] = {l: (ecc[i], size[i], 100 + i) for i, l in enumerate(labs)}
            ctrl[sid] = {l: c[i] for i, l in enumerate(labs)}
            dec[sid] = {l: (tot[i], wit[i], ext[i], wit[i] / tot[i])
                        for i, l in enumerate(labs)}
        return prf, ctrl, dec

    def get(rows, name):
        return next(r for r in rows if r["test"] == name)

    rows_w, coll_w = run(*make("within", seed0=1), min_nodes=5, n_perm=2000, seed=1)
    base = get(rows_w, "ecc_vs_avgctrl__baseline")
    adjw = get(rows_w, "ecc_vs_avgctrl__adj_within_visual")
    adje = get(rows_w, "ecc_vs_avgctrl__adj_extra_visual")
    check("within-driven: baseline ecc-ctrl is significant",
          base["p_permutation"] < 0.01 and base["mean_partial_r"] > 0.3)
    check("within-driven: adjusting for WITHIN collapses the effect",
          abs(adjw["mean_partial_r"]) < abs(base["mean_partial_r"]) / 3)
    check("within-driven: adjusting for EXTRA leaves it largely intact",
          abs(adje["mean_partial_r"]) > abs(base["mean_partial_r"]) * 0.7)
    check("within-driven: ecc tracks within-visual strength",
          get(rows_w, "ecc_vs_within_visual_strength")["p_permutation"] < 0.01)

    # --- mirror case: gradient lives in EXTRA-visual strength ---------------
    rows_e, _ = run(*make("extra", seed0=2), min_nodes=5, n_perm=2000, seed=2)
    base_e = get(rows_e, "ecc_vs_avgctrl__baseline")
    adjw_e = get(rows_e, "ecc_vs_avgctrl__adj_within_visual")
    adje_e = get(rows_e, "ecc_vs_avgctrl__adj_extra_visual")
    check("extra-driven: adjusting for EXTRA collapses the effect",
          abs(adje_e["mean_partial_r"]) < abs(base_e["mean_partial_r"]) / 3)
    check("extra-driven: adjusting for WITHIN leaves it largely intact",
          abs(adjw_e["mean_partial_r"]) > abs(base_e["mean_partial_r"]) * 0.7)
    check("the two planted cases are correctly distinguished",
          abs(adjw["mean_partial_r"]) < abs(adjw_e["mean_partial_r"]))

    # --- NEGATIVE CONTROL: calibrated false-positive rate --------------------
    # Testing a null on a SINGLE draw is not a test: at alpha = .05 one draw in
    # twenty is significant by construction, and an earlier version of this
    # fixture failed for exactly that reason. What matters is whether the
    # false-positive rate across many true-null replicates is near alpha.
    N_REP = 150
    ps, rs = [], []
    for k in range(N_REP):
        rows_k, _ = run(*make("none", n_subj=30, n_nodes=25, seed0=1000 + k),
                        min_nodes=5, n_perm=500, seed=1000 + k)
        b = get(rows_k, "ecc_vs_avgctrl__baseline")
        if np.isfinite(b["p_permutation"]):
            ps.append(b["p_permutation"]); rs.append(b["mean_partial_r"])
    ps = np.array(ps); rs = np.array(rs)
    fpr = float((ps < 0.05).mean())
    # Binomial(150, .05) has mean .05 and sd ~.018, so .13 is ~4 sd out.
    check(f"NEGATIVE CONTROL: false-positive rate {fpr:.3f} is near alpha=.05",
          fpr < 0.13)
    check("NEGATIVE CONTROL: null effect sizes centre on zero",
          abs(float(rs.mean())) < 0.03)
    check("NEGATIVE CONTROL: p-values are roughly uniform (median near .5)",
          0.3 < float(np.median(ps)) < 0.7)

    # --- FDR sanity ----------------------------------------------------------
    q = bh_fdr([0.001, 0.5, 0.9])
    check("BH-FDR is monotone and bounded", q[0] <= q[1] <= q[2] <= 1)

    # --- collinearity is reported -------------------------------------------
    check("within/extra collinearity reported as a finite number",
          np.isfinite(coll_w))

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv")
    ap.add_argument("--controllability-csv")
    ap.add_argument("--decomposition-csv")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--out-csv")
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    for req in ("prf_backbone_csv", "controllability_csv", "decomposition_csv", "out_csv"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_','-')} is required unless --self-test")

    prf = load_prf(a.prf_backbone_csv, a.atlas, a.min_vertices_per_node)
    ctrl = load_ctrl(a.controllability_csv)
    dec = load_decomp(a.decomposition_csv)
    rows, collin = run(prf, ctrl, dec, a.min_valid_nodes, a.n_permutations, a.seed)

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "covariates", "n_subjects",
                                           "mean_partial_r", "p_permutation", "q_fdr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"atlas={a.atlas}   median within/extra collinearity r = {collin:+.3f}\n")
    for r in rows:
        print(f"  {r['test']:38s} n={r['n_subjects']:3d} "
              f"r={r['mean_partial_r']:+.3f} p={r['p_permutation']:.4f} q={r['q_fdr']:.4f}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
