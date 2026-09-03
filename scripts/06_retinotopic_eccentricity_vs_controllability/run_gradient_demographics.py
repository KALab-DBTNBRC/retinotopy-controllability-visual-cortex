#!/usr/bin/env python3
"""
Does the retinotopic gradient itself vary with age or sex?

WHY THIS EXISTS
---------------
The cohort carries a structural age-by-sex skew that selection cannot remove:
no male in the source pool is older than 34, and only one female is younger
than 26. The manuscript tests sex differences in MEAN controllability and
reports them as null. It never tests whether the quantity the paper is actually
about, the per-participant eccentricity-controllability gradient, varies with
age or sex.

That is a gap. The gradient is the inferential unit of the central finding, so
if it tracked age or sex the skew would become a live confound rather than a
noted limitation.

WHAT THIS DOES
--------------
Extracts each participant's gradient, the partial correlation between node
eccentricity and node controllability with the size covariate removed, Fisher-z
transformed. Then fits three models:

    gradient ~ age
    gradient ~ sex
    gradient ~ age + sex          (each coefficient adjusted for the other)

Inference is by permutation: the predictor is shuffled across participants
10,000 times, which preserves the gradient distribution exactly and destroys
only its association with the demographic variable. The joint model is reported
because age and sex are themselves correlated in this cohort, so a marginal
association with one could be carried by the other.

Also reports the age-sex correlation, so the reader can judge how far the two
coefficients can be separated at all.

USAGE
-----
    python3 run_gradient_demographics.py \
        --prf-backbone-csv <PRF.csv> \
        --controllability-csv <CTRL.csv> \
        --master-table <MASTER.xlsx> \
        --atlas glasser --out-csv <OUT.csv>

    python3 run_gradient_demographics.py --self-test
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np


def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def partial_corr(x, y, z):
    Z = np.column_stack([np.ones(len(z)), np.asarray(z, float)])
    try:
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1, 1))


def ols_beta(y, X):
    """Coefficients of y on [1, X]. Returns None if rank-deficient."""
    D = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    if np.linalg.matrix_rank(D) < D.shape[1]:
        return None
    try:
        return np.linalg.lstsq(D, y, rcond=None)[0][1:]
    except np.linalg.LinAlgError:
        return None


def perm_test(y, X, n_perm, rng):
    """Permutation test on each coefficient of y ~ X.

    Rows of the predictor matrix are permuted together, which preserves the
    correlation structure among predictors and destroys only their link to y.
    """
    obs = ols_beta(y, X)
    if obs is None:
        return None, None
    Xa = np.column_stack([np.asarray(c, float) for c in X])
    cnt = np.zeros(len(obs))
    for _ in range(n_perm):
        idx = rng.permutation(len(y))
        b = ols_beta(y, [Xa[idx, j] for j in range(Xa.shape[1])])
        if b is not None:
            cnt += (np.abs(b) >= np.abs(obs))
    return obs, (cnt + 1) / (n_perm + 1)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def load_prf(path, atlas, min_vertices):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("atlas") != atlas:
                continue
            try:
                if float(row.get("n_vertices_valid_R2gt30") or 0) < min_vertices:
                    continue
                ecc = float(row["mean_eccentricity_deg"])
                size = float(row.get("mean_rfsize_deg") or "nan")
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(ecc):
                out[row["subject"]][row["backbone_label"]] = (ecc, size)
    return out


def load_ctrl(path, metric):
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        lab = next((c for c in rd.fieldnames if "label" in c.lower()), None)
        for row in rd:
            try:
                v = float(row[metric])
            except (ValueError, KeyError, TypeError):
                continue
            if metric == "average_controllability":
                if v <= 0:
                    continue
                v = np.log(v)
            out[row["subject"]][row[lab]] = v
    return out


def load_demographics(path):
    """subject -> (age, sex) with sex coded 1 for male, 0 for female."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    hdr = [str(h) if h is not None else "" for h in next(rows)]

    def find(*names):
        for i, h in enumerate(hdr):
            hl = h.lower()
            if any(n in hl for n in names):
                return i
        return None

    i_s, i_a, i_x = find("subject"), find("age"), find("sex", "gender")
    if i_s is None or i_a is None or i_x is None:
        raise ValueError(f"could not find subject/age/sex columns in {hdr[:8]}")
    out = {}
    for r in rows:
        if r is None or r[i_s] is None:
            continue
        try:
            age = float(str(r[i_a]).split("-")[0].strip())
        except (ValueError, TypeError):
            continue
        sx = str(r[i_x]).strip().upper()
        if sx not in ("M", "F"):
            continue
        out[str(r[i_s]).strip()] = (age, 1.0 if sx == "M" else 0.0)
    return out


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------

def gradients(prf, ctrl, min_nodes):
    g = {}
    for s in sorted(set(prf) & set(ctrl)):
        nodes = sorted(set(prf[s]) & set(ctrl[s]))
        if len(nodes) < min_nodes:
            continue
        ecc = np.array([prf[s][n][0] for n in nodes])
        size = np.array([prf[s][n][1] for n in nodes])
        y = np.array([ctrl[s][n] for n in nodes])
        if not np.all(np.isfinite(size)):
            continue
        r = partial_corr(ecc, y, size)
        if np.isfinite(r):
            g[s] = fz(r)
    return g


def run(grad, demo, n_perm, seed):
    rng = np.random.default_rng(seed)
    subs = sorted(set(grad) & set(demo))
    if len(subs) < 8:
        return None
    y = np.array([grad[s] for s in subs])
    age = np.array([demo[s][0] for s in subs])
    sex = np.array([demo[s][1] for s in subs])

    rows = []
    for name, X, labels in [("age", [age], ["age"]),
                            ("sex", [sex], ["sex_male"]),
                            ("age+sex", [age, sex], ["age", "sex_male"])]:
        b, p = perm_test(y, X, n_perm, rng)
        if b is None:
            continue
        for lab, bi, pi in zip(labels, b, p):
            rows.append({"model": name, "term": lab, "n_subjects": len(subs),
                         "beta": float(bi), "p_permutation": float(pi)})
    return {"rows": rows, "n": len(subs),
            "age_sex_r": float(np.corrcoef(age, sex)[0, 1]),
            "mean_gradient_z": float(y.mean()),
            "age_range": (float(age.min()), float(age.max())),
            "n_male": int(sex.sum())}


# ----------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    rng = np.random.default_rng(4)

    def synth(n=46, age_effect=0.0, sex_effect=0.0, skew=True, seed0=0):
        r = np.random.default_rng(seed0)
        sex = np.array([1.0] * (n // 2) + [0.0] * (n - n // 2))
        # Reproduce the cohort's structural skew using its REAL parameters:
        # male 27.3 (SD 3.4), female 31.0 (SD 3.1). An earlier fixture used
        # overlapping uniform ranges, which gave an age-sex correlation of
        # only -0.09 and so did not exercise the confound at all.
        age = np.where(sex == 1, r.normal(27.3, 3.4, n), r.normal(31.0, 3.1, n)) if skew \
            else r.normal(29.2, 3.6, n)
        g = 0.15 + age_effect * (age - age.mean()) + sex_effect * sex + r.normal(0, 0.10, n)
        grad = {f"s{i:03d}": g[i] for i in range(n)}
        demo = {f"s{i:03d}": (age[i], sex[i]) for i in range(n)}
        return grad, demo

    def get(res, model, term):
        return next(r for r in res["rows"] if r["model"] == model and r["term"] == term)

    # --- NEGATIVE CONTROL: no true demographic effect ------------------------
    res = run(*synth(age_effect=0, sex_effect=0, seed0=1), n_perm=2000, seed=1)
    check("no true effect: age is null", get(res, "age", "age")["p_permutation"] > 0.05)
    check("no true effect: sex is null", get(res, "sex", "sex_male")["p_permutation"] > 0.05)
    check("no true effect: coefficients near zero",
          abs(get(res, "age", "age")["beta"]) < 0.01)

    # --- planted age effect --------------------------------------------------
    res_a = run(*synth(age_effect=0.03, seed0=2), n_perm=2000, seed=2)
    check("planted age effect is detected",
          get(res_a, "age", "age")["p_permutation"] < 0.01)
    check("planted age effect has the right sign",
          get(res_a, "age", "age")["beta"] > 0)

    # --- planted sex effect --------------------------------------------------
    res_s = run(*synth(sex_effect=0.25, seed0=3), n_perm=2000, seed=3)
    check("planted sex effect is detected",
          get(res_s, "sex", "sex_male")["p_permutation"] < 0.01)

    # --- the joint model must separate a real age effect from a spurious ----
    # sex effect induced by the age-sex skew
    res_j = run(*synth(age_effect=0.05, sex_effect=0.0, skew=True, seed0=4),
                n_perm=3000, seed=4)
    marg_sex = get(res_j, "sex", "sex_male")["p_permutation"]
    joint_sex = get(res_j, "age+sex", "sex_male")["p_permutation"]
    joint_age = get(res_j, "age+sex", "age")["p_permutation"]
    check("skew makes sex look significant on its own", marg_sex < 0.05)
    check("the joint model correctly keeps age", joint_age < 0.05)
    check("the joint model correctly drops the spurious sex effect",
          joint_sex > marg_sex)

    # --- the skew is real in the fixture ------------------------------------
    check("fixture reproduces a nonzero age-sex correlation",
          abs(res_j["age_sex_r"]) > 0.3)
    check("no-skew fixture has near-zero age-sex correlation",
          abs(run(*synth(skew=False, seed0=5), n_perm=200, seed=5)["age_sex_r"]) < 0.3)

    print("\nSELF-TEST: " + ("ALL PASSED" if ok else "FAILURES"))
    return 0 if ok else 1


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prf-backbone-csv"); ap.add_argument("--controllability-csv")
    ap.add_argument("--master-table")
    ap.add_argument("--atlas", default="glasser", choices=("schaefer200", "glasser"))
    ap.add_argument("--metric", default="average_controllability",
                    choices=("average_controllability", "modal_controllability"))
    ap.add_argument("--out-csv")
    ap.add_argument("--min-valid-nodes", type=int, default=15)
    ap.add_argument("--min-vertices-per-node", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    for r in ("prf_backbone_csv", "controllability_csv", "master_table", "out_csv"):
        if not getattr(a, r):
            ap.error(f"--{r.replace('_','-')} is required unless --self-test")

    grad = gradients(load_prf(a.prf_backbone_csv, a.atlas, a.min_vertices_per_node),
                     load_ctrl(a.controllability_csv, a.metric), a.min_valid_nodes)
    demo = load_demographics(a.master_table)
    res = run(grad, demo, a.n_permutations, a.seed)
    if res is None:
        sys.exit("too few participants with both a gradient and demographics")

    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "term", "n_subjects",
                                           "beta", "p_permutation"])
        w.writeheader()
        for r in res["rows"]:
            w.writerow(r)

    print(f"atlas = {a.atlas}   metric = {a.metric}")
    print(f"  participants {res['n']} ({res['n_male']} male), age "
          f"{res['age_range'][0]:.0f} to {res['age_range'][1]:.0f}")
    print(f"  mean gradient (Fisher z) {res['mean_gradient_z']:+.4f}")
    print(f"  age-sex correlation in this cohort r = {res['age_sex_r']:+.3f}\n")
    for r in res["rows"]:
        print(f"  {r['model']:8s} {r['term']:9s} beta = {r['beta']:+.5f}   "
              f"p = {r['p_permutation']:.4f}")
    print(f"\nwrote {a.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
