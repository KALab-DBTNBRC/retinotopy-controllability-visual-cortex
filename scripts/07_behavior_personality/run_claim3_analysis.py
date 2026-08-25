#!/usr/bin/env python3
"""
Claim 3 -- Sensory (visual-system) controllability as a behavioral proxy.

Tests whether individual differences in visual-system controllability
correlate with HCP-available impulsivity/delay-discounting measures
(Delay Discounting AUC at $200 and $40,000, Flanker Inhibitory Control
age-adjusted score), extending Cornblath et al. 2019's whole-brain/
fronto-parietal controllability-behavior link down to visual cortex
specifically (lit-review.docx, Claim 3).

Design is fundamentally different from Claims 1/2, and simpler, per the
already-settled decision ("simple per-subject scalar correlation for
Claim 3", todo.md 2026-08-11): this is a standard cross-subject correlation
with N=46 genuinely independent subjects as the sample -- NOT a small,
fixed spatial-unit design like Claim 1 (8 hOc units) or a per-subject
within-node design like Claim 2. There is no small-spatial-N chance-
alignment vulnerability here (that problem was specific to correlating
across a handful of spatial units within one subject/one fixed vector,
repeated across subjects) -- so no permutation test is used; a plain
Pearson correlation across the 46 (or fewer, after exclusions) subjects is
appropriate and is the PRIMARY test here, with Spearman reported alongside
as a standard nonparametric robustness check (not double-counted in FDR).

"Visual-system controllability" needs one scalar per subject, and this
project has two candidate node-set definitions in hand from Claims 1/2 --
user decision 2026-08-12: report BOTH, don't pick one:
  - "hoc": mean across the 8 Julich hOc1/hOc2/hOc3v/hOc3d x L/R region-level
    values (build_hoc_region_controllability.py's output) -- ties Claim 3 to
    the exact same receptor-covered regions as Claim 1, at the cost of a
    coarser per-subject summary (up to 8 units averaged, often fewer since
    the same per-subject region-coverage gaps from Claim 1 apply here too).
  - "vis": mean across all 29 Schaefer-200 Vis-network nodes (the full
    functional network used in Claim 2) -- broader, not limited by Julich
    receptor-data availability. Uses ALL 29 nodes, not Claim 2's 24-node
    pRF-coverage-restricted subset -- that restriction was about pRF signal
    quality, irrelevant here since Claim 3 doesn't touch pRF data at all.

average_controllability is again log-transformed before subject-level
averaging (mean of logs, i.e. a geometric mean) -- same rationale as Claims
1/2 (diagnose_connectome_hemisphere_asymmetry.py's finding that average
controllability's heavy-tailed distribution is concentrated in visual
cortex). modal_controllability is averaged raw. Real distributions of all
three behavioral measures were checked before building (2026-08-12): no
floor/ceiling clustering in either delay-discounting AUC measure (range
0.037-0.977, healthy spread), Flanker is a standard age-adjusted score
(range 79-121) -- no transform needed for any of the three.

Benjamini-Hochberg FDR applied across the resulting 4 controllability
summaries x 3 behavioral measures = 12 Pearson tests (per-claim FDR, per
the standing user decision, 2026-08-11).

Usage:
    python3 run_claim3_analysis.py \
        --master-table-xlsx /path/to/master-table.xlsx \
        --controllability-csv /path/to/all_subjects_controllability_schaefer200.csv \
        --hoc-region-controllability-csv /path/to/hoc_region_controllability/schaefer200.csv \
        --out-csv /path/to/claim3_results_schaefer200.csv

    # Glasser rerun (Claim 5, 2026-08-13) -- the 'hoc' arm is atlas-agnostic
    # and needs no flag; --atlas glasser only changes the 'vis' arm's node set:
    python3 run_claim3_analysis.py \
        --master-table-xlsx /path/to/master-table.xlsx \
        --controllability-csv /path/to/all_subjects_controllability_glasser.csv \
        --hoc-region-controllability-csv /path/to/hoc_region_controllability/glasser.csv \
        --atlas glasser \
        --out-csv /path/to/claim3_results_glasser.csv
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import openpyxl
from scipy import stats

# Glasser MMP1.0's "Vis-network" equivalent (2026-08-13 decision, see
# run_claim2_analysis.py's module docstring and todo.md decisions log for
# the full sourcing story -- Huang & Rolls et al. 2022's Table 1, which
# reproduces Glasser et al. 2016's own official area-to-division scheme).
# Union of all 5 visual-system divisions (Primary Visual + Early Visual +
# Dorsal Stream Visual + Ventral Stream Visual + MT+ Complex), 26 areas/hemisphere.
GLASSER_VIS_AREAS = {
    "V1",
    "V2", "V3", "V4",
    "V3A", "V3B", "V6", "V6A", "V7", "IPS1",
    "FFC", "PIT", "V8", "VMV1", "VMV2", "VMV3", "VVC",
    "FST", "LO1", "LO2", "LO3", "MST", "MT", "PH", "V3CD", "V4t",
}
GLASSER_LABEL_RE = re.compile(r"^[LR]_(.+)_ROI$")


def is_glasser_vis_node(node_label):
    """node_label format confirmed against real data (2026-08-13): 'L_V1_ROI'
    / 'R_V1_ROI' etc."""
    m = GLASSER_LABEL_RE.match(node_label)
    if not m:
        return False
    return m.group(1) in GLASSER_VIS_AREAS

BEHAVIORAL_COLUMNS = {
    "delay_discounting_auc_200": "Delay Discounting AUC ($200)",
    "delay_discounting_auc_40000": "Delay Discounting AUC ($40,000)",
    "flanker_inhibitory_control": "Flanker Inhibitory Control (age-adj)",
}


def load_behavioral(path):
    """Returns {subject: {measure_key: value}} from master-table.xlsx's
    Subjects sheet. Subject IDs are read as the sheet's own string/int
    representation and normalized to str for consistent joining against
    the controllability CSVs' subject IDs."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Subjects"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    missing_cols = [c for c in BEHAVIORAL_COLUMNS.values() if c not in idx]
    if missing_cols:
        raise ValueError(f"master-table.xlsx Subjects sheet missing expected column(s): {missing_cols}")
    subj_col = idx["Subject ID"]

    out = {}
    for r in rows[1:]:
        if r[subj_col] is None:
            continue
        subject = str(r[subj_col])
        vals = {}
        ok = True
        for key, colname in BEHAVIORAL_COLUMNS.items():
            v = r[idx[colname]]
            if v is None:
                ok = False
                break
            vals[key] = float(v)
        if ok:
            out[subject] = vals
    return out


def load_hoc_summary(path):
    """Returns {subject: (avg_log_mean, modal_mean, n_units)} -- unweighted
    mean across whichever of the 8 hOc region-level values that subject has
    (build_hoc_region_controllability.py already drops zero-coverage units,
    see that script's docstring)."""
    per_subj_avg_log = defaultdict(list)
    per_subj_modal = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "weighted_avg_controllability_log" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path}: no weighted_avg_controllability_log column -- this file needs to be "
                f"rebuilt with the current build_hoc_region_controllability.py (2026-08-12 log-space fix)."
            )
        for row in reader:
            log_val = row.get("weighted_avg_controllability_log", "")
            if log_val not in ("", None):
                per_subj_avg_log[row["subject"]].append(float(log_val))
            per_subj_modal[row["subject"]].append(float(row["weighted_modal_controllability"]))

    out = {}
    for s in set(per_subj_avg_log) | set(per_subj_modal):
        avg_vals = per_subj_avg_log.get(s, [])
        modal_vals = per_subj_modal.get(s, [])
        out[s] = (
            float(np.mean(avg_vals)) if avg_vals else None,
            float(np.mean(modal_vals)) if modal_vals else None,
            len(modal_vals),
        )
    return out


def load_vis_summary(path, atlas):
    """Returns {subject: (avg_log_mean, modal_mean, n_nodes)} -- unweighted
    mean of log(average_controllability) and raw modal_controllability
    across ALL Vis-network-equivalent nodes for the given atlas: Schaefer-200
    nodes whose label contains '_Vis_' (all 29, not Claim 2's pRF-coverage-
    restricted 24 -- see module docstring), or for Glasser, GLASSER_VIS_AREAS
    (all 52 nodes across both hemispheres, 2026-08-13 decision)."""
    per_subj_avg_log = defaultdict(list)
    per_subj_modal = defaultdict(list)
    n_nonpositive = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if atlas == "schaefer200":
                is_vis = "_Vis_" in row["node_label"]
            else:
                is_vis = is_glasser_vis_node(row["node_label"])
            if not is_vis:
                continue
            avg_c = float(row["average_controllability"])
            per_subj_modal[row["subject"]].append(float(row["modal_controllability"]))
            if avg_c > 0:
                per_subj_avg_log[row["subject"]].append(math.log(avg_c))
            else:
                n_nonpositive += 1
    if n_nonpositive:
        print(f"WARNING: {n_nonpositive} Vis-network rows had non-positive average_controllability "
              f"-- excluded from the log-space summary only")

    out = {}
    for s in set(per_subj_avg_log) | set(per_subj_modal):
        avg_vals = per_subj_avg_log.get(s, [])
        modal_vals = per_subj_modal.get(s, [])
        out[s] = (
            float(np.mean(avg_vals)) if avg_vals else None,
            float(np.mean(modal_vals)) if modal_vals else None,
            len(modal_vals),
        )
    return out


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns q-values in the same order as input."""
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


def run_one_pair(controllability_key, controllability_vals, behavioral_key, behavioral, subjects):
    x, y, used = [], [], []
    for s in subjects:
        cval = controllability_vals.get(s)
        bval = behavioral.get(s, {}).get(behavioral_key)
        if cval is None or bval is None:
            continue
        x.append(cval)
        y.append(bval)
        used.append(s)
    n = len(used)
    if n < 5:
        return {"n_subjects": n, "note": "too few subjects with both values, skipped"}
    x, y = np.array(x), np.array(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return {"n_subjects": n, "note": "zero variance in one variable, skipped"}
    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)
    return {
        "n_subjects": n,
        "pearson_r": float(r_pearson),
        "p_pearson": float(p_pearson),
        "spearman_rho": float(r_spearman),
        "p_spearman": float(p_spearman),
        "note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-table-xlsx", required=True)
    ap.add_argument("--controllability-csv", required=True,
                     help="all_subjects_controllability_schaefer200.csv (per-node, for the 'vis' summary)")
    ap.add_argument("--hoc-region-controllability-csv", required=True,
                     help="build_hoc_region_controllability.py's output (for the 'hoc' summary)")
    ap.add_argument("--atlas", default="schaefer200", choices=("schaefer200", "glasser"),
                     help="which atlas's Vis-network-equivalent node set to use for the 'vis' "
                          "summary arm (2026-08-13: Glasser's is GLASSER_VIS_AREAS, see module docstring). "
                          "The 'hoc' arm is unaffected -- it's atlas-agnostic by construction.")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    behavioral = load_behavioral(args.master_table_xlsx)
    hoc_summary = load_hoc_summary(args.hoc_region_controllability_csv)
    vis_summary = load_vis_summary(args.controllability_csv, args.atlas)

    subjects = sorted(behavioral.keys())
    print(f"{len(subjects)} subjects with complete behavioral data")
    print(f"{len(hoc_summary)} subjects with an hOc-region controllability summary")
    print(f"{len(vis_summary)} subjects with a Vis-network controllability summary")

    controllability_series = {
        "avg_controllability_log__hoc": {s: hoc_summary[s][0] for s in hoc_summary if hoc_summary[s][0] is not None},
        "modal_controllability__hoc": {s: hoc_summary[s][1] for s in hoc_summary if hoc_summary[s][1] is not None},
        "avg_controllability_log__vis": {s: vis_summary[s][0] for s in vis_summary if vis_summary[s][0] is not None},
        "modal_controllability__vis": {s: vis_summary[s][1] for s in vis_summary if vis_summary[s][1] is not None},
    }

    results = []
    for c_key, c_vals in controllability_series.items():
        for b_key in BEHAVIORAL_COLUMNS:
            stat = run_one_pair(c_key, c_vals, b_key, behavioral, subjects)
            row = {"controllability_summary": c_key, "behavioral_measure": b_key}
            row.update(stat)
            results.append(row)

    p_values = [r["p_pearson"] for r in results if "p_pearson" in r]
    q_values = bh_fdr(p_values) if p_values else []
    qi = 0
    for r in results:
        if "p_pearson" in r:
            r["q_value_fdr"] = float(q_values[qi])
            qi += 1
        else:
            r["q_value_fdr"] = None

    fieldnames = ["controllability_summary", "behavioral_measure", "n_subjects",
                  "pearson_r", "p_pearson", "q_value_fdr", "spearman_rho", "p_spearman", "note"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n=== Results (FDR q < 0.05) ===")
    sig = [r for r in results if r.get("q_value_fdr") is not None and r["q_value_fdr"] < 0.05]
    if sig:
        for r in sorted(sig, key=lambda x: x["q_value_fdr"]):
            print(f"  {r['controllability_summary']} vs {r['behavioral_measure']}: "
                  f"r={r['pearson_r']:.3f}, p={r['p_pearson']:.4g}, q={r['q_value_fdr']:.4g}, "
                  f"n={r['n_subjects']}")
    else:
        print("  (none)")

    print(f"\nFull results ({len(results)} rows) written to {out_csv}")


if __name__ == "__main__":
    main()
