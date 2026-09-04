#!/usr/bin/env python3
"""
Per-node effect maps: the node-level statistics behind the brain-surface
effect-map panels.

Three maps are produced per atlas: eccentricity against average
controllability, eccentricity against modal controllability, and
Agreeableness against average controllability.

Output filenames and log labels deliberately carry no figure numbers.
Manuscript figure and panel numbering changes between drafts; the content
of these tables does not, and filenames that named a panel went stale
twice before this note was written. Consult the manuscript for which panel
currently displays which map.

The surface renders themselves still need server-side access to the fsLR32k
surface geometry files, but the per-node statistics behind them do not, so
this runs against the already-available CSVs and the cohort master table.

Design, per the brain-render spec (deliberately simpler than the paper's
primary Claim 2 statistic, and captioned as such wherever used):
  - Primary Claim 2 statistic = per-SUBJECT partial correlation (controlling
    for node vertex count), one r per subject, pooled across nodes. That is
    the paper's actual inferential claim (Table with bootstrap CI).
  - This script = per-NODE simple Pearson correlation, across the 46
    subjects, one r per node. A different, lower-powered, purely
    descriptive statistic -- flips which axis is pooled. Not a partial
    correlation, not FDR-corrected across nodes (it isn't a hypothesis
    test suite the way Claim 1/3/4 are; it is a spatial map of one
    already-established relationship's local strength). This must be
    captioned as descriptive, not as 200 (or 52) new hypothesis tests.

Reuses run_claim2_analysis.py's load_prf_backbone/load_controllability and
run_personality_analysis.py's load_personality UNMODIFIED (imported, not
reimplemented), so every number here is guaranteed consistent with those
scripts' own already-verified loading logic (same Vis-network node
definitions, same log-transform convention, same source columns).

Usage (defaults point at the real files already used elsewhere this
project; substitute your own project root for /path/to/project):
    python3 run_pernode_analysis.py
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

DEFAULT_ROOT = "/path/to/project"

# import the two already-verified loader modules directly, no reimplementation
sys.path.insert(0, DEFAULT_ROOT)
from run_claim2_analysis import load_prf_backbone, load_controllability  # noqa: E402
from run_personality_analysis import load_personality, PERSONALITY_COLUMNS  # noqa: E402


def pernode_pearson(x_by_subj_node, y_by_subj_node, node_size, min_n=10):
    """x_by_subj_node, y_by_subj_node: dict[subject][node] -> float.
    Returns list of dicts: node_label, n, r, p, n_vertices."""
    nodes = sorted(node_size.keys())
    subjects_x = set(x_by_subj_node.keys())
    subjects_y = set(y_by_subj_node.keys())
    rows = []
    for node in nodes:
        xs, ys = [], []
        for s in subjects_x & subjects_y:
            if node in x_by_subj_node[s] and node in y_by_subj_node[s]:
                xs.append(x_by_subj_node[s][node])
                ys.append(y_by_subj_node[s][node])
        n = len(xs)
        if n < min_n or np.std(xs) == 0 or np.std(ys) == 0:
            rows.append({"node_label": node, "n": n, "r": np.nan, "p": np.nan,
                          "n_vertices": node_size.get(node, "")})
            continue
        r, p = stats.pearsonr(xs, ys)
        rows.append({"node_label": node, "n": n, "r": r, "p": p,
                      "n_vertices": node_size.get(node, "")})
    return rows


def write_csv(rows, path, extra_cols=None):
    fieldnames = ["node_label", "n", "r", "p", "n_vertices"] + (extra_cols or [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {path} ({len(rows)} nodes)")


def summarize(rows, label):
    valid = [r["r"] for r in rows if not (isinstance(r["r"], float) and math.isnan(r["r"]))]
    if not valid:
        print(f"{label}: no valid nodes")
        return
    print(f"{label}: n_nodes={len(valid)}  r range [{min(valid):.3f}, {max(valid):.3f}]  "
          f"mean={np.mean(valid):.3f}  median={np.median(valid):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--prf-backbone", default=None)
    ap.add_argument("--ctrl-schaefer", default=None)
    ap.add_argument("--ctrl-glasser", default=None)
    ap.add_argument("--master-table", default=None)
    ap.add_argument("--min-vertices-per-node", type=int, default=5,
                     help="passed through to load_prf_backbone, matches Claim 2's own default")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = args.root
    prf_path = args.prf_backbone or os.path.join(root, "all_subjects_prf_backbone.csv")
    ctrl_schaefer_path = args.ctrl_schaefer or os.path.join(root, "Received files", "all_subjects_controllability_schaefer200.csv")
    ctrl_glasser_path = args.ctrl_glasser or os.path.join(root, "Received files", "all_subjects_controllability_glasser.csv")
    master_table_path = args.master_table or os.path.join(root, "master-table.xlsx")
    out_dir = args.out_dir or os.path.join("/path/to/output")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Per-node eccentricity vs. controllability, both metrics
    # ---------------------------------------------------------------
    for atlas, ctrl_path in [("schaefer200", ctrl_schaefer_path), ("glasser", ctrl_glasser_path)]:
        subj_ecc, node_size = load_prf_backbone(prf_path, args.min_vertices_per_node, atlas)
        subj_avg_log, subj_modal = load_controllability(ctrl_path)

        rows_avg = pernode_pearson(subj_ecc, subj_avg_log, node_size)
        rows_modal = pernode_pearson(subj_ecc, subj_modal, node_size)

        write_csv(rows_avg, os.path.join(out_dir, f"pernode_effect_average_controllability_{atlas}.csv"))
        write_csv(rows_modal, os.path.join(out_dir, f"pernode_effect_modal_controllability_{atlas}.csv"))
        summarize(rows_avg, f"per-node eccentricity vs average controllability ({atlas})")
        summarize(rows_modal, f"per-node eccentricity vs modal controllability ({atlas})")

    # ---------------------------------------------------------------
    # Per-node Agreeableness vs. average controllability
    # (representative measure, see brain-images-rendering-spec.docx Section 7b)
    # ---------------------------------------------------------------
    personality = load_personality(master_table_path)  # dict[subject] -> {trait: value}

    for atlas, ctrl_path in [("schaefer200", ctrl_schaefer_path), ("glasser", ctrl_glasser_path)]:
        _, node_size = load_prf_backbone(prf_path, args.min_vertices_per_node, atlas)
        subj_avg_log, _ = load_controllability(ctrl_path)

        # personality is per-subject, not per-node -- broadcast the same
        # Agreeableness value across every node for that subject, so the
        # per-node correlation is driven purely by variation in
        # controllability across nodes vs. the (node-constant) trait score
        agreeableness_by_subj_node = defaultdict(dict)
        for s, vals in personality.items():
            if s not in subj_avg_log or "agreeableness" not in vals:
                continue
            for node in subj_avg_log[s]:
                agreeableness_by_subj_node[s][node] = vals["agreeableness"]

        rows = pernode_pearson(agreeableness_by_subj_node, subj_avg_log, node_size)
        write_csv(rows, os.path.join(out_dir, f"pernode_effect_agreeableness_{atlas}.csv"))
        summarize(rows, f"per-node Agreeableness vs average controllability ({atlas})")

    print("\nDone. All outputs are per-node DESCRIPTIVE Pearson r (not the paper's primary "
          "per-subject partial-correlation statistic) -- caption accordingly, per the brain-render spec.")


if __name__ == "__main__":
    main()
