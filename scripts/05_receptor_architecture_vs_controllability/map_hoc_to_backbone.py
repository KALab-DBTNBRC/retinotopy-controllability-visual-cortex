#!/usr/bin/env python3
"""
Phase 7 prerequisite: maps Julich hOc regions onto whichever Schaefer-200 /
Glasser MMP1.0 backbone node(s) they spatially overlap, per subject.

Why this exists: Phase 5's connectomes (tck2connectome.py) were deliberately
built on Schaefer-200/Glasser nodes, not directly on Julich hOc1-hOc5 (see
todo.md decisions log -- avoids the parcellation-size-bias problem of using
uneven cytoarchitectonic regions as connectome nodes). But Claim 1/4 need to
pair each hOc region's (group-level, from the Julich receptor atlas) receptor
density with a (per-subject) controllability value -- and nothing produced so
far connects "hOc1 left" to "which Schaefer/Glasser node(s) is that, for this
subject." This script builds that connection directly from real per-vertex
label data, not assumed correspondence.

Method: both the per-subject Julich hOc label file
(julich_hOc_MPM.<hemi>.32k_fs_LR.label.gii, real filename confirmed via
`tree` 2026-08-11) and the backbone label files
(<atlas>.<hemi>.label.gii, already produced per-subject as an intermediate
by tck2connectome.py's `wb_command -cifti-separate` step) live on the same
32k_fs_LR mesh -- so the overlap is a direct per-vertex lookup, no
resampling needed. For each hOc region's vertices (in that subject's own
Julich labeling), tally which backbone node each vertex falls in, and record
the fraction of that hOc region's vertices covered by each backbone node
(weights). Where an hOc region spans more than one backbone node, all
overlapping nodes are recorded with their respective weights -- downstream
code (build_claim_datasets.py) is responsible for weighting each node's
controllability by these fractions.

Each subject's OWN Julich labeling and OWN backbone label files are used
(not a shared reference copy) -- the backbone atlas assignment via
-cifti-separate has no subject-specific step and should be identical across
subjects, but using each subject's own copy costs nothing extra (already on
disk) and avoids relying on an unverified cross-subject-identity assumption.

A vertex falling on backbone key 0 (medial wall / non-cortex, e.g. right at
the edge of the parcellation) is recorded explicitly as "UNASSIGNED" rather
than silently dropped, so a real coverage gap is visible in the output
rather than just making weights not sum to 1 with no explanation.

Usage:
    python3 map_hoc_to_backbone.py \
        --subject 123456 \
        --julich-native-dir /path/to/project/BIDS/derivatives/julich_native \
        --connectome-dir /path/to/project/BIDS/derivatives/connectomes \
        --out-csv /path/to/project/BIDS/derivatives/hoc_backbone_overlap/sub-123456_overlap.csv

    # all subjects (discovered from julich-native-dir), combined output:
    python3 map_hoc_to_backbone.py \
        --rawdata-dir /path/to/project/BIDS/rawdata \
        --julich-native-dir /path/to/project/BIDS/derivatives/julich_native \
        --connectome-dir /path/to/project/BIDS/derivatives/connectomes \
        --out-csv /path/to/project/BIDS/derivatives/hoc_backbone_overlap/all_subjects_overlap.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import nibabel as nib

ATLASES = ("schaefer200", "glasser")


def discover_subjects(rawdata_dir):
    return sorted(
        p.name.replace("sub-", "")
        for p in Path(rawdata_dir).iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )


def load_gifti_label(path):
    g = nib.load(str(path))
    label_map = g.labeltable.get_labels_as_dict()
    data = None
    for arr in g.darrays:
        if arr.intent == nib.nifti1.intent_codes.code["NIFTI_INTENT_LABEL"]:
            data = arr.data
            break
    if data is None:
        data = g.darrays[0].data
    return data, label_map


def overlap_for_subject_hemi(subject, hemi, julich_dir, connectome_dir):
    """Returns list of row dicts: subject, hemisphere, hoc_key, hoc_name,
    atlas, backbone_key, backbone_label, n_hoc_vertices, n_overlap, weight."""
    julich_path = julich_dir / f"sub-{subject}" / f"julich_hOc_MPM.{hemi}.32k_fs_LR.label.gii"
    if not julich_path.exists():
        raise FileNotFoundError(f"Missing Julich label file: {julich_path}")

    julich_data, julich_labels = load_gifti_label(julich_path)

    rows = []
    for atlas in ATLASES:
        backbone_path = connectome_dir / f"sub-{subject}" / f"{atlas}.{hemi}.label.gii"
        if not backbone_path.exists():
            raise FileNotFoundError(f"Missing backbone label file: {backbone_path}")
        backbone_data, backbone_labels = load_gifti_label(backbone_path)

        if backbone_data.shape != julich_data.shape:
            raise ValueError(
                f"sub-{subject} {hemi} {atlas}: vertex count mismatch -- "
                f"julich={julich_data.shape}, backbone={backbone_data.shape}. "
                f"These are not on the same mesh; do not trust any overlap computed from this pair."
            )

        for hoc_key, hoc_name in julich_labels.items():
            if hoc_key == 0:
                continue
            mask = julich_data == hoc_key
            n_hoc_vertices = int(mask.sum())
            if n_hoc_vertices == 0:
                continue  # this key belongs to the other hemisphere, not present here

            overlap_vals = backbone_data[mask]
            counts = Counter(int(v) for v in overlap_vals)
            for backbone_key, n_overlap in sorted(counts.items()):
                backbone_label = "UNASSIGNED" if backbone_key == 0 else backbone_labels.get(
                    backbone_key, f"UNKNOWN_KEY_{backbone_key}")
                rows.append({
                    "subject": subject,
                    "hemisphere": hemi,
                    "hoc_name": hoc_name,
                    "atlas": atlas,
                    "backbone_label": backbone_label,
                    "n_hoc_vertices": n_hoc_vertices,
                    "n_overlap": n_overlap,
                    "weight": n_overlap / n_hoc_vertices,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", help="single subject ID")
    ap.add_argument("--rawdata-dir", help="used to discover all subjects if --subject not given")
    ap.add_argument("--julich-native-dir", required=True)
    ap.add_argument("--connectome-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    julich_dir = Path(args.julich_native_dir)
    connectome_dir = Path(args.connectome_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subjects = [args.subject]
    elif args.rawdata_dir:
        subjects = discover_subjects(args.rawdata_dir)
        print(f"Discovered {len(subjects)} subjects from {args.rawdata_dir}")
    else:
        raise SystemExit("Must pass either --subject or --rawdata-dir")

    all_rows = []
    n_ok, n_failed = 0, 0
    unassigned_warnings = []

    for subject in subjects:
        try:
            for hemi in ("L", "R"):
                all_rows.extend(overlap_for_subject_hemi(subject, hemi, julich_dir, connectome_dir))
            n_ok += 1
        except Exception as e:
            print(f"sub-{subject}: FAILED -- {type(e).__name__}: {e}")
            n_failed += 1

    fieldnames = ["subject", "hemisphere", "hoc_name", "atlas", "backbone_label",
                  "n_hoc_vertices", "n_overlap", "weight"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Surface any nontrivial UNASSIGNED coverage, not just write it and move on.
    unassigned = [r for r in all_rows if r["backbone_label"] == "UNASSIGNED" and r["weight"] > 0.05]
    if unassigned:
        print(f"\n{len(unassigned)} (subject, hemi, hoc_name, atlas) combos have >5% of their hOc "
              f"vertices falling outside any real backbone node (medial wall / parcellation edge):")
        for r in unassigned[:20]:
            print(f"    sub-{r['subject']} {r['hemisphere']} {r['hoc_name']} {r['atlas']}: "
                  f"{r['weight']*100:.1f}% unassigned")
        if len(unassigned) > 20:
            print(f"    ... and {len(unassigned) - 20} more")

    print(f"\nDone. {n_ok} subjects ok, {n_failed} failed. {len(all_rows)} overlap rows written to {out_csv}")


if __name__ == "__main__":
    main()
