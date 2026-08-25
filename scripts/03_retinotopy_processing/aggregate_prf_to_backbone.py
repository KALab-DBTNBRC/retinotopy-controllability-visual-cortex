#!/usr/bin/env python3
"""
Phase 7 prerequisite for Claim 2: aggregates per-subject pRF eccentricity
(from extract_prf_solutions.py's sub-<ID>_prf.dscalar.nii, R2>30-thresholded
per the project's already-settled cutoff) onto Schaefer-200/Glasser backbone
nodes, so it can be paired with that same node's controllability value.

Why this needs its own step: the pRF CIFTI is in the standard 91282-grayordinate
dense space (59412 cortical grayordinates + subcortical -- medial wall already
excluded, same convention as Glasser MMP1.0's 59412-row dlabel), while the
backbone label files (schaefer200.<hemi>.label.gii / glasser.<hemi>.label.gii,
already produced per-subject by tck2connectome.py's `wb_command -cifti-separate`
step) are on the full 32492-vertex 32k_fs_LR mesh per hemisphere, medial wall
included as key 0. These aren't automatically the same indexing -- a grayordinate
index in the pRF file is NOT the same number as a vertex index on the full mesh.

Method: nibabel's CIFTI BrainModelAxis.iter_structures() gives, for each
structure (CIFTI_STRUCTURE_CORTEX_LEFT / _RIGHT), a slice into the data array
AND the real mesh vertex index for each grayordinate in that slice (confirmed
via a synthetic-mesh test matching the real medial-wall-exclusion structure
before trusting this on real data). This lets eccentricity/R2 be placed back
onto a full 32492-length array per hemisphere (NaN where no grayordinate
exists, e.g. medial wall) -- the same mesh the backbone label files use, so a
direct per-vertex lookup against the backbone atlas is valid.

For each backbone node (any Schaefer-200 or Glasser parcel, not just visual
ones -- the "is_vis_network" column identifies Schaefer-200's 7-network "Vis"
label subset for Claim 2's own restriction, but every node's aggregate is
computed and written so nothing has to be re-run for exploratory checks
elsewhere): mean eccentricity across that node's R2>30 vertices only,
plus vertex counts so a node with only a handful of valid vertices is visible
as lower-confidence, not silently equivalent to one with hundreds.

Extended 2026-08-13 (Claim 2 Phase B3/C2 follow-up work) to also aggregate
two more per-vertex pRF quantities across the same R2>30 vertex mask:
  - mean_rfsize_deg: plain arithmetic mean of prf_rfsize (population receptive
    field size, degrees of visual angle) -- a literature-standard correlate
    of cortical magnification (RF size scales with eccentricity and inversely
    with magnification; Harvey & Dumoulin 2011, J Neurosci), added because a
    real check (2026-08-13) found the existing n_vertices_total size
    covariate does NOT reliably track a node's mean eccentricity in either
    atlas (Schaefer r=0.197 p=0.36; Glasser r=-0.069 p=0.63 -- both
    nonsignificant), meaning it is NOT actually functioning as a cortical-
    magnification control despite Claim 2's own wording invoking magnification
    independence. mean_rfsize_deg is intended as an additional/alternate
    covariate that IS literature-grounded for this specific purpose.
  - mean_polar_angle_deg + circular_concentration_r: prf_ang is CIRCULAR data
    (0/360 degrees wrap around, e.g. 359 deg and 1 deg are 2 degrees apart,
    not 358) -- a plain arithmetic mean would be badly wrong near the
    wraparound point. Computed via the standard circular-mean formula
    (mean angle = atan2(mean(sin(theta)), mean(cos(theta)))), normalized to
    [0, 360). circular_concentration_r (the mean resultant length, in [0,1])
    is also written: values near 1 mean the node's valid vertices have tightly
    clustered polar angles (the circular mean is a trustworthy summary);
    values near 0 mean angles are scattered around the full circle (the
    circular mean would be a misleading single-number summary of genuinely
    heterogeneous polar-angle coverage) -- downstream analyses should check
    this before trusting mean_polar_angle_deg, not use it blindly.
Both extensions verified against a synthetic fixture before trusting them on
real data (see test_aggregate_prf_circular_fixture.py): angles [350, 5, 10]
degrees (which straddle the 0/360 wraparound) circular-mean to ~5.0 degrees
with high concentration (r close to 1), NOT the ~121.7 degrees a naive
arithmetic mean would wrongly give.

Usage:
    python3 aggregate_prf_to_backbone.py \
        --subject 123456 \
        --prf-dir /path/to/project/BIDS/derivatives/prf \
        --connectome-dir /path/to/project/BIDS/derivatives/connectomes \
        --out-csv /path/to/project/BIDS/derivatives/prf_backbone/sub-123456_prf_backbone.csv

    # all subjects (discovered from rawdata), combined output:
    python3 aggregate_prf_to_backbone.py \
        --rawdata-dir /path/to/project/BIDS/rawdata \
        --prf-dir /path/to/project/BIDS/derivatives/prf \
        --connectome-dir /path/to/project/BIDS/derivatives/connectomes \
        --out-csv /path/to/project/BIDS/derivatives/prf_backbone/all_subjects_prf_backbone.csv
"""
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

ATLASES = ("schaefer200", "glasser")
R2_THRESHOLD = 30.0  # matches extract_prf_solutions.py's already-settled cutoff (todo.md decisions log)
N_MESH_VERTICES = 32492
STRUCT_TO_HEMI = {"CIFTI_STRUCTURE_CORTEX_LEFT": "L", "CIFTI_STRUCTURE_CORTEX_RIGHT": "R"}


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


def circular_mean_deg(angles_deg):
    """Standard circular mean for angles in degrees. Returns (mean_deg in
    [0,360), resultant_length in [0,1]). Verified against a synthetic
    wraparound fixture (2026-08-13) -- see module docstring."""
    theta = np.radians(angles_deg)
    sin_mean = np.mean(np.sin(theta))
    cos_mean = np.mean(np.cos(theta))
    mean_deg = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0
    resultant_r = float(np.hypot(sin_mean, cos_mean))
    return float(mean_deg), resultant_r


def load_prf_by_hemi(prf_path):
    """Returns {hemi: (ecc_array[32492], r2_array[32492], ang_array[32492],
    rfsize_array[32492])}, NaN where no grayordinate exists (e.g. medial
    wall) for that mesh vertex."""
    img = nib.load(str(prf_path))
    scalar_axis = img.header.get_axis(0)
    brain_axis = img.header.get_axis(1)
    data = img.get_fdata()  # (n_maps, n_grayordinates)

    names = list(scalar_axis.name)
    required = ["prf_ecc", "prf_R2", "prf_ang", "prf_rfsize"]
    missing = [m for m in required if m not in names]
    if missing:
        raise ValueError(f"{prf_path}: expected maps {required}, missing {missing}, found {names}")
    ecc_row = data[names.index("prf_ecc")]
    r2_row = data[names.index("prf_R2")]
    ang_row = data[names.index("prf_ang")]
    rfsize_row = data[names.index("prf_rfsize")]

    out = {}
    for struct_name, slc, bm_part in brain_axis.iter_structures():
        if struct_name not in STRUCT_TO_HEMI:
            continue  # subcortical structures -- not needed here
        hemi = STRUCT_TO_HEMI[struct_name]
        ecc_full = np.full(N_MESH_VERTICES, np.nan, dtype=np.float64)
        r2_full = np.full(N_MESH_VERTICES, np.nan, dtype=np.float64)
        ang_full = np.full(N_MESH_VERTICES, np.nan, dtype=np.float64)
        rfsize_full = np.full(N_MESH_VERTICES, np.nan, dtype=np.float64)
        verts = bm_part.vertex
        ecc_full[verts] = ecc_row[slc]
        r2_full[verts] = r2_row[slc]
        ang_full[verts] = ang_row[slc]
        rfsize_full[verts] = rfsize_row[slc]
        out[hemi] = (ecc_full, r2_full, ang_full, rfsize_full)
    return out


def aggregate_subject(subject, prf_dir, connectome_dir):
    prf_path = Path(prf_dir) / f"sub-{subject}_prf.dscalar.nii"
    if not prf_path.exists():
        raise FileNotFoundError(f"Missing pRF CIFTI: {prf_path}")
    prf_by_hemi = load_prf_by_hemi(prf_path)

    rows = []
    for atlas in ATLASES:
        for hemi in ("L", "R"):
            if hemi not in prf_by_hemi:
                raise ValueError(f"sub-{subject}: no {hemi} cortex structure found in {prf_path}")
            ecc, r2, ang, rfsize = prf_by_hemi[hemi]

            backbone_path = Path(connectome_dir) / f"sub-{subject}" / f"{atlas}.{hemi}.label.gii"
            if not backbone_path.exists():
                raise FileNotFoundError(f"Missing backbone label file: {backbone_path}")
            backbone_data, backbone_labels = load_gifti_label(backbone_path)

            if backbone_data.shape[0] != N_MESH_VERTICES:
                raise ValueError(
                    f"sub-{subject} {hemi} {atlas}: expected {N_MESH_VERTICES} vertices, "
                    f"got {backbone_data.shape[0]} -- not the same mesh, do not trust any lookup from this pair."
                )

            for node_key, node_label in sorted(backbone_labels.items()):
                if node_key == 0:
                    continue
                mask = backbone_data == node_key
                n_total = int(mask.sum())
                if n_total == 0:
                    continue  # this node has no vertices in this hemisphere (e.g. contralateral-only node)

                valid = mask & (r2 > R2_THRESHOLD) & np.isfinite(ecc)
                n_valid = int(valid.sum())
                mean_ecc = float(np.mean(ecc[valid])) if n_valid > 0 else None

                # rfsize: same simple arithmetic mean as eccentricity (not circular)
                valid_rfsize = valid & np.isfinite(rfsize)
                mean_rfsize = float(np.mean(rfsize[valid_rfsize])) if valid_rfsize.sum() > 0 else None

                # polar angle: circular mean, see circular_mean_deg() docstring
                valid_ang = valid & np.isfinite(ang)
                if valid_ang.sum() > 0:
                    mean_ang, resultant_r = circular_mean_deg(ang[valid_ang])
                else:
                    mean_ang, resultant_r = None, None

                rows.append({
                    "subject": subject,
                    "atlas": atlas,
                    "hemisphere": hemi,
                    "backbone_label": node_label,
                    "is_vis_network": "_Vis_" in node_label,
                    "n_vertices_total": n_total,
                    "n_vertices_valid_R2gt30": n_valid,
                    "pct_valid": n_valid / n_total,
                    "mean_eccentricity_deg": mean_ecc,
                    "mean_rfsize_deg": mean_rfsize,
                    "mean_polar_angle_deg": mean_ang,
                    "circular_concentration_r": resultant_r,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", help="single subject ID")
    ap.add_argument("--rawdata-dir", help="used to discover all subjects if --subject not given")
    ap.add_argument("--prf-dir", required=True)
    ap.add_argument("--connectome-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

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
    no_valid_data = []

    for subject in subjects:
        try:
            rows = aggregate_subject(subject, args.prf_dir, args.connectome_dir)
            all_rows.extend(rows)
            n_ok += 1
            for r in rows:
                if r["mean_eccentricity_deg"] is None and r["is_vis_network"]:
                    no_valid_data.append(r)
        except Exception as e:
            print(f"sub-{subject}: FAILED -- {type(e).__name__}: {e}")
            n_failed += 1

    fieldnames = ["subject", "atlas", "hemisphere", "backbone_label", "is_vis_network",
                  "n_vertices_total", "n_vertices_valid_R2gt30", "pct_valid", "mean_eccentricity_deg",
                  "mean_rfsize_deg", "mean_polar_angle_deg", "circular_concentration_r"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    if no_valid_data:
        print(f"\n{len(no_valid_data)} Vis-network (subject, atlas, hemisphere, node) combos have ZERO "
              f"vertices passing R2>{R2_THRESHOLD:g} -- no eccentricity value available for these:")
        for r in no_valid_data[:20]:
            print(f"    sub-{r['subject']} {r['atlas']} {r['hemisphere']} {r['backbone_label']}")
        if len(no_valid_data) > 20:
            print(f"    ... and {len(no_valid_data) - 20} more")

    print(f"\nDone. {n_ok} subjects ok, {n_failed} failed. {len(all_rows)} rows written to {out_csv}")


if __name__ == "__main__":
    main()
