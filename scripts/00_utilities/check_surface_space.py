#!/usr/bin/env python3
"""
One-off diagnostic (not part of the pipeline): compares actual vertex
coordinates between a subject's T1w/Native/ surface (if it exists in the
original extracted HCP package) and the MNINonLinear/Native/ surface that
register_julich_native.py has been using, to determine directly -- not by
trusting directory-naming convention -- whether "MNINonLinear/Native/"
surfaces are really in native T1w/ACPC space or in MNI nonlinear space.

Why this matters: register_julich_native.py builds its MPM volume in true
native T1w/ACPC space (via applywarp --ref=<native T1w>), then maps it onto
"MNINonLinear/Native/*.native.surf.gii". If those surfaces are actually in
MNI space (native mesh DENSITY, not native VOLUME space -- these are
independent in real HCP output), that's a real spatial misalignment that
would not be caught by qc_pre_connectome.py (which only checks the label
table is populated, not whether labels land on the anatomically correct
vertices).

Usage:
    python3 check_surface_space.py --subject XXXXXX \
        --extracted-root /path/to/project/Extracted \
        --derivatives-root /path/to/project/BIDS/derivatives/hcp_pipelines
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def describe(label, path):
    g = nib.load(str(path))
    coords = None
    for arr in g.darrays:
        if arr.intent == nib.nifti1.intent_codes.code["NIFTI_INTENT_POINTSET"]:
            coords = arr.data
            break
    if coords is None:
        coords = g.darrays[0].data  # fall back to first array
    print(f"{label}  ({path})")
    print(f"    vertex 0:      {coords[0]}")
    print(f"    bounding box min: {coords.min(axis=0)}")
    print(f"    bounding box max: {coords.max(axis=0)}")
    print(f"    centroid:         {coords.mean(axis=0)}")
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--extracted-root", required=True, type=Path)
    ap.add_argument("--derivatives-root", required=True, type=Path)
    ap.add_argument("--hemi", default="L", choices=["L", "R"])
    args = ap.parse_args()
    sid = args.subject

    mni_candidates = list(
        (args.derivatives_root / f"sub-{sid}" / "anat" / "MNINonLinear" / "Native")
        .glob(f"*.{args.hemi}.white.native.surf.gii")
    )
    t1w_candidates = list(
        (args.extracted_root / f"sub-{sid}" / "StructuralRecommended" / sid / "T1w")
        .rglob(f"*.{args.hemi}.white.native.surf.gii")
    )

    if not mni_candidates:
        print("ERROR: no MNINonLinear/Native white surface found -- check the derivatives-root path")
        return
    if not t1w_candidates:
        print("ERROR: no T1w/**/white.native.surf.gii found under the extracted StructuralRecommended tree "
              "-- check the extracted-root path, or the T1w tree may use different naming (paste a `find` "
              "of the T1w directory and this script needs adjusting, not guessing)")
        return

    mni_coords = describe("MNINonLinear/Native (used by register_julich_native.py)", mni_candidates[0])
    print()
    t1w_coords = describe("T1w/Native (true native ACPC space, per standard HCP convention)", t1w_candidates[0])

    print()
    if mni_coords.shape != t1w_coords.shape:
        print(f"RESULT: vertex counts differ ({mni_coords.shape[0]} vs {t1w_coords.shape[0]}) -- "
              f"can't compare vertex-for-vertex, something more fundamental is mismatched. Investigate before proceeding.")
        return

    # Per-vertex displacement is the real signal: a genuine MNI-nonlinear warp
    # produces displacement that VARIES by brain region (some vertices move
    # much more than others depending on local anatomy-vs-template match) --
    # a uniform/near-zero spread instead indicates the same underlying space
    # with only minor independent-processing noise, not a real warp.
    per_vertex_dist = np.linalg.norm(mni_coords - t1w_coords, axis=1)
    print("Per-vertex displacement between the two surfaces (mm):")
    print(f"    mean:   {per_vertex_dist.mean():.2f}")
    print(f"    median: {np.median(per_vertex_dist):.2f}")
    print(f"    p95:    {np.percentile(per_vertex_dist, 95):.2f}")
    print(f"    max:    {per_vertex_dist.max():.2f}")
    print(f"    std:    {per_vertex_dist.std():.2f}  (spread -- a real nonlinear warp should show high spread "
          f"across regions, not a tight uniform value)")
    print()
    print("This is diagnostic, not a verdict -- read the numbers: a real MNI-nonlinear warp for an adult "
          "human brain typically produces mean per-vertex displacement in the several-mm range with a WIDE "
          "spread (some regions barely move, others shift 10-20+ mm, e.g. temporal poles/orbital cortex). "
          "A tight, small, low-spread distribution (sub-mm to ~1-2mm, low std) is more consistent with the "
          "same underlying space plus minor independent-processing noise, not a real warp.")


if __name__ == "__main__":
    main()
