#!/usr/bin/env python3
"""
Targeted follow-up to check_surface_space.py: now that register_julich_native.py
has been repointed at T1w/Native (true native ACPC space) for its anatomical
surfaces, this checks WHERE the Julich labels actually landed on that surface,
not just whether the label table is populated.

hOc1-hOc5 are all early/mid visual areas clustered at the occipital pole. If
the registration is spatially correct, the centroid of the LABELED vertices
(keys 1-14) should be clearly offset from the centroid of ALL vertices on the
hemisphere, in one consistent direction (toward the back of the brain). A
labeled-vertex centroid that's barely offset from the whole-hemisphere
centroid, or offset in an inconsistent/implausible direction, would indicate
the labels are not landing in a coherent anatomical location -- i.e. a
remaining spatial problem that the label-table check (which only confirms the
table is non-empty, not where the labels sit) would not catch.

Run this against the NATIVE-mesh label file (before 32k resampling), since
that's the direct output of Step 3 (-volume-label-to-surface-mapping) and
therefore the most direct test of whether that step is spatially correct.

Usage:
    python3 check_label_location.py --subject XXXXXX \
        --julich-native-dir /path/to/project/BIDS/derivatives/julich_native \
        --t1w-native-surf-dir /path/to/project/BIDS/derivatives/hcp_pipelines/sub-XXXXXX/anat/T1w/Native \
        --hemi L
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def load_surf_coords(path):
    g = nib.load(str(path))
    for arr in g.darrays:
        if arr.intent == nib.nifti1.intent_codes.code["NIFTI_INTENT_POINTSET"]:
            return arr.data
    return g.darrays[0].data


def load_label_array(path):
    g = nib.load(str(path))
    for arr in g.darrays:
        if arr.intent == nib.nifti1.intent_codes.code["NIFTI_INTENT_LABEL"]:
            return arr.data
    return g.darrays[0].data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--julich-native-dir", required=True, type=Path)
    ap.add_argument("--t1w-native-surf-dir", required=True, type=Path)
    ap.add_argument("--hemi", default="L", choices=["L", "R"])
    args = ap.parse_args()
    sid = args.subject
    h = args.hemi

    label_candidates = list(args.julich_native_dir.glob(f"sub-{sid}/**/*julich_hOc_MPM*.{h}.native.label.gii"))
    if not label_candidates:
        label_candidates = list(args.julich_native_dir.glob(f"sub-{sid}/**/*julich_hOc_MPM*.{h}.*.label.gii"))
    if not label_candidates:
        print(f"ERROR: no native-mesh label.gii found for sub-{sid} hemi {h} under {args.julich_native_dir}")
        print("Paste a real `find` of that subject's julich_native output dir and this script needs adjusting.")
        return

    surf_candidates = list(args.t1w_native_surf_dir.glob(f"*.{h}.midthickness.native.surf.gii"))
    if not surf_candidates:
        print(f"ERROR: no {h} midthickness native surface found under {args.t1w_native_surf_dir}")
        return

    label_path = label_candidates[0]
    surf_path = surf_candidates[0]
    print(f"Label file:   {label_path}")
    print(f"Surface file: {surf_path}")
    print()

    coords = load_surf_coords(surf_path)
    labels = load_label_array(label_path)

    if coords.shape[0] != labels.shape[0]:
        print(f"ERROR: vertex count mismatch -- surface has {coords.shape[0]}, label array has {labels.shape[0]}. "
              f"These files are not a matching pair, comparison is meaningless. Investigate before trusting anything below.")
        return

    all_centroid = coords.mean(axis=0)
    labeled_mask = labels > 0
    n_labeled = labeled_mask.sum()

    print(f"Total vertices:   {coords.shape[0]}")
    print(f"Labeled vertices: {n_labeled}  ({100 * n_labeled / coords.shape[0]:.2f}% of hemisphere)")
    print()

    if n_labeled == 0:
        print("RESULT: zero labeled vertices on this native-mesh file -- that's a real problem, stop here.")
        return

    labeled_centroid = coords[labeled_mask].mean(axis=0)
    offset = labeled_centroid - all_centroid
    offset_mag = np.linalg.norm(offset)

    print(f"Whole-hemisphere centroid (all vertices):  {all_centroid}")
    print(f"Labeled-vertex centroid (hOc1-hOc5 only):   {labeled_centroid}")
    print(f"Offset vector (labeled - whole):            {offset}")
    print(f"Offset magnitude:                           {offset_mag:.2f} mm")
    print()
    print("How to read this: early/mid visual cortex (hOc1-hOc5) sits at the occipital pole, the back of the")
    print("brain -- anatomically it should NOT be near the whole-hemisphere centroid. Expect a clear, single-")
    print("direction offset in the tens-of-mm range along whichever axis is anterior-posterior in this file's")
    print("convention (commonly Y). A near-zero offset (a few mm) or an offset that looks scattered/inconsistent")
    print("across repeated runs would indicate the labels are not landing in a coherent anatomical location.")
    print()
    print("This is still diagnostic, not a formal verdict -- but a large, single-direction offset here, combined")
    print("with the correct label table and vertex count already confirmed, is good convergent evidence the fix")
    print("is spatially correct. A real visual check in wb_view (labels overlaid on the inflated surface, visually")
    print("confirming they cluster at the occipital pole) is still the most decisive remaining check before fully")
    print("trusting this for the manuscript, but doesn't need to block the mechanical cohort rebuild below.")


if __name__ == "__main__":
    main()
