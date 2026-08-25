#!/usr/bin/env python3
"""
Download the two connectome backbone parcellations decided on 2026-08-07:
  - Schaefer-200 (7-network), fsLR32k CIFTI dlabel -- primary backbone
  - Glasser MMP1.0, fsLR32k CIFTI dlabel -- Claim 5 alternate-parcellation check

Both are surface-native CIFTI (32k_fs_LR), matching the HCP pipeline's own output
space -- no extra registration needed to bring them onto subject surfaces.

Run this on the data server (not in a network-restricted sandbox -- GitHub raw
content and BALSA are not reachable from the Cowork sandbox's outbound proxy).

Usage:
    python3 pull_parcellations.py --out-dir /path/to/Parcellations
"""
import argparse
import sys
import urllib.request
from pathlib import Path

# Schaefer-200, 7-network, fsLR32k CIFTI dlabel.
# Source: ThomasYeoLab/CBIG GitHub repo (official parcellation release).
SCHAEFER_200_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/HCP/fslr32k/cifti/"
    "Schaefer2018_200Parcels_7Networks_order.dlabel.nii"
)
SCHAEFER_200_FNAME = "Schaefer2018_200Parcels_7Networks_order.dlabel.nii"

# Glasser MMP1.0, fsLR32k CIFTI dlabel.
# Source: BALSA (Q1-Q6_RelatedValidation210...), the atlas's official host.
# NOTE: BALSA direct-file links sometimes require the BALSA scene downloader
# or a free account. If this URL 403s, fall back to the manual path below.
GLASSER_MMP_URL = "https://balsa.wustl.edu/file/3VLx"
GLASSER_MMP_FNAME = (
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_"
    "Group_Colors.32k_fs_LR.dlabel.nii"
)
GLASSER_MMP_MANUAL_FALLBACK = (
    "If the automated pull fails: go to https://balsa.wustl.edu/study/show/RVVG "
    "(scene ID WN56), download "
    "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors."
    "32k_fs_LR.dlabel.nii manually, and place it in the same output directory "
    "under the filename above."
)


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already present, skipping: {dest.name}")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        size = dest.stat().st_size
        if size == 0:
            dest.unlink(missing_ok=True)
            print(f"  FAILED (empty response): {dest.name}")
            return False
        print(f"  downloaded {dest.name} ({size:,} bytes)")
        return True
    except Exception as e:
        print(f"  FAILED: {dest.name} -- {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Schaefer-200 (7-network, fsLR32k) -- primary connectome backbone")
    ok1 = download(SCHAEFER_200_URL, out_dir / SCHAEFER_200_FNAME)

    print("\nGlasser MMP1.0 (fsLR32k) -- Claim 5 alternate-parcellation check")
    ok2 = download(GLASSER_MMP_URL, out_dir / GLASSER_MMP_FNAME)
    if not ok2:
        print(f"  {GLASSER_MMP_MANUAL_FALLBACK}")

    print("\nDone." if (ok1 and ok2) else "\nOne or more downloads failed -- see notes above.")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
