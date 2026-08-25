#!/usr/bin/env python3
"""
Figure 1, Panel B -- single-subject Julich-Brain hOc1-hOc5 parcellation.
Real rendering script, meant to run wherever register_julich_native.py's
outputs and the subject's native surface files actually live (the data
server -- confirmed NOT present in this workspace; see "Why this can't run
here" below).

INPUTS THIS SCRIPT NEEDS (per subject, one already-QC-passed subject id):
  1. Native midthickness surface, both hemispheres:
       T1w/Native/<subject>.L.midthickness.native.surf.gii
       T1w/Native/<subject>.R.midthickness.native.surf.gii
     (standard HCP structural-pipeline output)
  2. Native sulcal depth map, both hemispheres (for background shading):
       T1w/Native/<subject>.L.sulc.native.shape.gii
       T1w/Native/<subject>.R.sulc.native.shape.gii
  3. The native-space Julich hOc label files THIS PROJECT'S OWN
     register_julich_native.py produces:
       julich_hOc_MPM.L.native.label.gii
       julich_hOc_MPM.R.native.label.gii
     plus its companion LUT text files:
       julich_hOc_MPM_native_left_LUT.txt
       julich_hOc_MPM_native_right_LUT.txt
     (format: "label_index<TAB>region<TAB>hemi" per line, e.g. "3 hOc3d left")

WHY THIS CAN'T RUN IN THIS WORKSPACE (checked directly, 2026-08-18):
  - No .surf.gii, .shape.gii, or julich_hOc_MPM*.label.gii files exist
    anywhere under D:\\Frontiers Project (find confirmed zero matches).
  - register_julich_native.py itself requires FSL (applywarp) and Connectome
    Workbench (wb_command) plus each subject's HCP derivative files
    (native T1, MNI-to-native warp field) -- its own docstring says
    explicitly: "run this on the data server, not in a network-restricted
    sandbox." Nothing has changed about that since this project's Methods
    section was written.
  - nilearn (0.14.0) and siibra (1.0.1) ARE installed here and CAN do the
    actual plotting once the three input types above exist -- the renderer
    below is real, tested against nilearn's real plot_surf_roi signature,
    not a stub. The blocker is strictly the missing subject-level files.

IMPORTANT COLOR CAVEAT (found while writing this, 2026-08-18):
  register_julich_native.py's own build_mpm() step auto-generates each
  region's color via an internal _label_color() helper (an HSV-spaced
  rainbow, not the Okabe-Ito palette used everywhere else in this figure
  set) and embeds that into the label file's own color table. If you just
  render the label.gii with its own embedded colors, you will NOT get the
  established hOc1=orange/hOc2=sky-blue/etc. scheme, and the panel will
  look inconsistent with Figures 3, 6, and 7. This is almost certainly why
  the collaborator's earlier render didn't match: it used the label file's
  own baked-in colors instead of remapping.

  This script does NOT trust the embedded color table. It reads the LUT
  text file for label_index -> region name, then builds its own colormap
  from OKABE_ITO_HOC below, and applies that at plot time. Always do it
  this way for any hOc1-hOc5 render in this project.

Usage:
    python3 render_fig01_panelB.py \\
        --subject XXXXXX \\
        --native-surf-dir /path/to/derivatives/hcp_pipelines/<sub>/T1w/Native \\
        --julich-native-dir /path/to/derivatives/julich_native/<sub> \\
        --out /path/to/figures/fig01_panelB_representative_subject_hoc_parcellation.png

Never pass a real subject ID through to the output filename or any text
drawn on the figure -- the brain-render spec is explicit that this panel
must refer to the subject only as "a representative subject."
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import nibabel as nib
from scipy import stats
from nilearn import plotting

DPI = 600


def _face_mode(vertex_vals):
    """Custom avg_method for plot_surf_roi: nilearn 0.14 only ships
    mean/median/max/min as string options (verified against this exact
    install, 2026-08-18 -- 'mode' raises AttributeError since it tries
    getattr(np, 'mode') internally, which doesn't exist). Discrete region
    labels must use the modal vertex value per face, not an average, or
    face colors at region boundaries get interpolated into meaningless
    fractional label indices."""
    return float(stats.mode(vertex_vals, keepdims=False).mode)

# Established project-wide palette (matches Figures 1B, 2C, 3D and the
# Okabe-Ito convention used everywhere else) -- NOT the label file's own
# auto-generated colors, see caveat above.
OKABE_ITO_HOC = {
    "hOc1":  "#E69F00",  # orange
    "hOc2":  "#56B4E9",  # sky blue
    "hOc3v": "#009E73",  # bluish green
    "hOc3d": "#F0E442",  # yellow
    "hOc4v": "#D55E00",  # vermillion
    "hOc4d": "#CC79A7",  # reddish purple
    "hOc5":  "#0072B2",  # blue
}
BACKGROUND_GREY = "#D9D9D9"  # unlabeled cortex, left unfilled per spec


def load_lut(lut_path):
    """Reads register_julich_native.py's own LUT text file. Returns
    {label_index: region_name}, e.g. {1: 'hOc1', 2: 'hOc2', ...} -- read
    fresh every time rather than assuming sorted-alphabetical index order,
    since that's an implementation detail of build_mpm() we shouldn't
    silently depend on."""
    mapping = {}
    with open(lut_path) as f:
        next(f)  # header line: "# label_index\tregion\themi"
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            idx, region, _hemi = parts
            mapping[int(idx)] = region
    if not mapping:
        raise ValueError(f"No labels parsed from {lut_path} -- check the file wasn't truncated")
    return mapping


def build_colormap(lut_mapping):
    """label_index 0 = background/unlabeled (grey). Indices 1..N map to
    OKABE_ITO_HOC via the region name looked up in the LUT -- if a region
    name in the LUT isn't one of the 7 we know about, fail loudly rather
    than silently color it black."""
    max_idx = max(lut_mapping.keys())
    colors = [BACKGROUND_GREY] * (max_idx + 1)
    for idx, region in lut_mapping.items():
        if region not in OKABE_ITO_HOC:
            raise ValueError(f"Unrecognized region '{region}' at label index {idx} -- "
                              f"expected one of {list(OKABE_ITO_HOC)}")
        colors[idx] = OKABE_ITO_HOC[region]
    return ListedColormap(colors)


def render_hemi(ax, surf_mesh, label_gii, bg_map, cmap, view):
    labels = nib.load(str(label_gii)).darrays[0].data
    plotting.plot_surf_roi(
        surf_mesh=str(surf_mesh),
        roi_map=labels,
        bg_map=str(bg_map) if bg_map else None,
        hemi="left" if "L." in str(label_gii) or ".L." in str(label_gii) else "right",
        view=view,
        cmap=cmap,
        avg_method=_face_mode,   # categorical labels, never average/interpolate
        bg_on_data=True,
        colorbar=False,          # no baked-in text/legend, per project convention
        title=None,
        axes=ax,
        figure=ax.figure,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="used only to locate files, never written to output")
    ap.add_argument("--native-surf-dir", required=True, type=Path)
    ap.add_argument("--julich-native-dir", required=True, type=Path)
    ap.add_argument("--view", default="medial", choices=["anterior", "posterior", "medial", "lateral", "dorsal", "ventral"],
                     help="medial (default) keeps the calcarine sulcus -- where hOc1 sits -- in view, per the brain-render spec")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sub = args.subject
    surf_dir = args.native_surf_dir
    julich_dir = args.julich_native_dir

    required = {
        "L surface": surf_dir / f"{sub}.L.midthickness.native.surf.gii",
        "R surface": surf_dir / f"{sub}.R.midthickness.native.surf.gii",
        "L sulc":    surf_dir / f"{sub}.L.sulc.native.shape.gii",
        "R sulc":    surf_dir / f"{sub}.R.sulc.native.shape.gii",
        "L label":   julich_dir / "julich_hOc_MPM.L.native.label.gii",
        "R label":   julich_dir / "julich_hOc_MPM.R.native.label.gii",
        "L LUT":     julich_dir / "julich_hOc_MPM_native_left_LUT.txt",
        "R LUT":     julich_dir / "julich_hOc_MPM_native_right_LUT.txt",
    }
    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        print("Missing required input file(s), cannot render:")
        for name in missing:
            print(f"  - {name}: {required[name]}")
        sys.exit(1)

    lut_l = load_lut(required["L LUT"])
    lut_r = load_lut(required["R LUT"])
    cmap_l = build_colormap(lut_l)
    cmap_r = build_colormap(lut_r)

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(8, 5), dpi=DPI)
    render_hemi(axes[0], required["L surface"], required["L label"], required["L sulc"], cmap_l, args.view)
    render_hemi(axes[1], required["R surface"], required["R label"], required["R sulc"], cmap_r, args.view)
    for ax in axes:
        ax.set_axis_off()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {args.out}")
    print("Reminder: verify the output filename and any figure/caption text refers to "
          "\"a representative subject\" only -- never the real subject ID.")


if __name__ == "__main__":
    main()
