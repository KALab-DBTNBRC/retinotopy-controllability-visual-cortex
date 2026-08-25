#!/usr/bin/env python3
"""
Extracts an ordered, per-node label list from a backbone CIFTI dlabel
parcellation (Schaefer-200, Glasser MMP1.0), for use as
compute_controllability.py's --labels-file.

Why this needs to exist: tck2connectome.py's node indices come from the
distinct label values present in the rasterized native-space parcellation
volume, in ascending order (standard MRtrix behavior for a label-volume
input to tck2connectome). Those label values are the same keys embedded in
the CIFTI's label table (Schaefer-200: keys 1-200, 1-100 LH/101-200 RH;
Glasser MMP1.0: keys 1-360, 1-180 RH/181-360 LH -- see todo.md decisions
log). So node i in the connectome corresponds to the i-th smallest label
key, and the label NAME for that node is whatever the CIFTI's label table
says for that key.

Method: run `wb_command -cifti-label-export-table`, which writes the
standard Workbench label-list text format (name line, then "key R G B A"
line, repeated per label -- the same format already used elsewhere in this
project by register_julich_native.py's build_mpm()). Parse that, drop key 0
(background/unlabeled -- not a real node), sort by key ascending, and
VERIFY the remaining keys form an exact contiguous 1..N range with no gaps
and no duplicates before trusting the ordering -- if that check fails, the
"node i = key i" assumption doesn't safely hold and this script aborts
rather than silently writing a mis-ordered labels file (a mis-ordered
labels file would silently mislabel every controllability value without
any error anywhere downstream).

Usage:
    python3 extract_atlas_labels.py \
        --cifti-path /path/to/project/Atlas/Schaefer2018_200Parcels_7Networks_order.dlabel.nii \
        --out-path /path/to/project/Atlas/schaefer200_labels.txt \
        --expected-n 200
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_label_list_file(text):
    """Parses the Workbench label-list text format: name line, then
    'key R G B A' line, repeated. Returns list of (key:int, name:str)."""
    lines = [l for l in text.splitlines() if l.strip() != ""]
    if len(lines) % 2 != 0:
        raise ValueError(
            f"Expected an even number of non-blank lines (name+key pairs), got {len(lines)}. "
            f"This doesn't match the expected Workbench label-list format -- inspect the raw "
            f"output before trusting anything downstream."
        )
    entries = []
    for i in range(0, len(lines), 2):
        name = lines[i].strip()
        key_line = lines[i + 1].strip().split()
        if len(key_line) < 1:
            raise ValueError(f"Malformed key line for label '{name}': {lines[i + 1]!r}")
        try:
            key = int(key_line[0])
        except ValueError:
            raise ValueError(f"Expected an integer key as the first token, got: {key_line[0]!r}")
        entries.append((key, name))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifti-path", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--map", default="1", help="map number or name within the CIFTI (default: 1)")
    ap.add_argument("--expected-n", type=int, required=True,
                     help="expected node count (200 for Schaefer-200, 360 for Glasser MMP1.0) -- "
                          "hard-checked, not just informational")
    ap.add_argument("--wb-command-bin", default="wb_command")
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [args.wb_command_bin, "-cifti-label-export-table", args.cifti_path, args.map, tmp_path]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout)
        sys.exit(f"wb_command failed (exit {result.returncode})")

    raw_text = Path(tmp_path).read_text()
    entries = parse_label_list_file(raw_text)

    # Drop key 0 (background/unlabeled) -- not a real node.
    entries = [(k, n) for k, n in entries if k != 0]
    entries.sort(key=lambda kv: kv[0])

    keys = [k for k, _ in entries]
    n = len(keys)
    expected_keys = list(range(1, args.expected_n + 1))
    if keys != expected_keys:
        missing = sorted(set(expected_keys) - set(keys))
        extra = sorted(set(keys) - set(expected_keys))
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        detail = []
        if missing:
            detail.append(f"missing keys: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if extra:
            detail.append(f"unexpected keys: {extra[:20]}{'...' if len(extra) > 20 else ''}")
        if dupes:
            detail.append(f"duplicate keys: {dupes[:20]}{'...' if len(dupes) > 20 else ''}")
        sys.exit(
            f"ABORTING: label keys are not an exact contiguous 1..{args.expected_n} range "
            f"(got {n} keys). The 'node i = key i' assumption does not safely hold -- "
            f"{'; '.join(detail) if detail else 'shape mismatch only'}. "
            f"Not writing a labels file that could silently mislabel every downstream value."
        )

    with open(args.out_path, "w") as f:
        for key, name in entries:
            f.write(name + "\n")

    print(f"OK: wrote {n} labels to {args.out_path}, keys verified as exact contiguous "
          f"1..{args.expected_n} range, node order = ascending key order.")


if __name__ == "__main__":
    main()
