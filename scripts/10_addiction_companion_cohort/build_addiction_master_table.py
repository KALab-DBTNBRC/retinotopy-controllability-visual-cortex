#!/usr/bin/env python3
"""
Build addiction-master-table.xlsx: the full-width restricted-data record
(all 801 columns from the freshly re-downloaded HCP_RETINOTOPY_FULLCOHORT.csv)
for every one of the 21 addiction-relevant candidate subjects identified in
addiction-cohort-selection.xlsx, mirroring master-table.xlsx's "Subjects"
sheet role for the main cohort but keeping every column rather than a
curated subset.
"""
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

SRC = "/path/to/project/Received files/HCP_RETINOTOPY_FULLCOHORT.csv"
CANDIDATES_XLSX = "/path/to/project/addiction-cohort-selection.xlsx"
OUT = "/path/to/output/addiction-master-table.xlsx"

full = pd.read_csv(SRC)
print(f"Full re-downloaded cohort: {full.shape[0]} subjects x {full.shape[1]} columns")

candidates = pd.read_excel(CANDIDATES_XLSX, sheet_name="Cohort")
candidate_ids = candidates["Subject ID"].astype(int).tolist()
print(f"Addiction-relevant candidates: {len(candidate_ids)}")

subset = full[full["Subject"].astype(int).isin(candidate_ids)].copy()
subset["Subject"] = subset["Subject"].astype(int)
subset = subset.sort_values("Subject").reset_index(drop=True)
print(f"Rows matched: {len(subset)} / {len(candidate_ids)}")
missing = set(candidate_ids) - set(subset["Subject"])
if missing:
    print(f"WARNING: {len(missing)} candidate(s) not found in the re-downloaded file: {missing}")

HEADER_FILL = PatternFill(start_color="7A1F2B", end_color="7A1F2B", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF", name="Arial", size=10)
BODY_FONT = Font(name="Arial", size=10)
TITLE_FONT = Font(bold=True, size=13, name="Arial")
KEYCOL_FILL = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

wb = openpyxl.Workbook()

# --- Subjects sheet: full 801-column raw record, all 21 candidates ---
ws = wb.active
ws.title = "Subjects"
ws.freeze_panes = "C2"  # keep Subject + Gender visible while scrolling right

columns = list(subset.columns)
ws.append(columns)
for c in range(1, len(columns) + 1):
    cell = ws.cell(row=1, column=c)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = Alignment(text_rotation=0, wrap_text=False)

def to_native(v):
    """Convert numpy/pandas scalar types to native Python types for openpyxl.

    Booleans are deliberately written out as the literal text "TRUE"/"FALSE"
    rather than a real boolean value: writing True/False (whether native
    Python bool or numpy.bool_) through openpyxl's ws.append() in this
    environment was observed to silently produce a literal formula cell
    ('=TRUE()' / data_type 'f') instead of a boolean cell for a subset of
    rows -- confirmed by direct inspection (cell.data_type == 'f') -- which
    in turn made LibreOffice's headless recalculation of this ~800-column
    sheet pathologically slow/hang (~1200 spurious formula cells). Writing
    plain text sidesteps the issue entirely regardless of root cause and is
    unambiguous for a raw data dump like this sheet.
    """
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return str(v)
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if bool(v) else "FALSE"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


for _, row in subset.iterrows():
    values = [to_native(v) for v in row.tolist()]
    ws.append(values)

# Sheet-wide default font (Arial) set once via the workbook's default named
# style rather than per-cell, for efficiency across 801 columns.
wb._named_styles["Normal"].font = BODY_FONT

ws.column_dimensions["A"].width = 12  # Subject
ws.column_dimensions[get_column_letter(columns.index("Gender") + 1)].width = 9
# NOTE: deliberately no auto_filter here -- across 801 columns it made
# LibreOffice's headless recalc pathologically slow/hang (isolated via direct
# testing: identical sheet without autofilter recalculated in <1s, with it
# timed out at 40s+). Use each column's header + Excel's own "Filter" button
# manually if needed once opened.

# --- Summary sheet ---
ws2 = wb.create_sheet("Summary")
ws2["A1"] = "Addiction Cohort Master Table -- Summary"
ws2["A1"].font = TITLE_FONT
ws2.append([])
last_row = len(subset) + 1
ws2.append(["Metric", "Value"])
for c in range(1, 3):
    ws2.cell(row=3, column=c).font = HEADER_FONT
    ws2.cell(row=3, column=c).fill = HEADER_FILL
# NOTE on formulas vs. computed values: COUNTIF/COUNTA formulas referencing
# the 801-column Subjects sheet were tried first (the project's normal
# convention -- see cohort-selection.xlsx, addiction-cohort-selection.xlsx,
# both of which use live formulas successfully). For this specific file,
# LibreOffice's headless recalc (used to verify formulas before shipping,
# per this project's standard workflow) reproducibly hung/timed out on this
# sheet's combination of size and formula count, isolated via direct A/B
# testing across a dozen variants (not a formula-correctness problem -- the
# same formulas DID evaluate correctly on the runs that completed). Given
# repeated non-viability of verifying live formulas here, these 6 counts are
# computed directly from the same Subjects data in pandas below and written
# as plain values instead, cross-checked to match every LibreOffice run that
# did complete. Excel itself will have no trouble recalculating live
# COUNTIF formulas over this data if you prefer them -- this is a
# sandbox-recalc-tooling limitation, not a property of the data or Excel.
n_male = int((subset["Gender"] == "M").sum())
n_female = int((subset["Gender"] == "F").sum())
n_alc_abuse = int((subset["SSAGA_Alc_D4_Ab_Dx"] == 5).sum())
n_alc_dep = int((subset["SSAGA_Alc_D4_Dp_Dx"] == 5).sum())
n_mj = int((subset["SSAGA_Mj_Ab_Dep"] == 1).sum())
n_thc = int(subset["THC"].sum())

summary_rows = [
    ("Total N (candidates matched)", len(subset)),
    ("N Male", n_male),
    ("N Female", n_female),
    ("N with Alcohol Abuse Dx (SSAGA_Alc_D4_Ab_Dx=5)", n_alc_abuse),
    ("N with Alcohol Dependence Dx (SSAGA_Alc_D4_Dp_Dx=5)", n_alc_dep),
    ("N with Marijuana Abuse/Dependence (SSAGA_Mj_Ab_Dep=1)", n_mj),
    ("N with THC-positive toxicology", n_thc),
    ("Total columns (all restricted-data fields retained)", len(columns)),
    ("Source file", "Received files/HCP_RETINOTOPY_FULLCOHORT.csv (re-downloaded 2026-08-14)"),
    ("Candidate list source", "addiction-cohort-selection.xlsx, Cohort sheet"),
    ("Counts computed via", "pandas, directly from the Subjects sheet's own source data (see build_addiction_master_table.py) -- not live Excel formulas; see Notes below"),
]
for label, val in summary_rows:
    ws2.append([label, val])
    ws2.cell(row=ws2.max_row, column=1).font = BODY_FONT
    ws2.cell(row=ws2.max_row, column=2).font = BODY_FONT
ws2.column_dimensions["A"].width = 55
ws2.column_dimensions["B"].width = 45

# --- Notes, folded into the bottom of the Summary sheet ---
# (A separate third "Notes" sheet reproducibly made LibreOffice's headless
# recalc hang in this environment -- isolated via direct A/B testing: the
# same workbook recalculates in <1s with a 3rd sheet removed, regardless of
# whether that sheet used wrap_text or not, and hangs at 25s+ with it
# present. Root cause not fully diagnosed -- something about a 3rd sheet
# combined with the Subjects sheet's 801 columns -- but folding the same
# content into Summary as plain rows is confirmed to resolve it.)
ws2.append([])
ws2.append(["Notes", None])
ws2.cell(row=ws2.max_row, column=1).font = Font(bold=True, name="Arial", size=10)
notes = [
    f"Contains the FULL restricted-data record (all 801 columns, unfiltered) for each of the "
    f"{len(subset)} addiction-relevant candidates in addiction-cohort-selection.xlsx -- nothing "
    f"subset or dropped.",
    "Source: Received files/HCP_RETINOTOPY_FULLCOHORT.csv, re-downloaded 2026-08-14. Verified "
    "cell-for-cell identical to the original 184-subject pool this project already had.",
    "21 candidates originally passed all selection criteria; one subject was subsequently "
    "excluded (data-availability, not eligibility) after Recommended-package download confirmed "
    "MEG-only data with no MRI release -- see addiction-cohort-selection.xlsx Methodology sheet "
    "(held privately; the subject ID is withheld from the public repository).",
    f"Still a CANDIDATE list, not a fully processed cohort: no connectomes/controllability/pRF "
    f"computed yet for these {len(subset)}. See addiction-cohort-server-runbook.md for the pipeline "
    f"commands.",
    "Real subject IDs are for internal/lab-server use only -- do not paste into the manuscript or "
    "any external-facing document (de-identified sub-01...sub-N codes are used externally).",
]
for n in notes:
    ws2.append([n])
    ws2.cell(row=ws2.max_row, column=1).font = BODY_FONT

wb.save(OUT)
print(f"Saved {OUT}")
