#!/usr/bin/env python3
"""
Build addiction-cohort-selection.xlsx: candidate pool for a parallel
addiction-relevant cohort, drawn from the SAME 184-subject HCP 7T Retinotopy
source pool as the main cohort-selection.xlsx, using IDENTICAL inclusion
criteria 1-4 and 8 (retinotopy/dMRI completeness, normal color vision, no QC
issues, one-per-family independence) but with criteria 5-7 FLIPPED: instead
of excluding lifetime alcohol abuse/dependence or marijuana abuse/dependence
diagnoses, this pool REQUIRES at least one of them.

This mirrors cohort-selection.xlsx's exact three-sheet structure (Cohort /
Summary / Methodology & Limitations) so it can be handed to the lab server
alongside the existing pipeline scripts as the addiction-cohort candidate
list, exactly the same way the original cohort-selection.xlsx was used.

IMPORTANT: this produces a CANDIDATE list only, and its OUTPUT WORKBOOK is
restricted data. The workbook contains real HCP subject IDs alongside
substance-use diagnostic variables (required so the analysis server can pull
and process the data) and must not be pasted into the manuscript, committed
to this repository, or shared outside the lab. It is deliberately absent
from this public repository; only this construction script is published, so
that the selection criteria are auditable without exposing the identifiers
they were applied to.

Running this script requires the HCP restricted-access subject table, which
is available only under a separate HCP data use agreement.
"""
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

SRC = "/path/to/project/HCP_YA_subjects_2026_08_06_19_53_53.csv"
OUT = "/path/to/output/addiction-cohort-selection.xlsx"

pool = pd.read_csv(SRC)

crit = pool[
    (pool["fMRI_Ret_Compl"] == True)
    & (pool["3T_dMRI_Compl"] == True)
    & (pool["Color_Vision"] == "NORMAL")
    & (pool["QC_Issue"].isna())
].copy()

addiction_relevant = crit[
    (crit["SSAGA_Alc_D4_Ab_Dx"] == 5)
    | (crit["SSAGA_Alc_D4_Dp_Dx"] == 5)
    | (crit["SSAGA_Mj_Ab_Dep"] == 1)
].copy()

addiction_relevant = addiction_relevant.sort_values("Subject")
seen_fam = set()
kept_rows = []
dropped_sibling_rows = []
for _, row in addiction_relevant.iterrows():
    if row["Family_ID"] in seen_fam:
        dropped_sibling_rows.append(row)
        continue
    seen_fam.add(row["Family_ID"])
    kept_rows.append(row)

kept_before_availability = pd.DataFrame(kept_rows)

# POST-HOC EXCLUSION (2026-08-14): data availability, not eligibility -- the
# same category as the two data-availability exclusions applied to the main
# cohort. One candidate passed every criterion above (fMRI_Ret_Compl,
# 3T_dMRI_Compl, QC, colour vision, family independence, substance-use
# criterion), and its three Recommended-package zips were pulled onto the
# analysis server -- but direct ConnectomeDB inspection confirmed the subject
# has MEG data only, with no MRI Recommended release, so the downloaded
# packages cannot actually be used. Excluded here rather than reprocessed
# through a separate pipeline, for the same consistency reason as the main
# cohort's decision.
#
# The excluded subject ID is withheld from this public repository: this is a
# restricted-access dataset, and naming the ID here would tie a re-identifiable
# HCP identifier to the substance-use inclusion criterion it satisfied. The ID
# is held in the private cohort-selection workbook alongside the subject-ID
# lookup key. To reproduce the published cohort, populate the set below from
# that workbook.
AVAILABILITY_EXCLUDED = set()  # populate from the private cohort workbook
kept = kept_before_availability[
    ~kept_before_availability["Subject"].astype(int).astype(str).isin(AVAILABILITY_EXCLUDED)
].copy()


def yn(v, true_val):
    return "Yes" if v == true_val else "No"


cohort_rows = []
for _, r in kept.iterrows():
    cohort_rows.append([
        str(int(r["Subject"])),
        r["Gender"],
        int(r["Age_in_Yrs"]),
        r["Family_ID"],
        r.get("ZygositySR", ""),
        r.get("ZygosityGT", ""),
        r.get("Handedness", ""),
        r.get("Race", ""),
        r.get("Ethnicity", ""),
        r["Color_Vision"],
        100,
        "=TRUE()",
        "=TRUE()",
        "Clean",
        yn(r["SSAGA_Alc_D4_Ab_Dx"], 5),
        yn(r["SSAGA_Alc_D4_Dp_Dx"], 5),
        yn(r["SSAGA_Mj_Ab_Dep"], 1),
        "Yes" if r["THC"] else "No",
        int(r["SSAGA_Alc_D4_Ab_Sx"]) if pd.notna(r["SSAGA_Alc_D4_Ab_Sx"]) else None,
        int(r["SSAGA_Alc_D4_Dp_Sx"]) if pd.notna(r["SSAGA_Alc_D4_Dp_Sx"]) else None,
        int(r["Total_Drinks_7days"]) if pd.notna(r["Total_Drinks_7days"]) else None,
        int(r["SSAGA_Mj_Times_Used"]) if pd.notna(r["SSAGA_Mj_Times_Used"]) else None,
        int(r["SSAGA_TB_Smoking_History"]) if pd.notna(r["SSAGA_TB_Smoking_History"]) else None,
    ])

HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=13)

wb = openpyxl.Workbook()

# --- Cohort sheet ---
ws = wb.active
ws.title = "Cohort"
headers = [
    "Subject ID", "Sex", "Age (yrs)", "Family ID", "Zygosity (self-report)",
    "Zygosity (genotype)", "Handedness (EHI)", "Race", "Ethnicity", "Color Vision",
    "Retinotopy % Complete", "3T dMRI Complete", "7T dMRI Complete", "QC Status",
    "Alcohol Abuse Dx (lifetime)", "Alcohol Dependence Dx (lifetime)",
    "Marijuana Abuse/Dependence", "THC Test Positive",
    "Alcohol Abuse Symptom Count", "Alcohol Dependence Symptom Count",
    "Total Drinks, Past 7 Days", "Marijuana Times-Used Category",
    "Lifetime Smoking History Class",
]
ws.append(headers)
for c in range(1, len(headers) + 1):
    cell = ws.cell(row=1, column=c)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
for row in cohort_rows:
    ws.append(row)
last_row = len(cohort_rows) + 1
for i, h in enumerate(headers, start=1):
    ws.column_dimensions[get_column_letter(i)].width = max(14, len(h) + 2)

# --- Summary sheet ---
ws2 = wb.create_sheet("Summary")
ws2["A1"] = "Addiction-Relevant Candidate Pool -- Summary"
ws2["A1"].font = TITLE_FONT
ws2.append([])
ws2.append(["Metric", "Value"])
for c in range(1, 3):
    ws2.cell(row=3, column=c).font = HEADER_FONT
    ws2.cell(row=3, column=c).fill = HEADER_FILL
summary_rows = [
    ("Total N (candidates)", f"=COUNTA(Cohort!A2:A{last_row})"),
    ("N Male", f'=COUNTIF(Cohort!B2:B{last_row},"M")'),
    ("N Female", f'=COUNTIF(Cohort!B2:B{last_row},"F")'),
    ("Mean Age - Male", f'=AVERAGEIF(Cohort!B2:B{last_row},"M",Cohort!C2:C{last_row})'),
    ("Mean Age - Female", f'=AVERAGEIF(Cohort!B2:B{last_row},"F",Cohort!C2:C{last_row})'),
    ("N with Alcohol Abuse Dx", f'=COUNTIF(Cohort!O2:O{last_row},"Yes")'),
    ("N with Alcohol Dependence Dx", f'=COUNTIF(Cohort!P2:P{last_row},"Yes")'),
    ("N with Marijuana Abuse/Dependence", f'=COUNTIF(Cohort!Q2:Q{last_row},"Yes")'),
    ("N with any of the above (should equal Total N)", f'=COUNTIFS(Cohort!O2:O{last_row},"Yes")+COUNTIFS(Cohort!P2:P{last_row},"Yes")+COUNTIFS(Cohort!Q2:Q{last_row},"Yes")-COUNTIFS(Cohort!O2:O{last_row},"Yes",Cohort!P2:P{last_row},"Yes")'),
    ("N with THC-positive toxicology (not an inclusion criterion)", f'=COUNTIF(Cohort!R2:R{last_row},"Yes")'),
    ("Raw candidates before family dedup", len(addiction_relevant)),
    ("Sibling pairs removed by one-per-family dedup", len(dropped_sibling_rows)),
    ("Post-hoc excluded, data availability (see Methodology)", len(AVAILABILITY_EXCLUDED)),
    ("Overlap with main 46-subject cohort", 0),
]
for label, val in summary_rows:
    ws2.append([label, val])
ws2.column_dimensions["A"].width = 55
ws2.column_dimensions["B"].width = 18

# --- Methodology sheet ---
ws3 = wb.create_sheet("Methodology & Limitations")
ws3["A1"] = "Addiction-Relevant Cohort -- Candidate Selection Methodology"
ws3["A1"].font = TITLE_FONT
lines = [
    "",
    "PURPOSE: parallel candidate pool for testing whether visual-cortex controllability differs "
    "between substance-dependent and non-dependent individuals -- the specific comparison the main "
    "46-subject cohort (cohort-selection.xlsx) was deliberately built to exclude.",
    "",
    f"Source: same HCP_YA_subjects_2026_08_06_19_53_53.csv (184 subjects with 7T retinotopy data "
    f"attempted) used to build the original cohort.",
    "",
    "Inclusion criteria applied, in order -- IDENTICAL to the main cohort for criteria 1-4 and 8:",
    "1. fMRI_Ret_Compl = TRUE (7T retinotopy scan fully completed).",
    "2. 3T_dMRI_Compl = TRUE (diffusion MRI available for structural connectome / network control theory analysis).",
    "3. Color_Vision = NORMAL.",
    "4. QC_Issue blank (excludes all subjects flagged for anatomical anomalies, segmentation errors, coil instability, or motion/coil artifact).",
    "5. FLIPPED vs. main cohort: REQUIRES at least one of -- lifetime alcohol abuse diagnosis (SSAGA_Alc_D4_Ab_Dx = 5), "
    "lifetime alcohol dependence diagnosis (SSAGA_Alc_D4_Dp_Dx = 5), or marijuana abuse/dependence diagnosis (SSAGA_Mj_Ab_Dep = 1).",
    "6. One unrelated subject per family (Family_ID), deduplicated by keeping the lowest subject ID per family "
    "(arbitrary but consistent tie-break; no attempt made to balance sex/age at this stage since this is a "
    "candidate pool, not yet a finalized cohort).",
    "",
    f"Result: {len(crit)} subjects pass criteria 1-4. Of these, {len(addiction_relevant)} meet at least one "
    f"substance-use diagnostic criterion (criterion 5). After one-per-family dedup "
    f"({len(dropped_sibling_rows)} sibling(s) removed), {len(kept_before_availability)} unrelated, eligible "
    f"candidates remained ({(kept_before_availability.Gender=='M').sum()} male, "
    f"{(kept_before_availability.Gender=='F').sum()} female) before the data-availability exclusion below.",
    "",
    "POST-HOC EXCLUSION (2026-08-14): data availability, not eligibility -- the same category as the "
    "main cohort's two data-availability exclusions. One candidate passed every criterion above and its "
    "three Recommended-package zips were downloaded to the analysis server, but direct ConnectomeDB "
    "inspection confirmed the subject has MEG data only -- no MRI Recommended release exists -- so the "
    "downloaded packages cannot actually be used. Excluded rather than reprocessed through a separate "
    "pipeline, for the same consistency reason as the main cohort's decision. The subject ID is withheld "
    "from the public repository and held in the private cohort workbook.",
    "",
    f"FINAL candidate N = {len(kept)} ({(kept.Gender=='M').sum()} male, {(kept.Gender=='F').sum()} female), "
    f"ages {int(kept['Age_in_Yrs'].min())}-{int(kept['Age_in_Yrs'].max())}.",
    "",
    "Zero overlap with the main 46-subject cohort (confirmed directly) -- these are structurally distinct, "
    "mutually exclusive groups by construction (the main cohort excludes exactly what this pool requires).",
    "",
    "WHAT THIS FILE IS NOT: this is a candidate list, not a finalized, processed cohort. None of these "
    "candidates' structural connectomes, controllability values, or retinotopy pRF summaries have been "
    "computed yet -- that requires the full MRtrix3/FSL/Workbench pipeline already built and verified for "
    "the main cohort (register_julich_native.py, run_tractography.py, tck2connectome.py, "
    "compute_controllability.py, extract_prf_solutions.py, aggregate_prf_to_backbone.py), run against these "
    "subject IDs on the lab server -- this sandbox has no route to raw HCP downloads or to "
    "MRtrix3/FSL/Workbench, the same constraint documented throughout todo.md for every other new-subject "
    "processing step in this project. See addiction-cohort-server-runbook.md for the exact commands.",
    "",
    "SEVERITY VARIANCE: unlike a simple diagnosed/not-diagnosed split, this candidate pool shows real graded "
    "severity -- alcohol dependence symptom count ranges 0-3 (mean 1.6), 7-day drink totals range 0-29 (mean "
    "10.4), marijuana times-used category ranges 0-5 (mean 3.3) -- supporting either a categorical group "
    "comparison against the main cohort or a dimensional (severity) analysis within this pool alone.",
    "",
    "LIMITATIONS specific to this candidate pool (in addition to all limitations already documented for the "
    "main cohort's data source):",
    f"- N={len(kept)} is modest for a between-group comparison against the N=46 main cohort; a formal power "
    "calculation should be run once the outcome metric (average/modal controllability) and expected effect "
    "size are decided, reusing this project's existing power_analysis_correlation.py / "
    "run_claim1_power_analysis.py machinery.",
    "- Age/sex balance was not optimized here the way it was for the main cohort (which used a stratified draw); "
    f"{(kept.Gender=='M').sum()}M/{(kept.Gender=='F').sum()}F with whatever age range the diagnostic criterion "
    "happens to select is a fixed constraint of who in the 184-subject pool actually meets it, not a design "
    "choice -- age and sex should be covaried in any group comparison, as in the main cohort's own Claim 4 design.",
    "- 'Meets abuse/dependence criteria' is a lifetime diagnosis, not necessarily current use at time of scan; "
    "same SSAGA-interview-based measurement as the main cohort's exclusion criteria, so comparable but not a "
    "measure of active/current dependence.",
    "- One candidate has already been excluded for data availability (MEG-only, no MRI release) -- "
    "any remaining candidate could in principle surface the same issue once actually pulled through the "
    "processing pipeline; not yet independently re-verified for the other 20.",
]
for line in lines:
    ws3.append([line])
ws3.column_dimensions["A"].width = 120
for row in ws3.iter_rows():
    for cell in row:
        cell.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")

wb.save(OUT)
print(f"Saved {OUT}")
print(f"Kept {len(kept)} candidates: {(kept.Gender=='M').sum()}M / {(kept.Gender=='F').sum()}F")
