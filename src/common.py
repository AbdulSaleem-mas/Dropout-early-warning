"""Shared configuration for the staged dropout early-warning study.

Dataset: "Predict Students' Dropout and Academic Success" (Realinho et al., 2022),
Kaggle release "Higher Education Predictors of Student Retention" (dataset.csv,
4,424 rows x 35 columns).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

SEED = 42
CLASSES = ["Dropout", "Enrolled", "Graduate"]

# ---- colour system (validated categorical slots 1-3; fixed per entity) ----
CLASS_COLORS = {"Graduate": "#2a78d6", "Dropout": "#eb6834", "Enrolled": "#1baf7a"}
STAGE_COLORS = {"S0": "#86b6ef", "S1": "#2a78d6", "S2": "#104281"}  # sequential blue
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
HIGHLIGHT, BASE = "#2a78d6", "#b7d3f6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})

COL1, COL2 = 3.5, 7.16  # IEEE single / double column widths (inches)

# ---- variable dictionary ----
RENAME = {
    "Marital status": "marital_status", "Application mode": "application_mode",
    "Application order": "application_order", "Course": "course",
    "Daytime/evening attendance": "daytime_attendance",
    "Previous qualification": "previous_qualification", "Nacionality": "nationality",
    "Mother's qualification": "mother_qualification",
    "Father's qualification": "father_qualification",
    "Mother's occupation": "mother_occupation", "Father's occupation": "father_occupation",
    "Displaced": "displaced", "Educational special needs": "special_needs",
    "Debtor": "debtor", "Tuition fees up to date": "tuition_up_to_date",
    "Gender": "gender_male", "Scholarship holder": "scholarship",
    "Age at enrollment": "age_at_enrollment", "International": "international",
    "Curricular units 1st sem (credited)": "s1_credited",
    "Curricular units 1st sem (enrolled)": "s1_enrolled",
    "Curricular units 1st sem (evaluations)": "s1_evaluations",
    "Curricular units 1st sem (approved)": "s1_approved",
    "Curricular units 1st sem (grade)": "s1_grade",
    "Curricular units 1st sem (without evaluations)": "s1_no_eval",
    "Curricular units 2nd sem (credited)": "s2_credited",
    "Curricular units 2nd sem (enrolled)": "s2_enrolled",
    "Curricular units 2nd sem (evaluations)": "s2_evaluations",
    "Curricular units 2nd sem (approved)": "s2_approved",
    "Curricular units 2nd sem (grade)": "s2_grade",
    "Curricular units 2nd sem (without evaluations)": "s2_no_eval",
    "Unemployment rate": "unemployment_rate", "Inflation rate": "inflation_rate",
    "GDP": "gdp_growth", "Target": "target",
}

NOMINAL = ["marital_status", "application_mode", "course", "previous_qualification",
           "nationality", "mother_qualification", "father_qualification",
           "mother_occupation", "father_occupation"]
BINARY = ["daytime_attendance", "displaced", "special_needs", "debtor",
          "tuition_up_to_date", "gender_male", "scholarship", "international"]
ENROLL_NUM = ["application_order", "age_at_enrollment", "unemployment_rate",
              "inflation_rate", "gdp_growth"]
S1_RAW = ["s1_credited", "s1_enrolled", "s1_evaluations", "s1_approved", "s1_grade", "s1_no_eval"]
S2_RAW = ["s2_credited", "s2_enrolled", "s2_evaluations", "s2_approved", "s2_grade", "s2_no_eval"]

# Engineered features (defined in engineer())
S1_ENG = ["s1_approval_rate", "s1_eval_success", "s1_zero_activity"]
S2_ENG = ["s2_approval_rate", "s2_eval_success", "s2_zero_activity",
          "grade_trend", "approval_trend", "total_approved", "cum_approval_rate"]
ENROLL_ENG = ["age_group_mature", "parents_higher_ed"]

# Course labels (Kaggle/Zenodo v1 coding; verified against published course sizes)
COURSE_LABELS = {
    1: "Biofuel Prod. Tech.", 2: "Animation & Multimedia", 3: "Social Service (eve.)",
    4: "Agronomy", 5: "Communication Design", 6: "Veterinary Nursing",
    7: "Informatics Eng.", 8: "Equinculture", 9: "Management", 10: "Social Service",
    11: "Tourism", 12: "Nursing", 13: "Oral Hygiene", 14: "Advertising & Mktg.",
    15: "Journalism & Comm.", 16: "Basic Education", 17: "Management (eve.)",
}

PRETTY = {
    "s2_approved": "2nd-sem units approved", "s1_approved": "1st-sem units approved",
    "s2_grade": "2nd-sem mean grade", "s1_grade": "1st-sem mean grade",
    "s2_approval_rate": "2nd-sem approval rate", "s1_approval_rate": "1st-sem approval rate",
    "cum_approval_rate": "Cumulative approval rate", "total_approved": "Total units approved",
    "tuition_up_to_date": "Tuition fees up to date", "scholarship": "Scholarship holder",
    "debtor": "Debtor", "age_at_enrollment": "Age at enrolment", "gender_male": "Gender (male)",
    "s2_enrolled": "2nd-sem units enrolled", "s1_enrolled": "1st-sem units enrolled",
    "s2_evaluations": "2nd-sem evaluations", "s1_evaluations": "1st-sem evaluations",
    "s1_eval_success": "1st-sem evaluation success", "s2_eval_success": "2nd-sem evaluation success",
    "grade_trend": "Grade trend (S2-S1)", "approval_trend": "Approval-rate trend (S2-S1)",
    "course": "Course", "application_mode": "Application mode",
    "unemployment_rate": "Unemployment rate", "inflation_rate": "Inflation rate",
    "gdp_growth": "GDP growth", "application_order": "Application order",
    "displaced": "Displaced", "mother_occupation": "Mother's occupation",
    "father_occupation": "Father's occupation", "mother_qualification": "Mother's qualification",
    "father_qualification": "Father's qualification", "previous_qualification": "Previous qualification",
    "marital_status": "Marital status", "nationality": "Nationality",
    "daytime_attendance": "Daytime attendance", "international": "International",
    "special_needs": "Special educational needs", "s1_credited": "1st-sem units credited",
    "s2_credited": "2nd-sem units credited", "s1_no_eval": "1st-sem units w/o evaluation",
    "s2_no_eval": "2nd-sem units w/o evaluation", "s1_zero_activity": "No 1st-sem activity",
    "s2_zero_activity": "No 2nd-sem activity", "age_group_mature": "Mature student (>=25)",
    "parents_higher_ed": "Parent with higher education",
}


def pretty(name: str) -> str:
    base = name.split("__")[-1]
    for nom in NOMINAL:
        if base.startswith(nom + "_"):
            code = base[len(nom) + 1:]
            if nom == "course" and code.isdigit():
                return f"Course: {COURSE_LABELS.get(int(code), code)}"
            return f"{PRETTY.get(nom, nom)} = {code}"
    return PRETTY.get(base, base)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA, encoding="utf-8-sig").rename(columns=RENAME)
    return df


def _safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise feature engineering (no leakage: uses only the row)."""
    d = df.copy()
    d["s1_approval_rate"] = _safe_div(d.s1_approved, d.s1_enrolled)
    d["s2_approval_rate"] = _safe_div(d.s2_approved, d.s2_enrolled)
    d["s1_eval_success"] = _safe_div(d.s1_approved, d.s1_evaluations)
    d["s2_eval_success"] = _safe_div(d.s2_approved, d.s2_evaluations)
    d["s1_zero_activity"] = ((d.s1_enrolled > 0) & (d.s1_evaluations == 0)).astype(int)
    d["s2_zero_activity"] = ((d.s2_enrolled > 0) & (d.s2_evaluations == 0)).astype(int)
    d["grade_trend"] = d.s2_grade - d.s1_grade
    d["approval_trend"] = d.s2_approval_rate - d.s1_approval_rate
    d["total_approved"] = d.s1_approved + d.s2_approved
    d["cum_approval_rate"] = _safe_div(d.s1_approved + d.s2_approved, d.s1_enrolled + d.s2_enrolled)
    d["age_group_mature"] = (d.age_at_enrollment >= 25).astype(int)
    # Parental qualification codes 2-5 = completed higher-education degree
    # (bachelor's, degree, master's, doctorate) in the dataset's data dictionary.
    he = {2, 3, 4, 5}
    d["parents_higher_ed"] = (d.mother_qualification.isin(he) | d.father_qualification.isin(he)).astype(int)
    return d


STAGES = {
    "S0": {"label": "Enrolment", "num": ENROLL_NUM + ENROLL_ENG, "bin": BINARY, "nom": NOMINAL},
    "S1": {"label": "End of semester 1", "num": ENROLL_NUM + ENROLL_ENG + S1_RAW + S1_ENG,
           "bin": BINARY, "nom": NOMINAL},
    "S2": {"label": "End of semester 2",
           "num": ENROLL_NUM + ENROLL_ENG + S1_RAW + S1_ENG + S2_RAW + S2_ENG,
           "bin": BINARY, "nom": NOMINAL},
}


def stage_columns(stage: str):
    s = STAGES[stage]
    return s["num"], s["bin"], s["nom"]


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
