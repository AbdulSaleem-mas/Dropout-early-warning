"""Step 1: data audit, cleaning report, exploratory analysis and inferential statistics.

Outputs: results/*.csv tables and figures/*.png|pdf
Run:     python src/01_eda_stats.py
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import (load_raw, engineer, savefig, RES, CLASSES, CLASS_COLORS, NOMINAL,
                    BINARY, ENROLL_NUM, S1_RAW, S2_RAW, S1_ENG, S2_ENG, COURSE_LABELS,
                    pretty, COL1, COL2, INK, INK2, MUTED)

raw = load_raw()
audit = {}

# ------------------------------------------------------------------ 1. data audit
audit["n_rows"], audit["n_cols"] = map(int, raw.shape)
audit["missing_cells"] = int(raw.isna().sum().sum())
audit["duplicate_rows"] = int(raw.duplicated().sum())
audit["class_counts"] = raw.target.value_counts().to_dict()
audit["class_pct"] = (raw.target.value_counts(normalize=True) * 100).round(2).to_dict()
# Logical-consistency checks
for s in ("s1", "s2"):
    audit[f"{s}_approved_gt_enrolled"] = int((raw[f"{s}_approved"] > raw[f"{s}_enrolled"] + raw[f"{s}_credited"]).sum())
    audit[f"{s}_grade0_but_approved"] = int(((raw[f"{s}_grade"] == 0) & (raw[f"{s}_approved"] > 0)).sum())
    audit[f"{s}_grade_pos_but_none_approved"] = int(((raw[f"{s}_grade"] > 0) & (raw[f"{s}_approved"] == 0)).sum())
    audit[f"{s}_no_units_enrolled"] = int((raw[f"{s}_enrolled"] == 0).sum())
    audit[f"{s}_enrolled_but_no_eval"] = int(((raw[f"{s}_enrolled"] > 0) & (raw[f"{s}_evaluations"] == 0)).sum())
audit["grade_range_violations"] = int(((raw.s1_grade > 20) | (raw.s2_grade > 20) |
                                       (raw.s1_grade < 0) | (raw.s2_grade < 0)).sum())
audit["age_range"] = [int(raw.age_at_enrollment.min()), int(raw.age_at_enrollment.max())]

# Outliers (Tukey IQR rule, 1.5 x IQR) on continuous / count variables
num_vars = ENROLL_NUM + S1_RAW + S2_RAW
out_rows = []
for c in num_vars:
    q1, q3 = raw[c].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((raw[c] < lo) | (raw[c] > hi)).sum())
    z_out = int((np.abs(stats.zscore(raw[c])) > 3).sum())
    out_rows.append({"variable": pretty(c), "Q1": q1, "Q3": q3, "lower_fence": lo, "upper_fence": hi,
                     "n_iqr_outliers": n_out, "pct_iqr_outliers": round(100 * n_out / len(raw), 2),
                     "n_abs_z_gt_3": z_out, "min": raw[c].min(), "max": raw[c].max()})
outliers = pd.DataFrame(out_rows)
outliers.to_csv(RES / "table_outliers.csv", index=False)
(RES / "data_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))

df = engineer(raw)

# ------------------------------------------------------------------ 2. descriptive table
def kw_eps2(x, y):
    groups = [x[y == k] for k in CLASSES]
    H, p = stats.kruskal(*groups)
    n, k = len(x), len(groups)
    eps2 = (H - k + 1) / (n - k)  # epsilon-squared (Tomczak & Tomczak, 2014)
    return H, p, max(eps2, 0.0)

cont = ENROLL_NUM + S1_RAW + S2_RAW + S1_ENG[:2] + S2_ENG[:2] + ["grade_trend", "cum_approval_rate"]
rows = []
for c in cont:
    H, p, e2 = kw_eps2(df[c].values, df.target.values)
    r = {"variable": pretty(c)}
    for k in CLASSES:
        v = df.loc[df.target == k, c]
        r[k] = f"{v.mean():.2f} ({v.std():.2f})"
    r.update({"H": H, "p": p, "epsilon2": e2})
    rows.append(r)
desc = pd.DataFrame(rows)
desc["p_holm"] = multipletests(desc.p, method="holm")[1]
desc.sort_values("epsilon2", ascending=False).to_csv(RES / "table_numeric_by_class.csv", index=False)

# ------------------------------------------------------------------ 3. categorical association
def cramers_v(ct):
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    # bias-corrected Cramer's V (Bergsma, 2013)
    phi2 = chi2 / n
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v = np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))
    low_exp = float((exp < 5).mean())
    return chi2, p, dof, v, low_exp

rows = []
for c in BINARY + NOMINAL + ["age_group_mature", "parents_higher_ed"]:
    ct = pd.crosstab(df[c], df.target)
    chi2, p, dof, v, low = cramers_v(ct)
    r = {"variable": pretty(c), "levels": ct.shape[0], "chi2": chi2, "dof": dof, "p": p,
         "cramers_v": v, "pct_cells_expected_lt5": round(100 * low, 1)}
    if c in BINARY + ["age_group_mature", "parents_higher_ed"]:
        r["dropout_rate_if_1"] = round(100 * (df.loc[df[c] == 1, "target"] == "Dropout").mean(), 1)
        r["dropout_rate_if_0"] = round(100 * (df.loc[df[c] == 0, "target"] == "Dropout").mean(), 1)
    rows.append(r)
cat = pd.DataFrame(rows)
cat["p_holm"] = multipletests(cat.p, method="holm")[1]
cat.sort_values("cramers_v", ascending=False).to_csv(RES / "table_categorical_assoc.csv", index=False)

# ------------------------------------------------------------------ 4. multicollinearity (VIF)
vif_vars = ENROLL_NUM + S1_RAW + S2_RAW
X = df[vif_vars].astype(float)
X = (X - X.mean()) / X.std()
X.insert(0, "const", 1.0)
vif = pd.DataFrame({"variable": [pretty(v) for v in vif_vars],
                    "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(vif_vars))]})
vif.sort_values("VIF", ascending=False).to_csv(RES / "table_vif.csv", index=False)

# ------------------------------------------------------------------ FIGURES
# Fig 1: class distribution
fig, ax = plt.subplots(figsize=(COL1, 1.5))
counts = df.target.value_counts().reindex(["Graduate", "Dropout", "Enrolled"])
ax.barh(counts.index[::-1], counts.values[::-1], color=[CLASS_COLORS[k] for k in counts.index[::-1]],
        height=0.55)
for i, (k, v) in enumerate(zip(counts.index[::-1], counts.values[::-1])):
    ax.text(v + 30, i, f"{v:,} ({100*v/len(df):.1f}%)", va="center", color=INK, fontsize=7)
ax.set_xlim(0, counts.max() * 1.35)
ax.set_xlabel("Number of students")
ax.grid(axis="y", visible=False)
savefig(fig, "fig01_class_distribution")

# Fig 2: outcome composition by course (100% stacked horizontal bars)
ct = pd.crosstab(df.course, df.target, normalize="index")[["Dropout", "Enrolled", "Graduate"]] * 100
ct = ct.sort_values("Dropout")
n_by_course = df.course.value_counts()
fig, ax = plt.subplots(figsize=(COL2, 3.6))
left = np.zeros(len(ct))
labels = [f"{COURSE_LABELS[c]} (n={n_by_course[c]})" for c in ct.index]
for k in ["Dropout", "Enrolled", "Graduate"]:
    ax.barh(labels, ct[k], left=left, color=CLASS_COLORS[k], label=k, height=0.7,
            edgecolor="white", linewidth=1)
    for i, v in enumerate(ct[k]):
        if v >= 9:
            ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white")
    left += ct[k].values
ax.set_xlim(0, 100)
ax.set_xlabel("Share of students (%)")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="y", visible=False)
savefig(fig, "fig02_outcome_by_course")

# Fig 3: academic trajectory variables by outcome (box plots, 2x2)
fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.8))
for ax, c in zip(axes.ravel(), ["s1_approved", "s2_approved", "s1_grade", "s2_grade"]):
    data = [df.loc[df.target == k, c] for k in CLASSES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgecolor=MUTED),
                    medianprops=dict(color=INK, linewidth=1.2))
    for patch, k in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLASS_COLORS[k]); patch.set_alpha(0.85); patch.set_edgecolor(INK2)
    ax.set_xticks([1, 2, 3], CLASSES)
    ax.set_title(pretty(c), color=INK)
    ax.grid(axis="x", visible=False)
fig.tight_layout()
savefig(fig, "fig03_academic_by_outcome")

# Fig 4: Spearman correlation heatmap (diverging blue <-> red, grey midpoint)
corr_vars = ENROLL_NUM + S1_RAW + S2_RAW + ["tuition_up_to_date", "debtor", "scholarship", "gender_male"]
tmp = df[corr_vars].copy()
tmp["target_dropout"] = (df.target == "Dropout").astype(int)
rho = tmp.corr(method="spearman")
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(COL2, 6.2))
im = ax.imshow(rho.values, cmap=cmap, vmin=-1, vmax=1)
names = [pretty(c) if c != "target_dropout" else "Dropout (1/0)" for c in rho.columns]
ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
ax.set_yticks(range(len(names)), names)
ax.grid(False)
for i in range(len(names)):
    for j in range(len(names)):
        v = rho.values[i, j]
        if abs(v) >= 0.3 and i != j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5,
                    color="white" if abs(v) > 0.6 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Spearman's rho")
savefig(fig, "fig04_spearman_heatmap")
rho.to_csv(RES / "spearman_matrix.csv")

# Fig 5: effect-size ranking (univariate association strength with outcome)
eff = pd.concat([
    desc[["variable", "epsilon2"]].rename(columns={"epsilon2": "effect"}).assign(kind="Kruskal-Wallis ε²"),
    cat[["variable", "cramers_v"]].rename(columns={"cramers_v": "effect"}).assign(kind="Cramér's V"),
]).sort_values("effect", ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(COL1, 3.6))
colors = ["#2a78d6" if k == "Kruskal-Wallis ε²" else "#eb6834" for k in eff.kind]
ax.barh(eff.variable, eff.effect, color=colors, height=0.65)
ax.set_xlabel("Effect size")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2a78d6", label="Kruskal-Wallis ε² (numeric)"),
                   Patch(color="#eb6834", label="Cramér's V (categorical)")],
          loc="lower right")
ax.grid(axis="y", visible=False)
savefig(fig, "fig05_effect_sizes")

# Fig 6: dropout rate by financial / social status with Wilson 95% CIs
fin = ["tuition_up_to_date", "debtor", "scholarship", "gender_male", "displaced", "age_group_mature"]
rows = []
for c in fin:
    for lvl in (0, 1):
        sub = df[df[c] == lvl]
        k = int((sub.target == "Dropout").sum()); n = len(sub)
        lo, hi = proportion_confint(k, n, method="wilson")
        rows.append({"var": pretty(c), "level": "Yes" if lvl else "No", "rate": 100 * k / n,
                     "lo": 100 * lo, "hi": 100 * hi, "n": n})
fr = pd.DataFrame(rows)
fr.to_csv(RES / "table_dropout_rates_groups.csv", index=False)
fig, ax = plt.subplots(figsize=(COL1, 2.6))
ys = np.arange(len(fin))
for off, lvl, col in [(-0.15, "No", "#86b6ef"), (0.15, "Yes", "#104281")]:
    s = fr[fr.level == lvl].reset_index(drop=True)
    ax.errorbar(s.rate, ys + off, xerr=[s.rate - s.lo, s.hi - s.rate], fmt="o", color=col,
                markersize=4, capsize=2, linewidth=1, label=lvl)
ax.set_yticks(ys, [pretty(c) for c in fin])
ax.axvline(100 * (df.target == "Dropout").mean(), color=MUTED, linestyle="--", linewidth=0.8)
ax.set_xlabel("Dropout rate (%) with Wilson 95% CI")
ax.legend(title="Status", loc="lower right")
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
savefig(fig, "fig06_dropout_rate_groups")

print("EDA complete.")
