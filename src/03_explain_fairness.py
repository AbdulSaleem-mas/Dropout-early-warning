"""Step 3: model comparison figures, statistical comparison, calibration, SHAP explanations,
feature-selection curve, imbalance-handling ablation, fairness slicing and intervention
capacity analysis.  Requires outputs of 02_modeling.py.
Run: python src/03_explain_fairness.py
"""
import json, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import scikit_posthocs as sp
from sklearn.base import clone
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, brier_score_loss, f1_score, recall_score,
                             precision_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from common import (load_raw, engineer, RES, FIG, SEED, CLASSES, CLASS_COLORS, STAGE_COLORS, STAGES,
                    NOMINAL, savefig, pretty, COL1, COL2, INK, INK2, MUTED, HIGHLIGHT, BASE)
from models import model_zoo, pipeline_for, make_preprocessor

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)

df = engineer(load_raw())
y_all = df.target.map({c: i for i, c in enumerate(CLASSES)}).values
X_all = df.drop(columns="target")
idx_tr, idx_te = np.load(RES / "idx_train.npy"), np.load(RES / "idx_test.npy")
Xtr, Xte, ytr, yte = X_all.iloc[idx_tr], X_all.iloc[idx_te], y_all[idx_tr], y_all[idx_te]

cv = pd.read_csv(RES / "cv_fold_scores.csv")
test = pd.read_csv(RES / "test_scores.csv")
boot = pd.read_csv(RES / "test_bootstrap_ci.csv")
params = json.loads((RES / "best_params.json").read_text())
MODELS = list(model_zoo().keys())
STAGE_LBL = {s: f"{s}: {STAGES[s]['label']}" for s in STAGES}

# ------------------------------------------------------------------ best model selection
cvm = cv.groupby(["stage", "model"]).macro_f1.mean().unstack(0)
BEST = {s: cvm[s].idxmax() for s in STAGES}
print("Best by CV macro-F1:", BEST)
(RES / "best_models.json").write_text(json.dumps(BEST, indent=2))

# ------------------------------------------------------------------ Table: CV summary
keys = ["macro_f1", "balanced_accuracy", "accuracy", "mcc", "roc_auc_ovr", "recall_dropout", "pr_auc_dropout"]
g = cv.groupby(["stage", "model"])[keys]
tab = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
tab.reset_index().to_csv(RES / "table_cv_models.csv", index=False)

tt = test.merge(boot, on=["stage", "model"])
for k in ["macro_f1", "balanced_accuracy", "roc_auc_ovr", "recall_dropout", "mcc"]:
    tt[f"{k}_ci"] = tt.apply(lambda r: f"{r[k]:.3f} [{r[k+'_lo']:.3f}, {r[k+'_hi']:.3f}]", axis=1)
tt[["stage", "model", "accuracy", "macro_f1_ci", "balanced_accuracy_ci", "mcc_ci", "roc_auc_ovr_ci",
    "recall_dropout_ci", "f1_dropout", "f1_enrolled", "f1_graduate", "pr_auc_dropout", "log_loss"]] \
    .to_csv(RES / "table_test_models.csv", index=False)

# ------------------------------------------------------------------ Fig 7: model x stage dot plot
fig, ax = plt.subplots(figsize=(COL2, 2.8))
order = cvm["S2"].sort_values().index.tolist()
ys = np.arange(len(order))
for off, s in zip([-0.22, 0, 0.22], STAGES):
    sub = cv[cv.stage == s].groupby("model").macro_f1
    m, sd = sub.mean().reindex(order), sub.std().reindex(order)
    ax.errorbar(m, ys + off, xerr=sd, fmt="o", color=STAGE_COLORS[s], markersize=5, capsize=2,
                linewidth=1, label=STAGE_LBL[s])
ax.set_yticks(ys, order)
ax.set_xlabel("Macro-F1 (mean ± SD, 5×3 repeated stratified CV on training set)")
ax.legend(loc="upper left", ncol=1)
ax.grid(axis="y", visible=False)
savefig(fig, "fig07_model_stage_macroF1")

# ------------------------------------------------------------------ Fig 8: critical difference diagrams
fig, axes = plt.subplots(3, 1, figsize=(COL2, 3.9))
for ax, s in zip(axes, STAGES):
    piv = cv[cv.stage == s].pivot(index="fold", columns="model", values="macro_f1")
    ranks = piv.rank(axis=1, ascending=False).mean()
    nem = sp.posthoc_nemenyi_friedman(piv.values); nem.index = nem.columns = piv.columns
    sp.critical_difference_diagram(ranks, nem, ax=ax, label_fmt_left="{label} ({rank:.2f})",
                                   label_fmt_right="({rank:.2f}) {label}",
                                   label_props={"fontsize": 7},
                                   crossbar_props={"color": INK2, "linewidth": 1.5},
                                   elbow_props={"color": MUTED})
    ax.set_title(STAGE_LBL[s], fontsize=8, color=INK, loc="left")
fig.tight_layout()
savefig(fig, "fig08_critical_difference")

# ------------------------------------------------------------------ Fig 9: confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))
from matplotlib.colors import LinearSegmentedColormap
seq = LinearSegmentedColormap.from_list("seq", ["#f5f9fe", "#9ec5f4", "#2a78d6", "#0d366b"])
for ax, s in zip(axes, STAGES):
    p = np.load(RES / f"proba_test_{s}_{BEST[s]}.npy").argmax(1)
    cm = confusion_matrix(yte, p); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap=seq, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]*100:.0f}%\n({cm[i,j]})", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), CLASSES, rotation=0); ax.set_yticks(range(3), CLASSES if s == "S0" else [""] * 3)
    ax.set_xlabel("Predicted"); ax.grid(False)
    if s == "S0": ax.set_ylabel("Actual")
    ax.set_title(f"{s} – {BEST[s]}", fontsize=8, color=INK)
fig.tight_layout()
savefig(fig, "fig09_confusion_matrices")

# ------------------------------------------------------------------ Fig 10: ROC (S2) + PR for dropout by stage
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))
p2 = np.load(RES / f"proba_test_S2_{BEST['S2']}.npy")
for k, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(yte == k, p2[:, k])
    axes[0].plot(fpr, tpr, color=CLASS_COLORS[c], lw=2, label=f"{c} (AUC={roc_auc_score(yte == k, p2[:, k]):.3f})")
axes[0].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
axes[0].set_title(f"(a) One-vs-rest ROC, S2 {BEST['S2']}", fontsize=8, color=INK, loc="left")
axes[0].legend(loc="lower right")
for s in STAGES:
    p = np.load(RES / f"proba_test_{s}_{BEST[s]}.npy")[:, 0]
    pr, rc, _ = precision_recall_curve(yte == 0, p)
    axes[1].plot(rc, pr, color=STAGE_COLORS[s], lw=2,
                 label=f"{s} {BEST[s]} (AP={average_precision_score(yte == 0, p):.3f})")
axes[1].axhline((yte == 0).mean(), ls="--", color=MUTED, lw=0.8)
axes[1].set_xlabel("Recall (dropout)"); axes[1].set_ylabel("Precision (dropout)")
axes[1].set_title("(b) Dropout precision–recall by stage", fontsize=8, color=INK, loc="left")
axes[1].legend(loc="lower left")
fig.tight_layout()
savefig(fig, "fig10_roc_pr")

# ------------------------------------------------------------------ Calibration (dropout probability)
cal_rows = []
fig, ax = plt.subplots(figsize=(COL1, 2.8))
ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
for s in STAGES:
    name = BEST[s]
    est = model_zoo()[name][0]
    tuned = pipeline_for(s, clone(est)).set_params(**params[f"{s}|{name}"])
    raw_p = np.load(RES / f"proba_test_{s}_{name}.npy")
    cal = CalibratedClassifierCV(tuned, method="isotonic", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    cal.fit(Xtr, ytr)
    cp = cal.predict_proba(Xte)
    np.save(RES / f"proba_test_{s}_{name}_calibrated.npy", cp)
    for lbl, p in [("uncalibrated", raw_p), ("isotonic", cp)]:
        frac, mean = calibration_curve(yte == 0, p[:, 0], n_bins=10, strategy="quantile")
        ece = np.average(np.abs(frac - mean), weights=np.histogram(p[:, 0], bins=np.quantile(p[:, 0], np.linspace(0, 1, 11)))[0])
        cal_rows.append({"stage": s, "model": name, "version": lbl,
                         "brier_dropout": brier_score_loss(yte == 0, p[:, 0]),
                         "ece_dropout": ece,
                         "macro_f1": f1_score(yte, p.argmax(1), average="macro"),
                         "recall_dropout": recall_score(yte, p.argmax(1), labels=[0], average="macro")})
        if s == "S2":
            ax.plot(mean, frac, marker="o", ms=4, lw=1.5,
                    color=HIGHLIGHT if lbl == "isotonic" else "#eb6834",
                    label=f"{name} {lbl} (Brier={brier_score_loss(yte == 0, p[:, 0]):.3f})")
ax.set_xlabel("Mean predicted dropout probability"); ax.set_ylabel("Observed dropout fraction")
ax.legend(loc="upper left", fontsize=6.5)
ax.set_title("Reliability diagram (S2, test set)", fontsize=8, color=INK, loc="left")
savefig(fig, "fig11_calibration")
pd.DataFrame(cal_rows).to_csv(RES / "table_calibration.csv", index=False)


# ------------------------------------------------------------------ SHAP
def shap_for(stage, name):
    model = joblib.load(RES / f"model_{stage}_{name}.joblib")
    prep, est = model.named_steps["prep"], model.named_steps["model"]
    Xt = prep.transform(Xte)
    fn = prep.get_feature_names_out()
    if name in ("RF", "XGBoost", "LightGBM", "CatBoost"):
        sv = shap.TreeExplainer(est).shap_values(Xt)
    elif name == "LR":
        sv = shap.LinearExplainer(est, prep.transform(Xtr)).shap_values(Xt)
    else:
        bg = shap.sample(prep.transform(Xtr), 100, random_state=SEED)
        sv = shap.KernelExplainer(est.predict_proba, bg).shap_values(Xt[:300], nsamples=200)
        Xt = Xt[:300]
    sv = np.array(sv)
    if sv.ndim == 3 and sv.shape[0] == 3 and sv.shape[-1] != 3:  # (classes, n, f) -> (n, f, classes)
        sv = np.transpose(sv, (1, 2, 0))
    return sv, Xt, fn

def group_name(f):
    base = f.split("__", 1)[1]
    for nom in NOMINAL:
        if base.startswith(nom + "_"):
            return nom
    return base

def shap_grouped(sv, fn, k):
    groups = pd.Series([group_name(f) for f in fn])
    S = pd.DataFrame(sv[:, :, k], columns=fn).T.groupby(groups.values).sum().T
    return S.abs().mean().sort_values(ascending=False)

shap_tables = {}
for s in ["S0", "S2"]:
    # use the best tree ensemble at this stage for TreeSHAP (exact, fast)
    tree_models = [m for m in ["LightGBM", "CatBoost", "XGBoost", "RF"]]
    name = BEST[s] if BEST[s] in tree_models else cvm[s][tree_models].idxmax()
    sv, Xt, fn = shap_for(s, name)
    imp = pd.DataFrame({c: shap_grouped(sv, fn, k) for k, c in enumerate(CLASSES)})
    imp["overall"] = imp.sum(1)
    imp = imp.sort_values("overall", ascending=False)
    imp.index = [pretty(i) for i in imp.index]
    imp.to_csv(RES / f"table_shap_{s}_{name}.csv")
    shap_tables[s] = (name, imp, sv, Xt, fn)

# Fig 12: grouped mean |SHAP| stacked by class for S0 and S2
fig, axes = plt.subplots(1, 2, figsize=(COL2, 3.6))
for ax, s, ttl in zip(axes, ["S0", "S2"], ["(a) Enrolment stage (S0)", "(b) End of semester 2 (S2)"]):
    name, imp, *_ = shap_tables[s]
    top = imp.head(12).iloc[::-1]
    left = np.zeros(len(top))
    for c in CLASSES:
        ax.barh(top.index, top[c], left=left, color=CLASS_COLORS[c], height=0.65, label=c,
                edgecolor="white", linewidth=0.8)
        left += top[c].values
    ax.set_title(f"{ttl} – {name}", fontsize=8, color=INK, loc="left")
    ax.set_xlabel("Mean |SHAP| (log-odds), summed over one-hot levels")
    ax.grid(axis="y", visible=False)
axes[1].legend(loc="lower right")
fig.tight_layout()
savefig(fig, "fig12_shap_importance")

# Fig 13: beeswarm for Dropout class at S2 (ungrouped features)
name, imp, sv, Xt, fn = shap_tables["S2"]
expl = shap.Explanation(values=sv[:, :, 0], data=Xt, feature_names=[pretty(f) for f in fn])
plt.figure(figsize=(COL2, 4.2))
shap.plots.beeswarm(expl, max_display=15, show=False, color_bar=True, plot_size=None)
fig = plt.gcf(); fig.set_size_inches(COL2, 4.2)
plt.xlabel("SHAP value for Dropout (log-odds contribution)")
savefig(fig, "fig13_shap_beeswarm_dropout_S2")

# ------------------------------------------------------------------ Feature selection: MI ranking + top-k curve (S2)
prep = make_preprocessor("S2").fit(Xtr)
Xt_tr = prep.transform(Xtr); fn = prep.get_feature_names_out()
mi = mutual_info_classif(Xt_tr, ytr, random_state=SEED)
mi_s = pd.Series(mi, index=fn).sort_values(ascending=False)
mi_df = mi_s.rename("mutual_information").reset_index().rename(columns={"index": "feature"})
mi_df["feature"] = mi_df.feature.map(pretty)
mi_df.to_csv(RES / "table_mutual_information_S2.csv", index=False)

best2 = BEST["S2"]
est2 = clone(model_zoo()[best2][0]).set_params(**{k.replace("model__", ""): v for k, v in params[f"S2|{best2}"].items()})
ks = [3, 5, 8, 10, 15, 20, 30, 45, 60, len(fn)]
fs_rows = []
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
for k in ks:
    cols = [list(fn).index(f) for f in mi_s.index[:k]]
    sc = []
    for a, b in cv5.split(Xt_tr, ytr):
        # MI ranking re-computed inside each fold to avoid selection leakage
        mi_f = mutual_info_classif(Xt_tr[a], ytr[a], random_state=SEED)
        sel = np.argsort(-mi_f)[:k]
        m = clone(est2).fit(Xt_tr[a][:, sel], ytr[a])
        sc.append(f1_score(ytr[b], m.predict(Xt_tr[b][:, sel]).ravel(), average="macro"))
    fs_rows.append({"k": k, "macro_f1_mean": np.mean(sc), "macro_f1_sd": np.std(sc, ddof=1)})
fs = pd.DataFrame(fs_rows); fs.to_csv(RES / "table_feature_selection_curve.csv", index=False)
fig, ax = plt.subplots(figsize=(COL1, 2.3))
ax.errorbar(fs.k, fs.macro_f1_mean, yerr=fs.macro_f1_sd, color=HIGHLIGHT, marker="o", ms=4, lw=2, capsize=2)
ax.set_xscale("log"); ax.set_xticks(ks, [str(k) for k in ks])
ax.set_xlabel("Number of top-ranked features (mutual information, in-fold)")
ax.set_ylabel("Macro-F1 (5-fold CV)")
ax.set_title(f"S2 – {best2}", fontsize=8, color=INK, loc="left")
savefig(fig, "fig14_feature_selection_curve")

# ------------------------------------------------------------------ Imbalance-handling ablation (S2 best model)
def unweighted(name, est):
    p = est.get_params()
    if name == "XGBoost":
        return XGBClassifier(**{k: v for k, v in p.items()})
    if name == "MLP":
        return MLPClassifier(**p)
    if name == "CatBoost":
        e = clone(est); e.set_params(auto_class_weights=None); return e
    e = clone(est); e.set_params(class_weight=None); return e

uw = unweighted(best2, est2)
variants = {
    "No re-balancing": Pipeline([("prep", make_preprocessor("S2")), ("model", uw)]),
    "Class weighting (adopted)": Pipeline([("prep", make_preprocessor("S2")), ("model", clone(est2))]),
    "SMOTE (in-fold)": ImbPipeline([("prep", make_preprocessor("S2")), ("smote", SMOTE(random_state=SEED)),
                                    ("model", clone(uw))]),
}
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
from sklearn.metrics import make_scorer
scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy",
           "recall_dropout": make_scorer(recall_score, labels=[0], average="macro"),
           "recall_enrolled": make_scorer(recall_score, labels=[1], average="macro"),
           "accuracy": "accuracy"}
ab_rows = []
for v, pipe in variants.items():
    r = cross_validate(pipe, Xtr, ytr, cv=rcv, scoring=scoring, n_jobs=1)
    ab_rows.append({"strategy": v, **{k: f"{r['test_'+k].mean():.4f} ± {r['test_'+k].std(ddof=1):.4f}" for k in scoring}})
pd.DataFrame(ab_rows).to_csv(RES / "table_imbalance_ablation.csv", index=False)
print(pd.DataFrame(ab_rows))

# ------------------------------------------------------------------ Fairness slicing (test set)
groups = {"Gender": ("gender_male", {0: "Female", 1: "Male"}),
          "Scholarship": ("scholarship", {0: "No", 1: "Yes"}),
          "Age at enrolment": ("age_group_mature", {0: "<25", 1: "≥25"}),
          "Displaced": ("displaced", {0: "No", 1: "Yes"}),
          "Debtor": ("debtor", {0: "No", 1: "Yes"})}
fr = []
for s in ["S0", "S2"]:
    p = np.load(RES / f"proba_test_{s}_{BEST[s]}.npy")
    pred = p.argmax(1)
    for gname, (col, lab) in groups.items():
        gv = Xte[col].values
        for lvl, lname in lab.items():
            m = gv == lvl
            yt, yp, pp = yte[m], pred[m], p[m, 0]
            tpr = recall_score(yt == 0, yp == 0, zero_division=0)
            fpr = ((yp == 0) & (yt != 0)).sum() / max(1, (yt != 0).sum())
            ppv = precision_score(yt == 0, yp == 0, zero_division=0)
            # bootstrap CI for TPR
            bt = []
            for _ in range(500):
                i = rng.integers(0, m.sum(), m.sum())
                if (yt[i] == 0).sum() > 0:
                    bt.append(recall_score(yt[i] == 0, yp[i] == 0, zero_division=0))
            fr.append({"stage": s, "model": BEST[s], "attribute": gname, "group": lname, "n": int(m.sum()),
                       "base_rate_dropout": (yt == 0).mean(), "tpr_dropout": tpr,
                       "tpr_lo": np.percentile(bt, 2.5), "tpr_hi": np.percentile(bt, 97.5),
                       "fpr_dropout": fpr, "ppv_dropout": ppv,
                       "macro_f1": f1_score(yt, yp, average="macro"),
                       "auc_dropout": roc_auc_score(yt == 0, pp) if 0 < (yt == 0).sum() < len(yt) else np.nan})
fair = pd.DataFrame(fr); fair.to_csv(RES / "table_fairness.csv", index=False)
# gaps
gap = fair.groupby(["stage", "attribute"]).agg(tpr_gap=("tpr_dropout", lambda x: x.max() - x.min()),
                                              fpr_gap=("fpr_dropout", lambda x: x.max() - x.min()),
                                              auc_gap=("auc_dropout", lambda x: x.max() - x.min())).reset_index()
gap.to_csv(RES / "table_fairness_gaps.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True)
for ax, s in zip(axes, ["S0", "S2"]):
    sub = fair[fair.stage == s].reset_index(drop=True)
    lbls = [f"{a}: {g}" for a, g in zip(sub.attribute, sub.group)]
    ys = np.arange(len(sub))
    ax.errorbar(sub.tpr_dropout, ys, xerr=[sub.tpr_dropout - sub.tpr_lo, sub.tpr_hi - sub.tpr_dropout],
                fmt="o", color=STAGE_COLORS[s], ms=4, capsize=2, lw=1, label="TPR (recall)")
    ax.scatter(sub.fpr_dropout, ys, marker="s", s=14, color="#eb6834", label="FPR", zorder=3)
    ax.set_yticks(ys, lbls); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_title(f"{s} – {BEST[s]}", fontsize=8, color=INK, loc="left")
    ax.set_xlabel("Rate for Dropout class")
    ax.grid(axis="y", visible=False)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
savefig(fig, "fig15_fairness_slicing")

# ------------------------------------------------------------------ Intervention capacity (cumulative gains)
fig, ax = plt.subplots(figsize=(COL1, 2.6))
cap_rows = []
n_drop = (yte == 0).sum()
for s in STAGES:
    p = np.load(RES / f"proba_test_{s}_{BEST[s]}.npy")[:, 0]
    order_ = np.argsort(-p)
    captured = np.cumsum(yte[order_] == 0) / n_drop
    frac = np.arange(1, len(p) + 1) / len(p)
    ax.plot(frac * 100, captured * 100, color=STAGE_COLORS[s], lw=2, label=f"{s} {BEST[s]}")
    for q in [0.10, 0.20, 0.30, 0.40]:
        k = int(round(q * len(p)))
        cap_rows.append({"stage": s, "model": BEST[s], "flagged_pct": int(q * 100),
                         "dropouts_captured_pct": 100 * captured[k - 1],
                         "precision_at_k": (yte[order_][:k] == 0).mean()})
ax.plot([0, 100], [0, 100], ls="--", color=MUTED, lw=0.8, label="Random")
ax.axvline(30, color="#c3c2b7", lw=0.8)
ax.set_xlabel("Students flagged, highest risk first (%)")
ax.set_ylabel("Eventual dropouts captured (%)")
ax.legend(loc="lower right")
savefig(fig, "fig16_capacity_gains")
pd.DataFrame(cap_rows).to_csv(RES / "table_capacity.csv", index=False)

print("Step 3 complete.")
