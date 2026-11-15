"""Step 2: staged model development, tuning, repeated-CV comparison, significance testing,
and held-out test evaluation.

Design
  * Stratified 80/20 train/test split (seed 42); test set untouched until final evaluation.
  * Three prediction stages: S0 enrolment, S1 end of semester 1, S2 end of semester 2.
  * 7 learners x 3 stages; hyper-parameters tuned by randomised search (5-fold CV, macro-F1)
    on the training set only.
  * Tuned configurations compared with 5x3 repeated stratified CV on the training set;
    Friedman + Nemenyi tests and Nadeau-Bengio corrected resampled t-tests.
  * Final models refit on the full training set and scored on the test set with
    1,000-sample bootstrap 95% CIs.
Run: python src/02_modeling.py
"""
import json, time, warnings
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     RandomizedSearchCV)
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef,
                             cohen_kappa_score, roc_auc_score, average_precision_score, recall_score,
                             precision_score, log_loss, confusion_matrix)
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

from common import load_raw, engineer, RES, SEED, CLASSES, STAGES
from models import model_zoo, pipeline_for

warnings.filterwarnings("ignore")
N_ITER = 15
t0 = time.time()

df = engineer(load_raw())
y_all = df.target.map({c: i for i, c in enumerate(CLASSES)}).values  # 0 Dropout,1 Enrolled,2 Graduate
X_all = df.drop(columns="target")
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y_all, random_state=SEED)
np.save(RES / "idx_train.npy", idx_tr); np.save(RES / "idx_test.npy", idx_te)
Xtr, Xte, ytr, yte = X_all.iloc[idx_tr], X_all.iloc[idx_te], y_all[idx_tr], y_all[idx_te]
print(f"train {len(ytr)}  test {len(yte)}")


def metrics(y, proba):
    pred = proba.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "mcc": matthews_corrcoef(y, pred),
        "kappa": cohen_kappa_score(y, pred),
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "log_loss": log_loss(y, proba, labels=[0, 1, 2]),
        "f1_dropout": f1_score(y, pred, labels=[0], average="macro"),
        "f1_enrolled": f1_score(y, pred, labels=[1], average="macro"),
        "f1_graduate": f1_score(y, pred, labels=[2], average="macro"),
        "recall_dropout": recall_score(y, pred, labels=[0], average="macro"),
        "precision_dropout": precision_score(y, pred, labels=[0], average="macro", zero_division=0),
        "pr_auc_dropout": average_precision_score((y == 0).astype(int), proba[:, 0]),
    }


zoo = model_zoo()
inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
best_params, cv_rows, test_rows, timing = {}, [], [], []

for stage in STAGES:
    for name, (est, space) in zoo.items():
        ts = time.time()
        pipe = pipeline_for(stage, clone(est))
        # SVM: tune without Platt scaling (5x faster); probability re-enabled for evaluation
        if name == "SVM":
            pipe.set_params(model__probability=False)
        search = RandomizedSearchCV(pipe, {f"model__{k}": v for k, v in space.items()}, n_iter=N_ITER,
                                    scoring="f1_macro", cv=inner, random_state=SEED, n_jobs=1,
                                    refit=False)
        search.fit(Xtr, ytr)
        bp = search.best_params_
        best_params[f"{stage}|{name}"] = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                                         for k, v in bp.items()}
        tuned = pipeline_for(stage, clone(est)).set_params(**bp)

        # repeated CV on the training set
        for fold, (a, b) in enumerate(outer.split(Xtr, ytr)):
            m = clone(tuned).fit(Xtr.iloc[a], ytr[a])
            r = metrics(ytr[b], m.predict_proba(Xtr.iloc[b]))
            r.update(stage=stage, model=name, fold=fold)
            cv_rows.append(r)

        # final refit + held-out test
        final = clone(tuned).fit(Xtr, ytr)
        proba = final.predict_proba(Xte)
        np.save(RES / f"proba_test_{stage}_{name}.npy", proba)
        joblib.dump(final, RES / f"model_{stage}_{name}.joblib")
        r = metrics(yte, proba); r.update(stage=stage, model=name, inner_cv_macro_f1=search.best_score_)
        test_rows.append(r)
        el = time.time() - ts
        timing.append({"stage": stage, "model": name, "seconds": el})
        print(f"[{time.time()-t0:7.0f}s] {stage} {name:9s} innerF1={search.best_score_:.4f} "
              f"testF1={r['macro_f1']:.4f} ({el:.0f}s)", flush=True)

cv = pd.DataFrame(cv_rows); cv.to_csv(RES / "cv_fold_scores.csv", index=False)
test = pd.DataFrame(test_rows); test.to_csv(RES / "test_scores.csv", index=False)
pd.DataFrame(timing).to_csv(RES / "timing.csv", index=False)
(RES / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

# ------------------------------------------------------------------ CV summary
summ = cv.groupby(["stage", "model"]).agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.drop(columns=[c for c in summ.columns if c.startswith("fold")]).to_csv(RES / "cv_summary.csv")


# ------------------------------------------------------------------ significance tests
def corrected_ttest(d, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold differences."""
    k = len(d); mean = d.mean(); var = d.var(ddof=1)
    t = mean / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return t, p

n_te_fold = len(ytr) // 5; n_tr_fold = len(ytr) - n_te_fold
sig = {}
fried_rows, pair_rows = [], []
for stage in STAGES:
    piv = cv[cv.stage == stage].pivot(index="fold", columns="model", values="macro_f1")
    chi2, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    fried_rows.append({"stage": stage, "friedman_chi2": chi2, "p": p,
                       **{f"rank_{m}": r for m, r in ranks.items()}})
    nem = sp.posthoc_nemenyi_friedman(piv.values)
    nem.index = nem.columns = piv.columns
    nem.to_csv(RES / f"nemenyi_{stage}.csv")
    best = piv.mean().idxmax()
    rows = []
    for m in piv.columns:
        if m == best:
            continue
        t, pp = corrected_ttest((piv[best] - piv[m]).values, n_tr_fold, n_te_fold)
        rows.append({"stage": stage, "best": best, "vs": m, "mean_diff": (piv[best] - piv[m]).mean(),
                     "t": t, "p": pp})
    pr = pd.DataFrame(rows); pr["p_holm"] = multipletests(pr.p, method="holm")[1]
    pair_rows.append(pr)
pd.DataFrame(fried_rows).to_csv(RES / "friedman.csv", index=False)
pd.concat(pair_rows).to_csv(RES / "pairwise_best_vs_others.csv", index=False)

# stage gains for every model (S1 vs S0, S2 vs S1)
rows = []
for name in zoo:
    for a, b in [("S0", "S1"), ("S1", "S2"), ("S0", "S2")]:
        fa = cv[(cv.stage == a) & (cv.model == name)].sort_values("fold").macro_f1.values
        fb = cv[(cv.stage == b) & (cv.model == name)].sort_values("fold").macro_f1.values
        t, pp = corrected_ttest(fb - fa, n_tr_fold, n_te_fold)
        rows.append({"model": name, "from": a, "to": b, "gain_macro_f1": (fb - fa).mean(), "t": t, "p": pp})
sg = pd.DataFrame(rows); sg["p_holm"] = multipletests(sg.p, method="holm")[1]
sg.to_csv(RES / "stage_gains.csv", index=False)

# ------------------------------------------------------------------ bootstrap CIs on the test set
rng = np.random.default_rng(SEED)
B = 1000
boot_rows = []
for stage in STAGES:
    for name in zoo:
        proba = np.load(RES / f"proba_test_{stage}_{name}.npy")
        vals = {k: [] for k in ["macro_f1", "balanced_accuracy", "roc_auc_ovr", "recall_dropout", "mcc"]}
        for _ in range(B):
            i = rng.integers(0, len(yte), len(yte))
            mm = metrics(yte[i], proba[i])
            for k in vals:
                vals[k].append(mm[k])
        r = {"stage": stage, "model": name}
        for k, v in vals.items():
            r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        boot_rows.append(r)
pd.DataFrame(boot_rows).to_csv(RES / "test_bootstrap_ci.csv", index=False)

print(f"Done in {time.time()-t0:.0f}s")
print(cv.groupby(["stage", "model"]).macro_f1.mean().unstack(0).round(4))
print(test.pivot(index="model", columns="stage", values="macro_f1").round(4))
