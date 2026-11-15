"""Model zoo, preprocessing and search spaces (shared by scripts 02 and 03)."""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from common import SEED, stage_columns


class BalancedXGB(XGBClassifier):
    """XGBoost with inverse-frequency ('balanced') sample weights computed inside fit,
    so weighting is re-derived on every training fold (no leakage)."""
    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y), verbose=False)


class BalancedMLP(MLPClassifier):
    """Multilayer perceptron with balanced sample weights computed inside fit."""
    def fit(self, X, y, **kw):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def make_preprocessor(stage: str) -> ColumnTransformer:
    num, binary, nom = stage_columns(stage)
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        # rare nominal levels (< 30 students) pooled into an 'infrequent' level
        ("nom", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(min_frequency=30, handle_unknown="infrequent_if_exist",
                                                   sparse_output=False))]), nom),
    ], verbose_feature_names_out=True)


def model_zoo():
    """Returns {name: (estimator, search_space)}. All learners use class re-weighting."""
    return {
        "LR": (LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
               {"C": loguniform(1e-3, 1e2)}),
        "SVM": (SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
                {"C": loguniform(1e-1, 1e2), "gamma": loguniform(1e-4, 1e-1)}),
        "RF": (RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                      n_jobs=2, random_state=SEED),
               {"max_depth": [None, 8, 12, 16, 24], "min_samples_leaf": randint(1, 10),
                "max_features": ["sqrt", 0.3, 0.5]}),
        "XGBoost": (BalancedXGB(n_estimators=400, tree_method="hist", eval_metric="mlogloss",
                                n_jobs=2, random_state=SEED),
                    {"learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 8),
                     "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                     "min_child_weight": randint(1, 10), "reg_lambda": loguniform(0.1, 10)}),
        "LightGBM": (LGBMClassifier(n_estimators=400, class_weight="balanced", n_jobs=2,
                                    random_state=SEED, verbose=-1),
                     {"learning_rate": loguniform(0.01, 0.2), "num_leaves": randint(8, 64),
                      "min_child_samples": randint(10, 60), "subsample": uniform(0.6, 0.4),
                      "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5),
                      "reg_lambda": loguniform(0.01, 10)}),
        "CatBoost": (CatBoostClassifier(iterations=500, auto_class_weights="Balanced",
                                        loss_function="MultiClass", thread_count=2,
                                        random_seed=SEED, verbose=0, allow_writing_files=False),
                     {"learning_rate": loguniform(0.02, 0.2), "depth": randint(3, 8),
                      "l2_leaf_reg": loguniform(1, 10)}),
        "MLP": (BalancedMLP(max_iter=600, early_stopping=True, n_iter_no_change=20,
                            random_state=SEED),
                {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                 "alpha": loguniform(1e-5, 1e-1), "learning_rate_init": loguniform(1e-4, 1e-2)}),
    }


def pipeline_for(stage, estimator):
    return Pipeline([("prep", make_preprocessor(stage)), ("model", estimator)])
