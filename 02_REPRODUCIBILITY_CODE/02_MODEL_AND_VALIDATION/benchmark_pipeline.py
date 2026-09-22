#!/usr/bin/env python3
"""Frozen preprocessing and repeated nested-CV building blocks.

This module is an execution engine, not a results script. It never selects or
excludes a dataset from model performance.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "data" / "00_tools" / "ml_vendor"
if sys.platform == "darwin" and VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


MASTER_SEED = 20260910
SPECIAL_COLUMNS = {"__source_row__", "__group_id__", "__target__"}


MODEL_SPACES: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0], "model__penalty": ["l1", "l2"]},
    "random_forest": {"model__n_estimators": [200, 500, 800], "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5, 10], "model__max_features": ["sqrt", "log2", 0.5]},
    "gradient_boosted_trees": {"model__learning_rate": [0.03, 0.05, 0.1, 0.2], "model__max_iter": [100, 200, 400], "model__max_leaf_nodes": [7, 15, 31, 63], "model__l2_regularization": [0.0, 0.1, 1.0, 10.0]},
}
for _space in MODEL_SPACES.values():
    _space["feature_selection__percentile"] = [25, 50, 75, 100]


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def load_reference(dataset_id: str) -> pd.DataFrame:
    freeze = json.loads((ROOT / "data" / "00_registry" / "cohort_freeze_01.json").read_text())
    entry = next(item for item in freeze["datasets"] if item["dataset_id"] == dataset_id)
    return pd.read_csv(ROOT / entry["analysis_reference"], low_memory=False)


def split_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    predictors = [column for column in frame.columns if column not in SPECIAL_COLUMNS]
    numeric, categorical = [], []
    for column in predictors:
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        else:
            numeric_fraction = pd.to_numeric(frame[column], errors="coerce").notna().mean()
            (numeric if numeric_fraction >= 0.98 else categorical).append(column)
    return numeric, categorical


def build_preprocessor(frame: pd.DataFrame, model_name: str) -> ColumnTransformer:
    numeric, categorical = split_columns(frame)
    scale = model_name == "logistic_regression"
    if model_name == "gradient_boosted_trees":
        numeric_pipe = SkPipeline([("impute", SimpleImputer(strategy="median"))])
        categorical_pipe = SkPipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        sparse_threshold = 0.0
    else:
        numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scale", StandardScaler()))
        numeric_pipe = SkPipeline(numeric_steps)
        categorical_pipe = SkPipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        sparse_threshold = 0.3
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )


def build_model(model_name: str, seed: int):
    if model_name == "logistic_regression":
        return LogisticRegression(solver="liblinear", max_iter=10000, random_state=seed)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    if model_name == "gradient_boosted_trees":
        return HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    raise KeyError(model_name)


def build_reference_pipeline(frame: pd.DataFrame, model_name: str, seed: int, percentile: int = 100) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(frame, model_name)),
        ("constant_filter", VarianceThreshold(0.0)),
        ("feature_selection", SelectPercentile(score_func=f_classif, percentile=percentile)),
        ("oversample", RandomOverSampler(random_state=stable_seed(seed, "oversample"))),
        ("model", build_model(model_name, seed)),
    ])


def run_c0_outer_fold(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    model_name: str,
    dataset_id: str,
    repeat: int,
    outer_fold: int,
    n_iter: int = 8,
    inner_folds: int = 3,
) -> dict[str, Any]:
    X = frame.drop(columns=list(SPECIAL_COLUMNS), errors="ignore")
    y = frame["__target__"].astype(int)
    groups = frame["__group_id__"].astype(str)
    if set(groups.iloc[train_index]) & set(groups.iloc[test_index]):
        raise RuntimeError("Outer group leakage detected")
    seed = stable_seed(MASTER_SEED, "model", dataset_id, model_name, "C0", repeat, outer_fold)
    inner_seed = stable_seed(MASTER_SEED, "inner", dataset_id, repeat, outer_fold)
    search_seed = stable_seed(MASTER_SEED, "search", dataset_id, model_name, "C0", repeat, outer_fold)
    pipeline = build_reference_pipeline(frame.iloc[train_index], model_name, seed)
    inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=inner_seed)
    inner_splits = list(inner.split(X.iloc[train_index], y.iloc[train_index], groups.iloc[train_index]))
    search = RandomizedSearchCV(
        pipeline,
        MODEL_SPACES[model_name],
        n_iter=min(n_iter, int(np.prod([len(v) for v in MODEL_SPACES[model_name].values()]))),
        scoring="roc_auc",
        cv=inner_splits,
        random_state=search_seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X.iloc[train_index], y.iloc[train_index])
    probabilities = search.best_estimator_.predict_proba(X.iloc[test_index])[:, 1]
    return {
        "dataset_id": dataset_id,
        "condition": "C0",
        "model": model_name,
        "repeat": repeat,
        "outer_fold": outer_fold,
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "auroc": float(roc_auc_score(y.iloc[test_index], probabilities)),
        "best_params": search.best_params_,
        "model_seed": seed,
        "inner_seed": inner_seed,
        "search_seed": search_seed,
    }
