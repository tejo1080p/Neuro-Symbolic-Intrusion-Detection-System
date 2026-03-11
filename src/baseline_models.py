"""Baseline model training utilities for Scenario 1."""

from __future__ import annotations

from typing import Any
from importlib import import_module

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def _load_optional_model(module_name: str, class_name: str) -> Any:
    try:
        module = import_module(module_name)
        return getattr(module, class_name)
    except Exception:
        return None


def _as_dense_if_needed(X: Any, model_name: str) -> Any:
    dense_models = {
        "gradient_boosting",
        "mlp_classifier",
    }
    if model_name in dense_models and sparse.issparse(X):
        return X.toarray()
    return X


def _positive_class_weight(y_train: pd.Series) -> float:
    counts = y_train.value_counts()
    negative_count = float(counts.get(0, 0.0))
    positive_count = float(counts.get(1, 0.0))
    if positive_count <= 0:
        return 1.0
    return max(negative_count / positive_count, 1.0)


def train_all_models(X_train: Any, y_train: pd.Series, random_state: int = 42) -> dict[str, Any]:
    """Train all configured baseline models and return fitted estimators."""
    XGBClassifier = _load_optional_model("xgboost", "XGBClassifier")
    LGBMClassifier = _load_optional_model("lightgbm", "LGBMClassifier")
    BalancedRandomForestClassifier = _load_optional_model(
        "imblearn.ensemble", "BalancedRandomForestClassifier"
    )

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "ridge_logistic": LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="lbfgs",
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "elasticnet_logistic": LogisticRegression(
            penalty="elasticnet",
            C=1.0,
            l1_ratio=0.5,
            solver="saga",
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "svm_linear": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            random_state=random_state,
        ),
    }

    positive_weight = _positive_class_weight(y_train)

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
        models["xgboost_weighted"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            n_jobs=-1,
            random_state=random_state,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )

    if BalancedRandomForestClassifier is not None:
        models["balanced_random_forest"] = BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            replacement=False,
            sampling_strategy="all",
            n_jobs=-1,
            random_state=random_state,
        )

    fitted_models: dict[str, Any] = {}
    for model_name, model in models.items():
        X_fit = _as_dense_if_needed(X_train, model_name)
        model.fit(X_fit, y_train)
        fitted_models[model_name] = model

    return fitted_models
