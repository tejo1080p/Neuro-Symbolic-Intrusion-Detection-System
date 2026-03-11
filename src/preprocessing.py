"""Generic preprocessing utilities for tabular binary classification datasets.

Supports intrusion-detection datasets such as NSL-KDD/CIC-IDS as well as
other structured datasets with mixed categorical and numeric features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "default"
NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]

TARGET_CANDIDATES = ("attack_binary", "attack", "label", "class", "target", "y", "default")


def _decode_bytes_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].apply(
                lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
            )
    return df


def _read_nsl_kdd_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] == len(NSL_KDD_COLUMNS):
        df.columns = NSL_KDD_COLUMNS
    elif df.shape[1] == len(NSL_KDD_COLUMNS) - 1:
        df.columns = NSL_KDD_COLUMNS[:-1]
    return df


def _infer_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not infer target column. Expected one of: "
        f"{', '.join(TARGET_CANDIDATES)}. Found columns: {list(df.columns)}"
    )


def _coerce_binary_target(y: pd.Series) -> pd.Series:
    non_null = y.dropna()
    unique_values = set(non_null.unique().tolist())

    if unique_values.issubset({0, 1}):
        return y.astype(int)

    if y.dtype == bool:
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        if len(unique_values) == 2:
            sorted_values = sorted(unique_values)
            mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
            return y.map(mapping).astype(int)
        return (y > 0).astype(int)

    normalized = y.astype(str).str.strip().str.lower().str.rstrip(".")
    normalized_non_null = normalized[~normalized.isna()]
    normalized_unique = set(normalized_non_null.unique().tolist())

    if normalized_unique.issubset({"0", "1"}):
        return normalized.astype(int)

    for normal_token in ("normal", "benign"):
        if normal_token in normalized_unique:
            return (normalized != normal_token).astype(int)

    if len(normalized_unique) == 2:
        sorted_values = sorted(normalized_unique)
        mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
        return normalized.map(mapping).astype(int)

    raise ValueError(
        "Target column is not binary and could not be coerced automatically. "
        f"Unique classes found: {sorted(normalized_unique)}"
    )


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load dataset from Excel, CSV, ARFF, or NSL-KDD TXT."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".txt", ".data"}:
        return _read_nsl_kdd_txt(path)

    if suffix == ".arff":
        from scipy.io import arff

        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        return _decode_bytes_columns(df)

    raise ValueError(f"Unsupported file format '{suffix}' for path: {path}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of input dataframe for non-destructive downstream processing."""
    cleaned = df.copy()

    if "label" in cleaned.columns:
        cleaned["label"] = (
            cleaned["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.rstrip(".")
        )

    if "difficulty_level" in cleaned.columns:
        cleaned["difficulty_level"] = pd.to_numeric(cleaned["difficulty_level"], errors="coerce")

    return cleaned


def split_features(df: pd.DataFrame, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and a binary target."""
    effective_target = target_column or _infer_target_column(df)
    if effective_target not in df.columns:
        raise ValueError(f"Target column '{effective_target}' was not found in dataframe.")

    excluded = [effective_target, "difficulty_level"]
    feature_columns = [column for column in df.columns if column not in excluded]

    X = df[feature_columns].copy()
    y = _coerce_binary_target(df[effective_target].copy())

    print(f"[preprocessing] detected target column: {effective_target}")
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test partitions for model development."""
    effective_stratify = y if stratify is None else stratify
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=effective_stratify,
    )


def build_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer with one-hot encoding and numeric scaling."""
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_features = [column for column in X.columns if column not in categorical_features]

    print(f"[preprocessing] number of categorical features: {len(categorical_features)}")
    print(f"[preprocessing] number of numeric features: {len(numeric_features)}")

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        ]
    )

    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor
