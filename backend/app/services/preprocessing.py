from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd


@dataclass
class PreprocessedData:
    dataframe: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series
    sensitive_values: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    filled_df = df.copy()
    for column in filled_df.columns:
        if pd.api.types.is_numeric_dtype(filled_df[column]):
            filled_df[column] = filled_df[column].fillna(filled_df[column].median())
        else:
            mode = filled_df[column].mode(dropna=True)
            fallback_value = mode.iloc[0] if not mode.empty else "unknown"
            filled_df[column] = filled_df[column].fillna(fallback_value)
    return filled_df


def load_and_preprocess(file_bytes: bytes, target: str, sensitive: str) -> PreprocessedData:
    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as error:
        raise ValueError("Failed to parse CSV file") from error

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    if sensitive not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive}' not found in dataset")
    if target == sensitive:
        raise ValueError("Target and sensitive columns must be different")

    cleaned_df = _fill_missing_values(df)

    labels = cleaned_df[target].astype(str)
    sensitive_values = cleaned_df[sensitive].astype(str)
    features = cleaned_df.drop(columns=[target])

    if sensitive in features.columns:
        features = features.drop(columns=[sensitive])

    if features.empty:
        raise ValueError("Dataset must contain at least one feature column besides target and sensitive")

    numeric_features: list[str] = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features: list[str] = [
        column for column in features.columns if column not in numeric_features
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found after preprocessing")

    return PreprocessedData(
        dataframe=cleaned_df,
        features=features,
        labels=labels,
        sensitive_values=sensitive_values,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def dataset_profile(df: pd.DataFrame, target: str, sensitive: str) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": target,
        "sensitive": sensitive,
        "column_names": list(df.columns),
    }
