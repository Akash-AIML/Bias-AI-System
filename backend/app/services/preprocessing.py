from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
import warnings

import pandas as pd


@dataclass
class PreprocessedData:
    dataframe: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series
    prediction_values: pd.Series | None
    sensitive_values: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]
    text_columns: list[str]
    time_column: str | None


def suggest_column_roles(file_bytes: bytes) -> dict[str, Any]:
    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as error:
        raise ValueError("Failed to parse CSV file") from error

    if df.empty or not list(df.columns):
        raise ValueError("Dataset has no usable columns")

    columns = [str(column) for column in df.columns]
    row_count = len(df)
    time_column = _detect_time_column(df, exclude_columns=set())

    prediction_column, prediction_score = _best_column(
        df=df,
        columns=[column for column in columns if column != time_column],
        scorer=lambda column: _score_prediction_column(df=df, column=column, row_count=row_count),
    )
    if prediction_score < 3.0:
        prediction_column = None

    excluded_for_target = {value for value in [time_column, prediction_column] if value}
    target_column, target_score = _best_column(
        df=df,
        columns=[column for column in columns if column not in excluded_for_target],
        scorer=lambda column: _score_target_column(df=df, column=column, row_count=row_count),
    )
    if target_score < 1.0:
        target_column = None

    excluded_for_sensitive = excluded_for_target | ({target_column} if target_column else set())
    sensitive_column, sensitive_score = _best_column(
        df=df,
        columns=[column for column in columns if column not in excluded_for_sensitive],
        scorer=lambda column: _score_sensitive_column(df=df, column=column, row_count=row_count),
    )
    if sensitive_score < 1.5:
        sensitive_column = None

    notes = [
        "Role mapping uses profile-based scoring over names, cardinality, and numeric patterns.",
    ]
    if time_column:
        notes.append(f"Detected time column: {time_column}.")
    if prediction_column:
        notes.append(f"Detected prediction-like column: {prediction_column}.")
    if not target_column:
        notes.append("No high-confidence target candidate was found. Please select target manually.")
    if not sensitive_column:
        notes.append("No high-confidence sensitive candidate was found. Please select sensitive manually.")

    return {
        "columns": columns,
        "target": target_column,
        "sensitive": sensitive_column,
        "prediction_column": prediction_column,
        "time_column": time_column,
        "method": "heuristic_scoring",
        "notes": notes,
    }


def _best_column(
    df: pd.DataFrame,
    columns: list[str],
    scorer: Any,
) -> tuple[str | None, float]:
    if not columns:
        return None, float("-inf")
    scored = [(column, float(scorer(column))) for column in columns]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0]


def _score_target_column(df: pd.DataFrame, column: str, row_count: int) -> float:
    lowered = column.lower()
    series = df[column]
    unique_count = int(series.dropna().nunique())
    unique_ratio = unique_count / max(row_count, 1)
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()

    score = 0.0
    if any(token in lowered for token in ("target", "label", "outcome", "approved", "decision", "status", "class", "result")):
        score += 4.0
    if unique_count == 2:
        score += 3.0
    elif 2 < unique_count <= 20:
        score += 2.0
    elif numeric_ratio > 0.95 and unique_count > 20:
        score += 1.5
    if unique_count < 2:
        score -= 5.0
    if any(token in lowered for token in ("id", "uuid", "index")):
        score -= 4.0
    if unique_ratio > 0.98 and unique_count > 20:
        score -= 2.5
    if any(token in lowered for token in ("time", "date", "timestamp")):
        score -= 2.0
    if any(token in lowered for token in ("pred", "prob", "score", "logit")):
        score -= 1.5
    return score


def _score_sensitive_column(df: pd.DataFrame, column: str, row_count: int) -> float:
    lowered = column.lower()
    series = df[column]
    unique_count = int(series.dropna().nunique())
    unique_ratio = unique_count / max(row_count, 1)
    numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()

    score = 0.0
    if any(
        token in lowered
        for token in (
            "gender",
            "sex",
            "race",
            "ethnic",
            "age",
            "disab",
            "relig",
            "national",
            "marital",
            "region",
            "country",
            "state",
            "zip",
            "caste",
            "language",
        )
    ):
        score += 5.0
    if unique_count == 2:
        score += 3.0
    elif 2 < unique_count <= 15:
        score += 2.0
    elif 15 < unique_count <= 50:
        score += 0.5
    if unique_count < 2:
        score -= 5.0
    if unique_ratio > 0.8:
        score -= 2.0
    if any(token in lowered for token in ("id", "uuid", "index")):
        score -= 3.0
    if any(token in lowered for token in ("target", "label", "outcome", "pred", "prob", "score", "logit")):
        score -= 2.0
    if any(token in lowered for token in ("time", "date", "timestamp")):
        score -= 2.0
    if numeric_ratio > 0.95 and unique_count > 30:
        score -= 1.5
    return score


def _score_prediction_column(df: pd.DataFrame, column: str, row_count: int) -> float:
    lowered = column.lower()
    series = df[column]
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean()
    unique_count = int(series.dropna().nunique())
    unique_ratio = unique_count / max(row_count, 1)

    score = 0.0
    if any(token in lowered for token in ("pred", "prediction", "prob", "score", "logit", "output")):
        score += 5.0
    if numeric_ratio > 0.95:
        min_val = float(numeric.min()) if numeric.notna().any() else 0.0
        max_val = float(numeric.max()) if numeric.notna().any() else 0.0
        if 0.0 <= min_val and max_val <= 1.0:
            score += 2.5
        if unique_count == 2:
            score += 2.0
        elif unique_count > 5:
            score += 1.0
    if any(token in lowered for token in ("target", "label", "outcome", "sensitive", "gender", "race")):
        score -= 2.0
    if any(token in lowered for token in ("id", "uuid", "index")):
        score -= 3.0
    if unique_ratio > 0.98 and unique_count > 20:
        score -= 1.5
    return score


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    filled_df = df.copy()
    for column in filled_df.columns:
        if pd.api.types.is_numeric_dtype(filled_df[column]):
            median_val = filled_df[column].median()
            if pd.isna(median_val):
                median_val = 0.0
            filled_df[column] = filled_df[column].fillna(median_val)
        else:
            mode = filled_df[column].mode(dropna=True)
            fallback_value = mode.iloc[0] if not mode.empty else "unknown"
            filled_df[column] = filled_df[column].fillna(fallback_value)
    return filled_df


def load_and_preprocess(
    file_bytes: bytes,
    target: str,
    sensitive: str,
    prediction_column: str | None = None,
    time_column: str | None = None,
    text_columns: list[str] | None = None,
) -> PreprocessedData:
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
    if prediction_column and prediction_column not in df.columns:
        raise ValueError(f"Prediction column '{prediction_column}' not found in dataset")
    if prediction_column and prediction_column in {target, sensitive}:
        raise ValueError("Prediction column must be different from target and sensitive columns")

    cleaned_df = _fill_missing_values(df)

    labels = cleaned_df[target]
    sensitive_values = cleaned_df[sensitive].astype(str)
    prediction_values = cleaned_df[prediction_column] if prediction_column else None
    features = cleaned_df.drop(columns=[target])

    if sensitive in features.columns:
        features = features.drop(columns=[sensitive])
    if prediction_column and prediction_column in features.columns:
        features = features.drop(columns=[prediction_column])

    if features.empty:
        raise ValueError("Dataset must contain at least one feature column besides target and sensitive")

    numeric_features: list[str] = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features: list[str] = [
        column for column in features.columns if column not in numeric_features
    ]

    detected_text_columns = _detect_text_columns(
        cleaned_df,
        exclude_columns={target, sensitive, prediction_column} if prediction_column else {target, sensitive},
    )
    if text_columns is not None:
        resolved_text_columns = [
            column
            for column in text_columns
            if column in cleaned_df.columns and column not in {target, sensitive, prediction_column}
        ]
    else:
        resolved_text_columns = detected_text_columns
    detected_time_column = _detect_time_column(
        cleaned_df,
        exclude_columns={target, sensitive, prediction_column} if prediction_column else {target, sensitive},
    )
    if time_column and time_column in cleaned_df.columns:
        resolved_time_column = time_column
    else:
        resolved_time_column = detected_time_column

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found after preprocessing")

    return PreprocessedData(
        dataframe=cleaned_df,
        features=features,
        labels=labels,
        prediction_values=prediction_values,
        sensitive_values=sensitive_values,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        text_columns=resolved_text_columns,
        time_column=resolved_time_column,
    )


def dataset_profile(
    df: pd.DataFrame,
    target: str,
    sensitive: str,
    prediction_column: str | None = None,
) -> dict[str, Any]:
    exclude_columns = {target, sensitive}
    if prediction_column:
        exclude_columns.add(prediction_column)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": target,
        "sensitive": sensitive,
        "prediction_column": prediction_column,
        "has_predictions": prediction_column is not None,
        "column_names": list(df.columns),
        "detected_text_columns": _detect_text_columns(df, exclude_columns=exclude_columns),
        "detected_time_column": _detect_time_column(df, exclude_columns=exclude_columns),
    }


def _detect_text_columns(df: pd.DataFrame, exclude_columns: set[str]) -> list[str]:
    text_columns: list[str] = []
    for column in df.columns:
        if column in exclude_columns:
            continue
        if not pd.api.types.is_object_dtype(df[column]):
            continue
        series = df[column].dropna().astype(str)
        if series.empty:
            continue
        unique_ratio = series.nunique() / max(len(series), 1)
        avg_tokens = series.str.split().map(len).mean()
        if unique_ratio >= 0.5 and avg_tokens >= 5:
            text_columns.append(column)
    return text_columns


def _detect_time_column(df: pd.DataFrame, exclude_columns: set[str]) -> str | None:
    def _parse_datetime(series: pd.Series) -> pd.Series:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Could not infer format, so each element will be parsed individually, "
                    "falling back to `dateutil`."
                ),
                category=UserWarning,
            )
            return pd.to_datetime(series, errors="coerce")

    candidates = []
    for column in df.columns:
        if column in exclude_columns:
            continue
        lowered = column.lower()
        if any(token in lowered for token in ("date", "time", "year")):
            candidates.append(column)

    for column in candidates:
        parsed = _parse_datetime(df[column])
        if parsed.notna().mean() >= 0.6:
            return column

    for column in df.columns:
        if column in exclude_columns:
            continue
        parsed = _parse_datetime(df[column])
        if parsed.notna().mean() >= 0.8:
            return column
    return None
