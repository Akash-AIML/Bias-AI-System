from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app.services.fairness import compute_fairness

# Cap each temporal half at this many rows for proxy model training speed.
# Temporal drift only needs a representative sample to detect trends.
_TEMPORAL_ROW_CAP = 5_000


@dataclass
class TemporalDriftResult:
    status: str
    time_column: str | None
    early_bsi: float | None
    late_bsi: float | None
    delta: float | None


def analyze_temporal_drift(
    dataframe: pd.DataFrame,
    time_column: str | None,
    target: str,
    sensitive: str,
    prediction_column: str | None,
    numeric_features: list[str],
    categorical_features: list[str],
    target_binarization_threshold: float | None = None,
    weight_column: str | None = None,
) -> TemporalDriftResult:
    if not time_column or time_column not in dataframe.columns:
        return TemporalDriftResult(
            status="not_available",
            time_column=None,
            early_bsi=None,
            late_bsi=None,
            delta=None,
        )

    parsed = pd.to_datetime(dataframe[time_column], errors="coerce")
    if parsed.notna().mean() < 0.6:
        return TemporalDriftResult(
            status="not_available",
            time_column=time_column,
            early_bsi=None,
            late_bsi=None,
            delta=None,
        )

    sorted_df = dataframe.copy()
    sorted_df["_parsed_time"] = parsed
    sorted_df = sorted_df.dropna(subset=["_parsed_time"]).sort_values("_parsed_time")

    if len(sorted_df) < 60:
        return TemporalDriftResult(
            status="insufficient_data",
            time_column=time_column,
            early_bsi=None,
            late_bsi=None,
            delta=None,
        )

    midpoint = len(sorted_df) // 2
    early_df = sorted_df.iloc[:midpoint]
    late_df = sorted_df.iloc[midpoint:]

    # Speed fix: cap each half for proxy training
    if len(early_df) > _TEMPORAL_ROW_CAP:
        early_df = early_df.sample(n=_TEMPORAL_ROW_CAP, random_state=42)
    if len(late_df) > _TEMPORAL_ROW_CAP:
        late_df = late_df.sample(n=_TEMPORAL_ROW_CAP, random_state=42)

    def _slice(df: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        labels = df[target]
        sensitive_values = df[sensitive].astype(str)
        predictions = df[prediction_column] if prediction_column else None
        drop_columns = [target, sensitive, "_parsed_time"]
        if time_column and time_column in df.columns:
            drop_columns.append(time_column)
        features = df.drop(columns=drop_columns)
        if prediction_column and prediction_column in features.columns:
            features = features.drop(columns=[prediction_column])
        return {
            "labels": labels,
            "sensitive": sensitive_values,
            "predictions": predictions,
            "features": features,
        }

    try:
        early_payload = _slice(early_df)
        late_payload = _slice(late_df)

        early_result = compute_fairness(
            features=early_payload["features"],
            labels=early_payload["labels"],
            predictions=early_payload["predictions"],
            sensitive_values=early_payload["sensitive"],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_binarization_threshold=target_binarization_threshold,
            weight_column=weight_column,
        )
        late_result = compute_fairness(
            features=late_payload["features"],
            labels=late_payload["labels"],
            predictions=late_payload["predictions"],
            sensitive_values=late_payload["sensitive"],
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_binarization_threshold=target_binarization_threshold,
            weight_column=weight_column,
        )
    except ValueError:
        return TemporalDriftResult(
            status="insufficient_data",
            time_column=time_column,
            early_bsi=None,
            late_bsi=None,
            delta=None,
        )

    delta = round((late_result.bsi_score - early_result.bsi_score), 2)
    if abs(delta) < 3:
        status = "stable"
    elif delta > 0:
        status = "worsening"
    else:
        status = "improving"

    return TemporalDriftResult(
        status=status,
        time_column=time_column,
        early_bsi=early_result.bsi_score,
        late_bsi=late_result.bsi_score,
        delta=delta,
    )
