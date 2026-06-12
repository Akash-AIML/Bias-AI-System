from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.fairness import compute_fairness
from app.services.reweighting import apply_group_reweighting


@dataclass
class SimulationResult:
    name: str
    bsi_score: float
    dp_diff: float
    eo_diff: float


def simulate_mitigation(
    dataframe: pd.DataFrame,
    target: str,
    sensitive: str,
    prediction_column: str | None,
    numeric_features: list[str],
    categorical_features: list[str],
    target_binarization_threshold: float | None = None,
    weight_column: str | None = None,
) -> list[SimulationResult]:
    baseline = _compute(
        dataframe,
        target,
        sensitive,
        prediction_column,
        numeric_features,
        categorical_features,
        target_binarization_threshold,
        weight_column,
    )
    simulations = [SimulationResult(name="baseline", **baseline)]

    try:
        reweighted = apply_group_reweighting(
            dataframe,
            sensitive_column=sensitive,
            target_column=target,
            target_binarization_threshold=target_binarization_threshold,
        )
        reweighted_result = _compute(
            reweighted,
            target,
            sensitive,
            prediction_column,
            numeric_features,
            categorical_features,
            target_binarization_threshold,
            weight_column,
        )
        simulations.append(SimulationResult(name="reweighting", **reweighted_result))
    except Exception:
        pass

    return simulations


def _compute(
    dataframe: pd.DataFrame,
    target: str,
    sensitive: str,
    prediction_column: str | None,
    numeric_features: list[str],
    categorical_features: list[str],
    target_binarization_threshold: float | None = None,
    weight_column: str | None = None,
) -> dict[str, float]:
    labels = dataframe[target]
    sensitive_values = dataframe[sensitive].astype(str)
    predictions = dataframe[prediction_column] if prediction_column else None
    features = dataframe.drop(columns=[target, sensitive])
    if prediction_column and prediction_column in features.columns:
        features = features.drop(columns=[prediction_column])

    result = compute_fairness(
        features=features,
        labels=labels,
        predictions=predictions,
        sensitive_values=sensitive_values,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_binarization_threshold=target_binarization_threshold,
        weight_column=weight_column,
    )
    return {
        "bsi_score": result.bsi_score,
        "dp_diff": result.dp_diff,
        "eo_diff": result.eo_diff,
    }
