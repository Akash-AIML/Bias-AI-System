from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


@dataclass
class FairnessResult:
    bias: bool
    dp_diff: float
    eo_diff: float
    group_metrics: dict[str, dict[str, float]]
    audit_mode: str
    warnings: list[str]
    reliability_note: str
def _encode_binary(series: pd.Series) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []
    numeric_values = pd.to_numeric(series, errors="coerce")

    if numeric_values.notna().mean() > 0.95:
        unique_values = sorted(set(numeric_values.dropna().unique().tolist()))
        if len(unique_values) > 2:
            warnings.append("Predictions were numeric; values were thresholded at 0.5.")
            return (numeric_values >= 0.5).astype(int), warnings
        return numeric_values.astype(int), warnings

    label_encoder = LabelEncoder()
    return pd.Series(label_encoder.fit_transform(series.astype(str))), warnings



def _build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)),
        ]
    )


def compute_fairness(
    features: pd.DataFrame,
    labels: pd.Series,
    predictions: pd.Series | None,
    sensitive_values: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    threshold: float = 0.10,
) -> FairnessResult:
    if len(features) < 20:
        raise ValueError("Dataset is too small for fairness analysis. Provide at least 20 rows.")

    warnings: list[str] = []
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    if numeric_labels.notna().mean() > 0.95:
        unique_labels = sorted(set(numeric_labels.dropna().unique().tolist()))
        if len(unique_labels) != 2:
            raise ValueError(
                "Target column must be binary for fairness analysis. "
                "Provide a binary label or upload model predictions to audit directly."
            )
        y_true = numeric_labels.astype(int)
    else:
        label_encoder = LabelEncoder()
        y_true = pd.Series(label_encoder.fit_transform(labels.astype(str)))

    if len(set(y_true)) < 2:
        raise ValueError("Target column must contain at least two classes.")
    
    class_counts = Counter(y_true)
    min_class_count = min(class_counts.values())
    if min_class_count < 5:
        raise ValueError(
            "This dataset cannot support reliable fairness analysis because class counts are too small. "
            f"The smallest class has only {min_class_count} samples. "
            "Upload a larger sample or choose another target."
        )

    audit_mode = "predictions" if predictions is not None else "proxy"
    reliability_note = (
        "Direct predictions audited. Results reflect the provided model outputs."
        if audit_mode == "predictions"
        else "Proxy model used because no predictions were provided. Results are approximate."
    )

    if predictions is not None:
        encoded_pred, pred_warnings = _encode_binary(predictions)
        warnings.extend(pred_warnings)
        y_pred = encoded_pred.to_numpy()
        y_true_values = y_true.to_numpy()
        sensitive_test = sensitive_values.astype(str).to_numpy()
    else:
        warnings.append("Proxy model was trained because no prediction column was provided.")
        model = _build_model(numeric_features, categorical_features)

        x_train, x_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            features,
            y_true.to_numpy(),
            sensitive_values.astype(str),
            test_size=0.3,
            random_state=42,
            stratify=y_true,
        )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_true_values = y_test

    dp_diff = float(
        demographic_parity_difference(
            y_true=y_true_values,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
        )
    )
    eo_diff = float(
        equalized_odds_difference(
            y_true=y_true_values,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
        )
    )

    group_accuracy = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true_values,
        y_pred=y_pred,
        sensitive_features=sensitive_test,
    )

    positive_prediction_rate = MetricFrame(
        metrics=lambda yt, yp: float((yp == 1).mean()),
        y_true=y_true_values,
        y_pred=y_pred,
        sensitive_features=sensitive_test,
    )

    group_metrics: dict[str, dict[str, float]] = {}
    for group_name, accuracy in group_accuracy.by_group.items():
        group_key = str(group_name)
        group_metrics[group_key] = {
            "accuracy": round(float(accuracy), 4),
            "selection_rate": round(float(positive_prediction_rate.by_group[group_name]), 4),
        }

    is_biased = abs(dp_diff) > threshold or abs(eo_diff) > threshold

    return FairnessResult(
        bias=is_biased,
        dp_diff=round(dp_diff, 4),
        eo_diff=round(eo_diff, 4),
        group_metrics=group_metrics,
        audit_mode=audit_mode,
        warnings=warnings,
        reliability_note=reliability_note,
    )
