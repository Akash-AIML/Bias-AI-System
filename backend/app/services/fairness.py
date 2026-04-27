from __future__ import annotations

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
            ("classifier", LogisticRegression(max_iter=1200)),
        ]
    )


def compute_fairness(
    features: pd.DataFrame,
    labels: pd.Series,
    sensitive_values: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    threshold: float = 0.10,
) -> FairnessResult:
    if len(features) < 20:
        raise ValueError("Dataset is too small for fairness analysis. Provide at least 20 rows.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels.astype(str))

    if len(set(y)) < 2:
        raise ValueError("Target column must contain at least two classes.")

    model = _build_model(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        features,
        y,
        sensitive_values.astype(str),
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    dp_diff = float(
        demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
        )
    )
    eo_diff = float(
        equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
        )
    )

    group_accuracy = MetricFrame(
        metrics=accuracy_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test,
    )

    positive_prediction_rate = MetricFrame(
        metrics=lambda yt, yp: float((yp == 1).mean()),
        y_true=y_test,
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
    )
