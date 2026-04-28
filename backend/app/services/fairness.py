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
    bsi_score: float
    group_metrics: dict[str, dict[str, float]]
    audit_mode: str
    warnings: list[str]
    reliability_note: str
    target_transformation: dict[str, object]


def _compute_bsi(dp_diff: float, eo_diff: float, sensitive_values: pd.Series) -> float:
    abs_dp = min(1.0, abs(dp_diff))
    abs_eo = min(1.0, abs(eo_diff))

    group_counts = Counter(sensitive_values.astype(str))
    if not group_counts:
        imbalance_score = 0.0
    else:
        max_count = max(group_counts.values())
        min_count = min(group_counts.values())
        if min_count == 0:
            imbalance_score = 1.0
        else:
            ratio = max_count / min_count
            imbalance_score = min(1.0, (ratio - 1.0) / 4.0)

    bsi = 100.0 * (0.4 * abs_dp + 0.4 * abs_eo + 0.2 * imbalance_score)
    return round(bsi, 2)


def _normalize_binary_values(values: pd.Series) -> tuple[pd.Series, bool]:
    unique_values = sorted(set(values.dropna().unique().tolist()))
    if set(unique_values) == {0, 1}:
        return values.astype(int), False
    if set(unique_values) == {-1, 1}:
        return values.map({-1: 0, 1: 1}).astype(int), True

    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return values.map(mapping).astype(int), True


def _is_continuous_numeric(unique_count: int, sample_count: int) -> bool:
    if sample_count == 0:
        return False
    unique_ratio = unique_count / sample_count
    return unique_count > 10 and unique_ratio >= 0.10


def _normalize_target(
    labels: pd.Series, target_binarization_threshold: float | None
) -> tuple[pd.Series, list[str], dict[str, object]]:
    warnings: list[str] = []
    numeric_labels = pd.to_numeric(labels, errors="coerce")
    numeric_ratio = numeric_labels.notna().mean()

    if numeric_ratio > 0.95:
        numeric_values = numeric_labels.dropna()
        unique_labels = sorted(set(numeric_values.unique().tolist()))
        unique_count = len(unique_labels)

        if unique_count < 2:
            raise ValueError("Target column must contain at least two classes.")

        if unique_count == 2:
            y_true, did_normalize = _normalize_binary_values(numeric_labels)
            if did_normalize:
                warnings.append("Target values were normalized to binary values 0/1.")
            return y_true, warnings, {
                "source_type": "binary_numeric",
                "method": "binary_normalization",
                "mapping": {
                    str(unique_labels[0]): 0,
                    str(unique_labels[1]): 1,
                },
                "threshold": None,
            }

        if _is_continuous_numeric(unique_count=unique_count, sample_count=len(numeric_values)):
            threshold = (
                float(target_binarization_threshold)
                if target_binarization_threshold is not None
                else float(numeric_values.median())
            )
            y_true = (numeric_labels >= threshold).astype(int)
            warnings.append(
                f"Target was binarized using threshold {threshold:.6g} (>= threshold -> 1, below -> 0)."
            )
            return y_true, warnings, {
                "source_type": "continuous_numeric",
                "method": "threshold_binarization",
                "mapping": {
                    f"< {threshold:.6g}": 0,
                    f">= {threshold:.6g}": 1,
                },
                "threshold": threshold,
            }

        counts = numeric_values.value_counts()
        positive_label = counts.index[0]
        y_true = (numeric_labels == positive_label).astype(int)
        warnings.append(
            "Target had more than two classes; converted to binary one-vs-rest using the most frequent class as positive."
        )
        return y_true, warnings, {
            "source_type": "multiclass_numeric",
            "method": "one_vs_rest_majority_positive",
            "mapping": {
                str(positive_label): 1,
                "__all_other_classes__": 0,
            },
            "threshold": None,
        }

    normalized_labels = labels.astype(str)
    unique_labels = sorted(set(normalized_labels.unique().tolist()))
    unique_count = len(unique_labels)

    if unique_count < 2:
        raise ValueError("Target column must contain at least two classes.")

    if unique_count == 2:
        mapping = {
            unique_labels[0]: 0,
            unique_labels[1]: 1,
        }
        y_true = normalized_labels.map(mapping).astype(int)
        return y_true, warnings, {
            "source_type": "binary_categorical",
            "method": "binary_encoding",
            "mapping": {
                unique_labels[0]: 0,
                unique_labels[1]: 1,
            },
            "threshold": None,
        }

    counts = normalized_labels.value_counts()
    positive_label = str(counts.index[0])
    y_true = (normalized_labels == positive_label).astype(int)
    warnings.append(
        "Target had more than two classes; converted to binary one-vs-rest using the most frequent class as positive."
    )
    return y_true, warnings, {
        "source_type": "multiclass_categorical",
        "method": "one_vs_rest_majority_positive",
        "mapping": {
            positive_label: 1,
            "__all_other_classes__": 0,
        },
        "threshold": None,
    }


def _encode_binary(series: pd.Series) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []
    numeric_values = pd.to_numeric(series, errors="coerce")

    if numeric_values.notna().mean() > 0.95:
        unique_values = sorted(set(numeric_values.dropna().unique().tolist()))
        if len(unique_values) > 2:
            warnings.append("Predictions were numeric; values were thresholded at 0.5.")
            return (numeric_values >= 0.5).astype(int), warnings
        if len(unique_values) != 2:
            raise ValueError("Prediction column must be binary for fairness analysis.")

        normalized, did_normalize = _normalize_binary_values(numeric_values)
        if did_normalize:
            warnings.append("Predictions were normalized to binary values 0/1.")
        return normalized, warnings

    label_encoder = LabelEncoder()
    encoded = pd.Series(label_encoder.fit_transform(series.astype(str)))
    if len(label_encoder.classes_) != 2:
        raise ValueError("Prediction column must be binary for fairness analysis.")
    return encoded, warnings



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
    target_binarization_threshold: float | None = None,
) -> FairnessResult:
    if len(features) < 20:
        raise ValueError("Dataset is too small for fairness analysis. Provide at least 20 rows.")

    y_true, warnings, target_transformation = _normalize_target(
        labels=labels,
        target_binarization_threshold=target_binarization_threshold,
    )

    if len(set(y_true)) < 2:
        raise ValueError("Target column must contain at least two classes.")
    
    class_counts = Counter(y_true)
    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        raise ValueError(
            "Target binarization produced a class with fewer than 2 samples. "
            "Choose a different target or provide a target threshold."
        )
    if min_class_count < 5:
        warnings.append(
            f"Minority class has only {min_class_count} samples; fairness metrics may be less reliable."
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

    bsi_score = _compute_bsi(dp_diff=dp_diff, eo_diff=eo_diff, sensitive_values=pd.Series(sensitive_test))

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
        bsi_score=bsi_score,
        group_metrics=group_metrics,
        audit_mode=audit_mode,
        warnings=warnings,
        reliability_note=reliability_note,
        target_transformation=target_transformation,
    )
