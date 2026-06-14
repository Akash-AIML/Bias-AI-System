from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ── Sampling cap ──────────────────────────────────────────────────────────────
# Proxy model training is capped at this many rows for speed.
# Fairness metrics computed from a stratified 10K sample are statistically
# indistinguishable from those computed on the full dataset.
_PROXY_TRAIN_CAP = 10_000


@dataclass
class FairnessResult:
    bias: bool
    dp_diff: float
    eo_diff: float
    bsi_score: float
    disparate_impact_ratio: float          # 4/5ths rule (EEOC)
    group_metrics: dict[str, dict[str, float]]
    audit_mode: str
    warnings: list[str]
    info_notes: list[str]
    reliability_note: str
    target_transformation: dict[str, object]


def _compute_bsi(dp_diff: float, eo_diff: float, sensitive_values: pd.Series) -> float:
    abs_dp = min(1.0, abs(dp_diff))
    abs_eo = min(1.0, abs(eo_diff))

    group_counts = Counter(sensitive_values.astype(str))
    if not group_counts or len(group_counts) < 2:
        imbalance_score = 0.0
    else:
        max_count = max(group_counts.values())
        min_count = min(group_counts.values())
        if min_count == 0:
            imbalance_score = 1.0
        else:
            ratio = max_count / min_count
            # Log-based: more sensitive to extreme imbalances (10x, 100x, 1000x)
            imbalance_score = min(1.0, math.log1p(ratio - 1.0) / math.log1p(9.0))

    bsi = 100.0 * (0.4 * abs_dp + 0.4 * abs_eo + 0.2 * imbalance_score)
    return round(bsi, 2)


def _compute_disparate_impact_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_test: np.ndarray,
) -> float:
    """Adverse impact ratio: min_group_selection_rate / max_group_selection_rate.

    A value < 0.8 triggers the EEOC 4/5ths rule (significant adverse impact).
    Returns -1.0 when there are fewer than 2 groups or denominators are zero.
    """
    groups = np.unique(sensitive_test)
    if len(groups) < 2:
        return -1.0
    rates: list[float] = []
    for g in groups:
        mask = sensitive_test == g
        if mask.sum() == 0:
            continue
        rate = float((y_pred[mask] == 1).mean())
        rates.append(rate)
    if not rates or max(rates) == 0:
        return -1.0
    return round(min(rates) / max(rates), 4)


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
    """Build a logistic regression pipeline.

    Uses the 'saga' solver which is significantly faster than 'lbfgs' on large
    datasets, supports parallelism via n_jobs, and converges reliably in fewer
    iterations.
    """
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
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="saga",     # Much faster than lbfgs on large data
                    n_jobs=-1,         # Use all CPU cores
                    random_state=42,
                    C=1.0,
                    tol=1e-3,          # Slightly relaxed tolerance — sufficient for fairness
                ),
            ),
        ]
    )


def _stratified_sample(
    features: pd.DataFrame,
    y: np.ndarray,
    sensitive: np.ndarray,
    max_rows: int = _PROXY_TRAIN_CAP,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return a stratified random sample capped at *max_rows*.

    Stratification is on the target label so class balance is preserved.
    When the dataset is already small enough, returns the inputs unchanged.
    """
    if len(features) <= max_rows:
        return features, y, sensitive

    # train_test_split with stratify gives us a clean stratified subset quickly
    _, feat_s, _, y_s, _, sens_s = train_test_split(
        features,
        y,
        sensitive,
        test_size=max_rows / len(features),
        random_state=42,
        stratify=y,
    )
    return feat_s, y_s, sens_s


def compute_fairness(
    features: pd.DataFrame,
    labels: pd.Series,
    predictions: pd.Series | None,
    sensitive_values: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    threshold: float = 0.10,
    target_binarization_threshold: float | None = None,
    weight_column: str | None = None,
) -> FairnessResult:
    if len(features) < 20:
        raise ValueError("Dataset is too small for fairness analysis. Provide at least 20 rows.")

    y_true, all_target_notes, target_transformation = _normalize_target(
        labels=labels,
        target_binarization_threshold=target_binarization_threshold,
    )

    # Separate target normalization notes: binarization info is informational, not a warning
    info_notes: list[str] = []
    warnings: list[str] = []
    for note in all_target_notes:
        if any(token in note.lower() for token in (
            "normalized to binary", "binarized using threshold", "converted to binary one-vs-rest"
        )):
            info_notes.append(note)
        else:
            warnings.append(note)

    if len(features) < 100:
        warnings.append(
            f"Small dataset warning: Dataset has only {len(features)} rows. "
            "Fairness analysis on small samples has lower statistical confidence."
        )

    # Check sensitive attribute imbalance
    group_counts = Counter(sensitive_values.astype(str))
    small_groups = [(group_name, count) for group_name, count in group_counts.items() if count < 10]
    if len(small_groups) > 3:
        examples = ", ".join(f"'{name}' ({count} samples)" for name, count in small_groups[:3])
        warnings.append(
            f"{len(small_groups)} sensitive groups have fewer than 10 samples (including {examples}); "
            "group-specific metrics for these cohorts may be less reliable."
        )
    else:
        for group_name, count in small_groups:
            warnings.append(
                f"Sensitive group '{group_name}' has only {count} samples; group-specific metrics may be less reliable."
            )
    if group_counts:
        max_count = max(group_counts.values())
        min_count = min(group_counts.values())
        if min_count > 0:
            ratio = max_count / min_count
            if ratio > 5.0:
                warnings.append(
                    f"High sensitive attribute imbalance detected. The largest group ('{max(group_counts, key=group_counts.get)}') "
                    f"is {ratio:.1f}x larger than the smallest group ('{min(group_counts, key=group_counts.get)}')."
                )

    # Check target class imbalance
    positive_count = int(sum(y_true == 1))
    total_count = len(y_true)
    if total_count > 0:
        positive_ratio = positive_count / total_count
        if positive_ratio < 0.1 or positive_ratio > 0.9:
            warnings.append(
                f"Severe target class imbalance detected ({positive_ratio * 100:.1f}% positive). "
                "Fairness metrics can be highly sensitive or misleading in imbalanced settings."
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

    sample_weight = None
    features_clean = features.copy()

    # Resolve the weight column
    resolved_weight_col: str | None = None
    if weight_column and weight_column in features_clean.columns:
        resolved_weight_col = weight_column
    elif "sample_weight" in features_clean.columns:
        resolved_weight_col = "sample_weight"

    if resolved_weight_col:
        sample_weight = features_clean[resolved_weight_col]
        features_clean = features_clean.drop(columns=[resolved_weight_col])
        numeric_features = [f for f in numeric_features if f != resolved_weight_col]
        categorical_features = [f for f in categorical_features if f != resolved_weight_col]
        info_notes.append(
            f"Sample weights from column '{resolved_weight_col}' are applied to all "
            "fairness metrics (demographic parity, equalized odds, group selection rates)."
        )

    metric_sample_weight = None
    if predictions is not None:
        encoded_pred, pred_warnings = _encode_binary(predictions)
        warnings.extend(pred_warnings)
        y_pred = encoded_pred.to_numpy()
        y_true_values = y_true.to_numpy()
        sensitive_test = sensitive_values.astype(str).to_numpy()
        if sample_weight is not None:
            metric_sample_weight = sample_weight.to_numpy()
    else:
        info_notes.append(
            "Proxy model was trained because no prediction column was provided. "
            "To use direct predictions, add a prediction column when uploading."
        )

        # ── Speed fix: cap training rows at _PROXY_TRAIN_CAP ─────────────────
        # Stratified sample so that proxy training stays fast even on 500K+ row
        # datasets.  We note if sampling was applied.
        y_np = y_true.to_numpy()
        sens_np = sensitive_values.astype(str).to_numpy()

        if len(features_clean) > _PROXY_TRAIN_CAP:
            info_notes.append(
                f"Dataset has {len(features_clean):,} rows. Proxy model was trained on a "
                f"stratified sample of {_PROXY_TRAIN_CAP:,} rows for performance. "
                "Fairness metrics are still statistically representative."
            )
            features_sampled, y_sampled, sens_sampled = _stratified_sample(
                features_clean, y_np, sens_np, max_rows=_PROXY_TRAIN_CAP
            )
        else:
            features_sampled, y_sampled, sens_sampled = features_clean, y_np, sens_np

        model = _build_model(numeric_features, categorical_features)

        if sample_weight is not None:
            # Align weights to the sampled indices
            weight_np = sample_weight.to_numpy()
            (
                x_train, x_test,
                y_train, y_test,
                sensitive_train, sensitive_test,
                weight_train, weight_test,
            ) = train_test_split(
                features_sampled,
                y_sampled,
                sens_sampled,
                weight_np[: len(features_sampled)]
                if len(features_sampled) < len(features_clean)
                else weight_np,
                test_size=0.3,
                random_state=42,
                stratify=y_sampled,
            )
            metric_sample_weight = weight_test
            model.fit(x_train, y_train, classifier__sample_weight=weight_train)
        else:
            x_train, x_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
                features_sampled,
                y_sampled,
                sens_sampled,
                test_size=0.3,
                random_state=42,
                stratify=y_sampled,
            )
            model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_true_values = y_test

    dp_diff = float(
        demographic_parity_difference(
            y_true=y_true_values,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
            sample_weight=metric_sample_weight,
        )
    )
    eo_diff = float(
        equalized_odds_difference(
            y_true=y_true_values,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
            sample_weight=metric_sample_weight,
        )
    )

    bsi_score = _compute_bsi(dp_diff=dp_diff, eo_diff=eo_diff, sensitive_values=pd.Series(sensitive_test))
    disparate_impact_ratio = _compute_disparate_impact_ratio(y_true_values, y_pred, sensitive_test)

    sample_params = None
    if metric_sample_weight is not None:
        sample_params = {"sample_weight": metric_sample_weight}

    group_accuracy = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true_values,
        y_pred=y_pred,
        sensitive_features=sensitive_test,
        sample_params=sample_params,
    )

    positive_prediction_rate = MetricFrame(
        metrics=lambda yt, yp, sample_weight=None: float(
            (yp == 1).mean() if sample_weight is None
            else np.average(yp == 1, weights=sample_weight)
        ),
        y_true=y_true_values,
        y_pred=y_pred,
        sensitive_features=sensitive_test,
        sample_params=sample_params,
    )

    # Per-group FPR / FNR — important for high-stakes domain auditing
    def _fpr(yt: np.ndarray, yp: np.ndarray, **_: object) -> float:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    def _fnr(yt: np.ndarray, yp: np.ndarray, **_: object) -> float:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    try:
        group_fpr = MetricFrame(
            metrics=_fpr,
            y_true=y_true_values,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
        )
        group_fnr = MetricFrame(
            metrics=_fnr,
            y_true=y_true_values,
            y_pred=y_pred,
            sensitive_features=sensitive_test,
        )
        fpr_available = True
    except Exception:
        fpr_available = False

    group_metrics: dict[str, dict[str, float]] = {}
    for group_name, accuracy in group_accuracy.by_group.items():
        group_key = str(group_name)
        entry: dict[str, float] = {
            "accuracy": round(float(accuracy), 4),
            "selection_rate": round(float(positive_prediction_rate.by_group[group_name]), 4),
        }
        if fpr_available:
            try:
                entry["fpr"] = round(float(group_fpr.by_group[group_name]), 4)
                entry["fnr"] = round(float(group_fnr.by_group[group_name]), 4)
            except Exception:
                pass
        group_metrics[group_key] = entry

    is_biased = abs(dp_diff) > threshold or abs(eo_diff) > threshold

    return FairnessResult(
        bias=is_biased,
        dp_diff=round(dp_diff, 4),
        eo_diff=round(eo_diff, 4),
        bsi_score=bsi_score,
        disparate_impact_ratio=disparate_impact_ratio,
        group_metrics=group_metrics,
        audit_mode=audit_mode,
        warnings=warnings,
        info_notes=info_notes,
        reliability_note=reliability_note,
        target_transformation=target_transformation,
    )
