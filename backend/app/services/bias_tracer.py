from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Correlation computation is capped at this many rows for speed.
# On 500K-row datasets, Pearson/point-biserial correlation computed on 20K
# rows is statistically indistinguishable from the full-dataset result.
_CORR_ROW_CAP = 20_000

# Skip one-hot expansion for categoricals with more unique values than this.
# ZIP codes, user-IDs, etc. would blow up memory; use LabelEncoder fallback.
_MAX_CATEGORICAL_CARDINALITY = 50


@dataclass
class ProxyFeature:
    feature: str
    correlation: float
    direction: str


def trace_bias_proxies(
    features: pd.DataFrame,
    sensitive_values: pd.Series,
    top_k: int = 8,
) -> list[ProxyFeature]:
    if features.empty:
        return []

    # ── Speed fix: cap rows for correlation computation ────────────────────
    if len(features) > _CORR_ROW_CAP:
        sample_idx = (
            pd.Series(range(len(features)))
            .sample(n=_CORR_ROW_CAP, random_state=42)
            .values
        )
        features = features.iloc[sample_idx].reset_index(drop=True)
        sensitive_values = sensitive_values.iloc[sample_idx].reset_index(drop=True)

    encoder = LabelEncoder()
    sensitive_encoded = pd.Series(
        encoder.fit_transform(sensitive_values.astype(str)),
        index=features.index,
    )

    proxy_scores: list[ProxyFeature] = []

    for column in features.columns:
        series = features[column]
        if series.nunique(dropna=True) <= 1:
            continue

        signed_corr: float

        if pd.api.types.is_numeric_dtype(series):
            corr = pd.Series(series).corr(sensitive_encoded)
            signed_corr = 0.0 if corr is None or np.isnan(corr) else float(corr)

        elif series.nunique(dropna=True) <= _MAX_CATEGORICAL_CARDINALITY:
            # One-hot encode and find the dummy column with highest |correlation|
            dummies = pd.get_dummies(series.astype(str), prefix=column, drop_first=False)
            if dummies.empty:
                signed_corr = 0.0
            else:
                correlations = dummies.apply(lambda col: col.corr(sensitive_encoded)).fillna(0.0)
                # Pick the signed value of the max-absolute-correlation column
                best_col = correlations.abs().idxmax()
                signed_corr = float(correlations[best_col])
        else:
            # High-cardinality categorical → LabelEncode for a fast scalar correlation
            le = LabelEncoder()
            encoded_col = pd.Series(le.fit_transform(series.astype(str)), index=series.index)
            corr = encoded_col.corr(sensitive_encoded)
            signed_corr = 0.0 if corr is None or np.isnan(corr) else float(corr)

        # Direction is derived from the signed value; abs is stored as the score
        direction = "positive" if signed_corr >= 0 else "negative"
        proxy_scores.append(
            ProxyFeature(
                feature=str(column),
                correlation=round(abs(signed_corr), 4),
                direction=direction,
            )
        )

    proxy_scores.sort(key=lambda item: item.correlation, reverse=True)
    return proxy_scores[:top_k]
