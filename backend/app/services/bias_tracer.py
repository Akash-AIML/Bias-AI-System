from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

        if pd.api.types.is_numeric_dtype(series):
            corr = pd.Series(series).corr(sensitive_encoded)
            corr_value = 0.0 if corr is None or np.isnan(corr) else float(corr)
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=column, drop_first=False)
            if dummies.empty:
                corr_value = 0.0
            else:
                correlations = dummies.apply(lambda col: col.corr(sensitive_encoded))
                correlations = correlations.fillna(0.0)
                corr_value = float(correlations.abs().max())
                if correlations.abs().idxmax() is not None:
                    corr_value = float(correlations.loc[correlations.abs().idxmax()])

        direction = "positive" if corr_value >= 0 else "negative"
        proxy_scores.append(
            ProxyFeature(
                feature=str(column),
                correlation=round(abs(corr_value), 4),
                direction=direction,
            )
        )

    proxy_scores.sort(key=lambda item: item.correlation, reverse=True)
    return proxy_scores[:top_k]
