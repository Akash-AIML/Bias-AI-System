from __future__ import annotations

import pandas as pd


def apply_group_reweighting(df: pd.DataFrame, sensitive_column: str) -> pd.DataFrame:
    counts = df[sensitive_column].value_counts(dropna=False)
    total = float(counts.sum())
    num_groups = float(len(counts))

    weights = counts.apply(lambda c: total / (num_groups * float(c)))
    reweighted = df.copy()
    reweighted["sample_weight"] = reweighted[sensitive_column].map(weights).astype(float)
    return reweighted
