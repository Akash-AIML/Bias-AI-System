from __future__ import annotations

import pandas as pd


def apply_group_reweighting(
    df: pd.DataFrame,
    sensitive_column: str,
    target_column: str,
    target_binarization_threshold: float | None = None,
) -> pd.DataFrame:
    from app.services.fairness import _normalize_target

    y_true, _, _ = _normalize_target(df[target_column], target_binarization_threshold)

    n = float(len(df))
    if n == 0:
        reweighted = df.copy()
        reweighted["sample_weight"] = 1.0
        return reweighted

    s_values = df[sensitive_column].astype(str)

    s_counts = s_values.value_counts(dropna=False)
    y_counts = y_true.value_counts(dropna=False)

    temp_df = pd.DataFrame({
        "S": s_values,
        "Y": y_true
    })

    joint_counts = temp_df.groupby(["S", "Y"]).size().to_dict()

    weights = {}
    for (s_val, y_val) in joint_counts.keys():
        n_s = float(s_counts.get(s_val, 0))
        n_y = float(y_counts.get(y_val, 0))
        n_sy = float(joint_counts.get((s_val, y_val), 0))

        if n_sy > 0:
            weights[(s_val, y_val)] = (n_s * n_y) / (n * n_sy)
        else:
            weights[(s_val, y_val)] = 1.0

    reweighted = df.copy()
    reweighted["sample_weight"] = temp_df.apply(
        lambda row: weights.get((row["S"], row["Y"]), 1.0), axis=1
    ).astype(float)

    return reweighted
