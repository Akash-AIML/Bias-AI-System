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

    temp_df = pd.DataFrame({"S": s_values, "Y": y_true})

    joint_counts = temp_df.groupby(["S", "Y"]).size().rename("n_sy")

    # ── Speed fix: vectorized weight computation instead of row-wise apply ──
    # Build a lookup DataFrame and merge — O(n_groups * n_classes) not O(n_rows)
    weight_df = joint_counts.reset_index()
    weight_df["n_s"] = weight_df["S"].map(s_counts)
    weight_df["n_y"] = weight_df["Y"].map(y_counts)
    weight_df["weight"] = (weight_df["n_s"] * weight_df["n_y"]) / (n * weight_df["n_sy"])
    weight_df.loc[weight_df["n_sy"] == 0, "weight"] = 1.0

    # Merge weights back to the full DataFrame via a join — pure pandas, no Python loop
    temp_merged = temp_df.merge(
        weight_df[["S", "Y", "weight"]], on=["S", "Y"], how="left"
    )
    temp_merged["weight"] = temp_merged["weight"].fillna(1.0)

    reweighted = df.copy()
    reweighted["sample_weight"] = temp_merged["weight"].values.astype(float)
    return reweighted
