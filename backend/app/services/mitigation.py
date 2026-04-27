from __future__ import annotations


def suggest_mitigations(dp_diff: float, eo_diff: float, bias_detected: bool) -> list[str]:
    suggestions: list[str] = []

    if not bias_detected:
        return ["No severe bias detected. Continue monitoring with periodic fairness checks."]

    if abs(dp_diff) > 0.10:
        suggestions.append(
            "Apply reweighting by sensitive group to reduce demographic parity disparity."
        )
        suggestions.append(
            "Adjust decision thresholds per group after legal and policy review."
        )

    if abs(eo_diff) > 0.10:
        suggestions.append(
            "Use post-processing equalized odds optimization to align error rates across groups."
        )

    if abs(dp_diff) > 0.20 or abs(eo_diff) > 0.20:
        suggestions.append(
            "Resample underrepresented groups and retrain with fairness constraints."
        )

    if not suggestions:
        suggestions.append("Review feature engineering choices and collect more balanced data.")

    return suggestions
