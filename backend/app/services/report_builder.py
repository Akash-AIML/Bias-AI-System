from __future__ import annotations

from typing import Any


def build_audit_payload(
    audit_id: str,
    org_name: str | None,
    dataset_name: str | None,
    bsi_score: float,
    risk_tier: dict[str, str],
    proxy_features: list[dict[str, object]],
    temporal_drift: dict[str, object],
    text_bias: dict[str, object],
    legal_context: dict[str, str],
    audit_narrative: dict[str, object],
) -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "org_name": org_name,
        "dataset_name": dataset_name,
        "bsi_score": bsi_score,
        "risk_tier": risk_tier,
        "proxy_features": proxy_features,
        "temporal_drift": temporal_drift,
        "text_bias": text_bias,
        "legal_context": legal_context,
        "audit_narrative": audit_narrative,
    }
