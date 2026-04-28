from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskTier:
    level: str
    label: str
    action: str


def classify_risk(bsi_score: float) -> RiskTier:
    if bsi_score < 20:
        return RiskTier(level="green", label="Low Risk", action="No immediate action required.")
    if bsi_score < 40:
        return RiskTier(level="yellow", label="Moderate Risk", action="Monitoring and review recommended.")
    if bsi_score < 65:
        return RiskTier(level="orange", label="High Risk", action="Mitigation required before deployment.")
    return RiskTier(level="red", label="Critical Risk", action="Immediate halt recommended until bias is addressed.")
