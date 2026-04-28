from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


class LLMService:
    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL", "llama3-8b-8192")
        self.client = None

        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def infer_intent(self, query: str, dataset_context: dict[str, Any]) -> dict[str, str]:
        if not query.strip():
            return {"intent": "bias_detection", "domain": "general"}

        if not self.client:
            return self._fallback_intent(query)

        prompt = (
            "You are an intent classifier for an AI fairness audit API. "
            "Return strict JSON with keys intent and domain. "
            "Intent must be one of: bias_detection, mitigation. "
            f"Query: {query}\n"
            f"Dataset context: {json.dumps(dataset_context)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            intent = payload.get("intent", "bias_detection")
            domain = payload.get("domain", "general")
            if intent not in {"bias_detection", "mitigation"}:
                intent = "bias_detection"
            return {"intent": intent, "domain": str(domain)}
        except Exception:
            return self._fallback_intent(query)

    def generate_explanation(
        self,
        dp_diff: float,
        eo_diff: float,
        bsi_score: float,
        suggestions: list[str],
        group_metrics: dict[str, dict[str, float]],
        audit_mode: str = "proxy",
        reliability_note: str = "",
        target_transformation: dict[str, object] | None = None,
        warnings: list[str] | None = None,
    ) -> dict[str, str]:
        baseline = self._deterministic_explanation(
            dp_diff=dp_diff,
            eo_diff=eo_diff,
            bsi_score=bsi_score,
            suggestions=suggestions,
            group_metrics=group_metrics,
            audit_mode=audit_mode,
            reliability_note=reliability_note,
            target_transformation=target_transformation or {},
            warnings=warnings or [],
        )

        if not self.client:
            return baseline

        prompt = (
            "Generate a concise fairness explanation for non-technical users. "
            "Return strict JSON with keys: explanation, summary, report_text. "
            "Mention demographic parity and equalized odds in plain language. "
            "Do not claim disadvantage if both parity gaps are near zero. "
            f"dp_diff={dp_diff}, eo_diff={eo_diff}, bsi_score={bsi_score}, "
            f"suggestions={json.dumps(suggestions)}, "
            f"group_metrics={json.dumps(group_metrics)}, "
            f"audit_mode={audit_mode}, "
            f"reliability_note={reliability_note}, "
            f"target_transformation={json.dumps(target_transformation or {})}, "
            f"warnings={json.dumps(warnings or [])}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return {
                "explanation": str(payload.get("explanation", baseline["explanation"])),
                "summary": self._sanitize_summary(
                    summary=str(payload.get("summary", baseline["summary"])),
                    dp_diff=dp_diff,
                    eo_diff=eo_diff,
                ),
                "report_text": str(payload.get("report_text", baseline["report_text"])),
            }
        except Exception:
            return baseline

    def classify_legal_context(self, query: str, dataset_context: dict[str, Any]) -> dict[str, str]:
        if not self.client:
            return self._fallback_legal_context(query, dataset_context)

        prompt = (
            "Classify the real-world domain (hiring, credit, housing, healthcare, or general) "
            "and provide the most relevant legal framework for bias auditing. "
            "Return strict JSON with keys: domain, framework, notes. "
            f"Query: {query}\nContext: {json.dumps(dataset_context)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return {
                "domain": str(payload.get("domain", "general")),
                "framework": str(payload.get("framework", "General fairness standards")),
                "notes": str(payload.get("notes", "")),
            }
        except Exception:
            return self._fallback_legal_context(query, dataset_context)

    def generate_counterfactuals(
        self,
        group_metrics: dict[str, dict[str, float]],
        domain: str,
    ) -> list[str]:
        if not self.client:
            return self._fallback_counterfactuals(group_metrics, domain)

        prompt = (
            "Generate counterfactual statements for disadvantaged groups based on selection rates. "
            "Return strict JSON with key counterfactuals (array of strings). "
            f"Domain: {domain}\nGroup metrics: {json.dumps(group_metrics)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            counterfactuals = payload.get("counterfactuals", [])
            return [str(item) for item in counterfactuals if str(item).strip()]
        except Exception:
            return self._fallback_counterfactuals(group_metrics, domain)

    def interpret_temporal_drift(self, temporal_drift: dict[str, Any], domain: str) -> str:
        if not self.client:
            return self._fallback_temporal_interpretation(temporal_drift, domain)

        prompt = (
            "Explain the meaning of the bias drift trend for a compliance audience. "
            "Return strict JSON with key interpretation. "
            f"Domain: {domain}\nDrift: {json.dumps(temporal_drift)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return str(payload.get("interpretation", ""))
        except Exception:
            return self._fallback_temporal_interpretation(temporal_drift, domain)

    def generate_audit_narrative(
        self,
        legal_context: dict[str, str],
        bsi_score: float,
        risk_tier: dict[str, str],
        proxy_features: list[dict[str, object]],
        temporal_drift: dict[str, object],
        text_bias: dict[str, object],
        suggestions: list[str],
        counterfactuals: list[str],
    ) -> dict[str, object]:
        if not self.client:
            return self._fallback_audit_narrative(
                legal_context,
                bsi_score,
                risk_tier,
                proxy_features,
                temporal_drift,
                text_bias,
                suggestions,
                counterfactuals,
            )

        prompt = (
            "Write a structured AI accountability audit narrative. "
            "Return strict JSON with keys: executive_summary, findings, risk_assessment, "
            "legal_exposure, recommended_actions. "
            f"Legal context: {json.dumps(legal_context)}\n"
            f"BSI score: {bsi_score}\nRisk tier: {json.dumps(risk_tier)}\n"
            f"Proxy features: {json.dumps(proxy_features)}\n"
            f"Temporal drift: {json.dumps(temporal_drift)}\n"
            f"Text bias: {json.dumps(text_bias)}\n"
            f"Suggestions: {json.dumps(suggestions)}\n"
            f"Counterfactuals: {json.dumps(counterfactuals)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return {
                "executive_summary": str(payload.get("executive_summary", "")),
                "findings": [str(item) for item in payload.get("findings", [])],
                "risk_assessment": str(payload.get("risk_assessment", "")),
                "legal_exposure": str(payload.get("legal_exposure", "")),
                "recommended_actions": [str(item) for item in payload.get("recommended_actions", [])],
            }
        except Exception:
            return self._fallback_audit_narrative(
                legal_context,
                bsi_score,
                risk_tier,
                proxy_features,
                temporal_drift,
                text_bias,
                suggestions,
                counterfactuals,
            )

    @staticmethod
    def _fallback_intent(query: str) -> dict[str, str]:
        normalized = query.lower()
        if "mitig" in normalized or "fix" in normalized or "improv" in normalized:
            return {"intent": "mitigation", "domain": "general"}
        if "hiring" in normalized:
            return {"intent": "bias_detection", "domain": "hiring"}
        if "credit" in normalized:
            return {"intent": "bias_detection", "domain": "credit"}
        return {"intent": "bias_detection", "domain": "general"}

    @staticmethod
    def _sanitize_summary(summary: str, dp_diff: float, eo_diff: float) -> str:
        no_gap = abs(dp_diff) < 0.01 and abs(eo_diff) < 0.01
        lowered = summary.lower()
        if no_gap and any(token in lowered for token in ("less likely", "disadvantage", "unfair", "bias detected")):
            return "No material group disparity was detected in this run."
        return summary

    @staticmethod
    def _deterministic_explanation(
        dp_diff: float,
        eo_diff: float,
        bsi_score: float,
        suggestions: list[str],
        group_metrics: dict[str, dict[str, float]],
        audit_mode: str,
        reliability_note: str,
        target_transformation: dict[str, object],
        warnings: list[str],
    ) -> dict[str, str]:
        dp_percent = round(abs(dp_diff) * 100, 1)
        eo_percent = round(abs(eo_diff) * 100, 1)
        max_gap = max(abs(dp_diff), abs(eo_diff))
        if max_gap < 0.01:
            summary = "No material group disparity was detected in this run."
        elif max_gap <= 0.10:
            summary = "Small group differences were observed and should be monitored."
        else:
            summary = "Meaningful group disparity was detected and should be mitigated."

        group_gap_text = "Selection-rate spread across groups was unavailable."
        if len(group_metrics) >= 2:
            sorted_groups = sorted(
                group_metrics.items(),
                key=lambda item: item[1].get("selection_rate", 0.0),
            )
            low_group, low_metrics = sorted_groups[0]
            high_group, high_metrics = sorted_groups[-1]
            spread = round((high_metrics.get("selection_rate", 0.0) - low_metrics.get("selection_rate", 0.0)) * 100, 1)
            group_gap_text = (
                f"Selection-rate spread is {spread}% "
                f"({low_group}: {round(low_metrics.get('selection_rate', 0.0) * 100, 1)}%, "
                f"{high_group}: {round(high_metrics.get('selection_rate', 0.0) * 100, 1)}%)."
            )

        transformation_method = str(target_transformation.get("method", "none"))
        transformation_text = f"Target handling: {transformation_method}."

        explanation = (
            f"Demographic parity differs by {dp_percent}% and equalized odds differs by {eo_percent}% between groups. "
            f"Bias Severity Index is {round(bsi_score, 2)}. {group_gap_text} {transformation_text}"
        )

        mode_text = "Direct model predictions were audited." if audit_mode == "predictions" else "A proxy model was used because predictions were not provided."
        warning_text = ""
        if warnings:
            warning_text = f" Notes: {warnings[0]}"

        report_text = (
            f"{mode_text} {reliability_note} Recommended actions: {'; '.join(suggestions)}.{warning_text}"
        )
        return {
            "explanation": explanation,
            "summary": summary,
            "report_text": report_text,
        }

    @staticmethod
    def _fallback_legal_context(query: str, dataset_context: dict[str, Any]) -> dict[str, str]:
        combined = " ".join(
            [query, " ".join(dataset_context.get("column_names", []))]
        ).lower()
        if any(token in combined for token in ("hire", "employee", "job", "recruit")):
            return {
                "domain": "hiring",
                "framework": "EEOC / Title VII (4/5ths rule)",
                "notes": "Adverse impact analysis applies to hiring outcomes.",
            }
        if any(token in combined for token in ("loan", "credit", "fico", "lending")):
            return {
                "domain": "credit",
                "framework": "Fair Credit Reporting Act (FCRA)",
                "notes": "Disparate impact in credit decisions is regulated.",
            }
        if any(token in combined for token in ("rent", "housing", "mortgage", "tenant")):
            return {
                "domain": "housing",
                "framework": "Fair Housing Act (FHA)",
                "notes": "Protected classes apply to housing decisions.",
            }
        if any(token in combined for token in ("health", "medical", "clinic", "patient")):
            return {
                "domain": "healthcare",
                "framework": "ACA non-discrimination provisions",
                "notes": "Protected health access is subject to anti-discrimination rules.",
            }
        return {
            "domain": "general",
            "framework": "General fairness standards",
            "notes": "No specific domain detected; apply general fairness review.",
        }

    @staticmethod
    def _fallback_counterfactuals(
        group_metrics: dict[str, dict[str, float]],
        domain: str,
    ) -> list[str]:
        if len(group_metrics) < 2:
            return ["Insufficient groups for counterfactual comparison."]
        sorted_groups = sorted(
            group_metrics.items(),
            key=lambda item: item[1].get("selection_rate", 0),
        )
        low_group, low_metrics = sorted_groups[0]
        high_group, high_metrics = sorted_groups[-1]
        gap = round((high_metrics["selection_rate"] - low_metrics["selection_rate"]) * 100, 1)
        return [
            (
                f"A member of the {low_group} group is about {gap}% less likely to receive a favorable "
                f"outcome than a comparable member of the {high_group} group in the {domain} context."
            )
        ]

    @staticmethod
    def _fallback_temporal_interpretation(temporal_drift: dict[str, Any], domain: str) -> str:
        status = temporal_drift.get("status", "not_available")
        if status == "worsening":
            return f"Bias severity is increasing over time in the {domain} domain, suggesting risk escalation."
        if status == "improving":
            return f"Bias severity is improving over time in the {domain} domain."
        if status == "stable":
            return f"Bias severity appears stable over time in the {domain} domain."
        return "Temporal drift analysis is unavailable for this dataset."

    @staticmethod
    def _fallback_audit_narrative(
        legal_context: dict[str, str],
        bsi_score: float,
        risk_tier: dict[str, str],
        proxy_features: list[dict[str, object]],
        temporal_drift: dict[str, object],
        text_bias: dict[str, object],
        suggestions: list[str],
        counterfactuals: list[str],
    ) -> dict[str, object]:
        top_proxy = proxy_features[0]["feature"] if proxy_features else "no dominant proxy"
        summary = (
            f"The audit detected a Bias Severity Index of {bsi_score} with a {risk_tier.get('label', '')} tier "
            f"under {legal_context.get('framework', 'general')} standards."
        )
        findings = [
            f"Top proxy feature: {top_proxy}.",
            f"Temporal drift status: {temporal_drift.get('status', 'not_available')}.",
            text_bias.get("summary", "Text bias analysis not available."),
        ]
        risk_assessment = f"Risk tier: {risk_tier.get('label', 'Unknown')} ({risk_tier.get('action', '')})"
        legal_exposure = legal_context.get("notes", "No specific legal exposure identified.")
        return {
            "executive_summary": summary,
            "findings": [item for item in findings if item],
            "risk_assessment": risk_assessment,
            "legal_exposure": legal_exposure,
            "recommended_actions": suggestions,
            "counterfactuals": counterfactuals,
        }
