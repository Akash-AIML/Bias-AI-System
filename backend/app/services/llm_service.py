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
        suggestions: list[str],
        group_metrics: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        if not self.client:
            return self._fallback_explanation(dp_diff, eo_diff, suggestions)

        prompt = (
            "Generate a concise fairness explanation for non-technical users. "
            "Return strict JSON with keys: explanation, summary, report_text. "
            "Mention demographic parity and equalized odds in plain language. "
            f"dp_diff={dp_diff}, eo_diff={eo_diff}, "
            f"suggestions={json.dumps(suggestions)}, "
            f"group_metrics={json.dumps(group_metrics)}"
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
                "explanation": str(payload.get("explanation", "Fairness analysis completed.")),
                "summary": str(payload.get("summary", "Bias metrics generated.")),
                "report_text": str(payload.get("report_text", "See metrics and suggestions.")),
            }
        except Exception:
            return self._fallback_explanation(dp_diff, eo_diff, suggestions)

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
    def _fallback_explanation(dp_diff: float, eo_diff: float, suggestions: list[str]) -> dict[str, str]:
        dp_percent = round(abs(dp_diff) * 100, 1)
        eo_percent = round(abs(eo_diff) * 100, 1)
        explanation = (
            f"Demographic parity differs by {dp_percent}% and equalized odds differs by {eo_percent}% "
            "between groups."
        )
        summary = "One or more groups may be less likely to receive favorable outcomes."
        report_text = (
            "Fairness analysis completed using statistical metrics. "
            f"Recommended actions: {'; '.join(suggestions)}"
        )
        return {
            "explanation": explanation,
            "summary": summary,
            "report_text": report_text,
        }
