from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types


class LLMService:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.client = None
        self.openai_client = None

        if api_key:
            client_kwargs = {"api_key": api_key}
            if base_url:
                if "openai" in base_url.lower() or "groq" in base_url.lower():
                    print(
                        f"[LLM SERVICE WARNING] LLM_BASE_URL ({base_url}) appears to be an OpenAI or Groq endpoint. "
                        "The Google Gemini SDK uses a different URL structure (:generateContent) and will fail "
                        "with 404 errors if directed to OpenAI-compatible API routes. "
                        "Please clear LLM_BASE_URL or point it to a Gemini-compatible gateway."
                    )
                client_kwargs["http_options"] = types.HttpOptions(base_url=base_url, timeout=10)
            self.client = genai.Client(**client_kwargs)

            # Initialize OpenAI client as a fallback for custom/unsupported gateways
            import openai
            openai_base_url = base_url
            if openai_base_url:
                if not openai_base_url.endswith('/v1') and not openai_base_url.endswith('/v1/'):
                    openai_base_url = openai_base_url.rstrip('/') + '/v1'
            self.openai_client = openai.OpenAI(api_key=api_key, base_url=openai_base_url)

    def _call_gemini(
        self,
        system_instruction: str,
        prompt: str,
        temperature: float = 0.2,
        force_json: bool = True,
    ) -> str:
        if not self.client:
            return ""
        try:
            config_kwargs = {
                "system_instruction": system_instruction,
                "temperature": temperature,
            }
            if force_json:
                config_kwargs["response_mime_type"] = "application/json"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return response.text or ""
        except Exception as e:
            print(f"[LLM SERVICE ERROR] Gemini SDK call failed: {type(e)} {e}")
            if self.openai_client:
                print("[LLM SERVICE INFO] Attempting fallback to OpenAI client...")
                try:
                    messages = [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ]
                    response_format = {"type": "json_object"} if force_json else None
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        response_format=response_format
                    )
                    return response.choices[0].message.content or ""
                except Exception as openai_err:
                    print(f"[LLM SERVICE ERROR] OpenAI fallback also failed: {type(openai_err)} {openai_err}")
            raise e

    @staticmethod
    def _parse_json(content: str) -> Any:
        if not content:
            return {}
        s = content.strip()
        if s.startswith("```"):
            first_newline = s.find("\n")
            if first_newline != -1:
                s = s[first_newline:].strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                for closure in ['"', '"}', '"]}', '"]} }', '"} }', '"} } }']:
                    try:
                        return json.loads(s + closure)
                    except json.JSONDecodeError:
                        pass
                
                last_quote = s.rfind('"')
                if last_quote > 0:
                    s_chopped = s[:last_quote].strip()
                    if s_chopped.endswith(','):
                        s_chopped = s_chopped[:-1]
                    elif s_chopped.endswith(':'):
                        s_chopped += ' null'
                    
                    for closure in ['}', ']}', '}]}', '}}', '}}}']:
                        try:
                            return json.loads(s_chopped + closure)
                        except json.JSONDecodeError:
                            pass
            raise e

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
            content = self._call_gemini(
                system_instruction="Output only valid JSON.",
                prompt=prompt,
                temperature=0.1,
                force_json=True
            )
            payload = self._parse_json(content)
            intent = payload.get("intent", "bias_detection")
            domain = payload.get("domain", "general")
            if intent not in {"bias_detection", "mitigation"}:
                intent = "bias_detection"
            return {"intent": intent, "domain": str(domain)}
        except Exception as e:
            print(f"[LLM SERVICE ERROR] infer_intent failed: {type(e)} {e}")
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
            content = self._call_gemini(
                system_instruction="Output only valid JSON.",
                prompt=prompt,
                temperature=0.2,
                force_json=True
            )
            payload = self._parse_json(content)
            return {
                "explanation": str(payload.get("explanation", baseline["explanation"])),
                "summary": self._sanitize_summary(
                    summary=str(payload.get("summary", baseline["summary"])),
                    dp_diff=dp_diff,
                    eo_diff=eo_diff,
                ),
                "report_text": str(payload.get("report_text", baseline["report_text"])),
            }
        except Exception as e:
            print(f"[LLM SERVICE ERROR] generate_explanation failed: {type(e)} {e}")
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
            content = self._call_gemini(
                system_instruction="Output only valid JSON.",
                prompt=prompt,
                temperature=0.2,
                force_json=True
            )
            payload = self._parse_json(content)
            return {
                "domain": str(payload.get("domain", "general")),
                "framework": str(payload.get("framework", "General fairness standards")),
                "notes": str(payload.get("notes", "")),
            }
        except Exception as e:
            print(f"[LLM SERVICE ERROR] classify_legal_context failed: {type(e)} {e}")
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
            content = self._call_gemini(
                system_instruction="Output only valid JSON.",
                prompt=prompt,
                temperature=0.3,
                force_json=True
            )
            payload = self._parse_json(content)
            counterfactuals = payload.get("counterfactuals", [])
            return [str(item) for item in counterfactuals if str(item).strip()]
        except Exception as e:
            print(f"[LLM SERVICE ERROR] generate_counterfactuals failed: {type(e)} {e}")
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
            content = self._call_gemini(
                system_instruction="Output only valid JSON.",
                prompt=prompt,
                temperature=0.3,
                force_json=True
            )
            payload = self._parse_json(content)
            return str(payload.get("interpretation", ""))
        except Exception as e:
            print(f"[LLM SERVICE ERROR] interpret_temporal_drift failed: {type(e)} {e}")
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
            content = self._call_gemini(
                system_instruction="Output only valid JSON.",
                prompt=prompt,
                temperature=0.25,
                force_json=True
            )
            payload = self._parse_json(content)
            return {
                "executive_summary": str(payload.get("executive_summary", "")),
                "findings": [str(item) for item in payload.get("findings", [])],
                "risk_assessment": str(payload.get("risk_assessment", "")),
                "legal_exposure": str(payload.get("legal_exposure", "")),
                "recommended_actions": [str(item) for item in payload.get("recommended_actions", [])],
            }
        except Exception as e:
            print(f"[LLM SERVICE ERROR] generate_audit_narrative failed: {type(e)} {e}")
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

    def generate_warnings(
        self,
        raw_warnings: list[str],
        raw_info_notes: list[str],
        dp_diff: float,
        eo_diff: float,
        bsi_score: float,
        group_metrics: dict[str, dict[str, float]],
        audit_mode: str,
        dataset_rows: int,
        weight_column: str | None = None,
    ) -> dict[str, list[str]]:
        """Enrich raw statistical warnings and info notes using Gemini.

        Returns a dict with keys 'warnings' and 'info_notes' — lists of
        plain-English, human-readable messages.
        """
        if not raw_warnings and not raw_info_notes:
            return {"warnings": [], "info_notes": []}

        if not self.client:
            return {"warnings": raw_warnings, "info_notes": raw_info_notes}

        # Build a compact group summary string for the prompt
        group_lines = ", ".join(
            f"{g}: selection_rate={round(m.get('selection_rate', 0)*100, 1)}%"
            for g, m in group_metrics.items()
        )

        prompt = (
            "You are writing audit findings for a BUSINESS EXECUTIVE — not a data scientist. "
            "Your task is to rewrite each raw statistical finding below into 1-2 plain-English sentences. "
            "STRICT RULES:\n"
            "1. EVERY rewritten item MUST include at least one specific number or group name from the context. "
            "2. Replace ALL technical jargon: instead of 'statistical confidence' say 'reliable conclusions'; "
            "instead of 'one-vs-rest' say 'most common outcome vs all others'. "
            "3. Do NOT copy the raw text verbatim — the output MUST be clearly rephrased. "
            "4. Do NOT add warnings that are not in raw_warnings. "
            "5. Return strict JSON with keys 'warnings' (list) and 'info_notes' (list). "
            "6. Preserve the EXACT count: same number of items out as in for each list.\n\n"
            f"raw_warnings (rewrite each → warnings list): {json.dumps(raw_warnings)}\n"
            f"raw_info_notes (rewrite each → info_notes list): {json.dumps(raw_info_notes)}\n\n"
            f"Context you MUST reference in your rewrites:\n"
            f"  - Dataset has {dataset_rows} rows\n"
            f"  - Demographic parity gap: {round(abs(dp_diff)*100, 1)}%\n"
            f"  - Equalized odds gap: {round(abs(eo_diff)*100, 1)}%\n"
            f"  - Bias Severity Index: {bsi_score}/100\n"
            f"  - Audit mode: {audit_mode} ({'no prediction column provided' if audit_mode == 'proxy' else 'real predictions used'})\n"
            f"  - Group selection rates: {group_lines}\n"
            + (f"  - Sample weights applied from column '{weight_column}'\n" if weight_column else "")
        )

        try:
            content = self._call_gemini(
                system_instruction="Output only valid JSON. Do not copy raw input verbatim.",
                prompt=prompt,
                temperature=0.4,
                force_json=True
            )
            payload = self._parse_json(content)
            enriched_warnings = [str(w) for w in payload.get("warnings", raw_warnings)]
            enriched_notes = [str(n) for n in payload.get("info_notes", raw_info_notes)]
            # Fall back to raw if Gemini returned identical content (not enriched)
            return {
                "warnings": enriched_warnings if enriched_warnings != raw_warnings else raw_warnings,
                "info_notes": enriched_notes if enriched_notes != raw_info_notes else raw_info_notes,
            }
        except Exception as e:
            print(f"[LLM SERVICE ERROR] generate_warnings failed: {type(e)} {e}")
            return {"warnings": raw_warnings, "info_notes": raw_info_notes}

    def generate_mitigation_suggestions(
        self,
        dp_diff: float,
        eo_diff: float,
        bsi_score: float,
        bias_detected: bool,
        group_metrics: dict[str, dict[str, float]],
        proxy_features: list[dict[str, object]],
        audit_mode: str,
        domain: str,
        weight_column: str | None = None,
    ) -> list[str]:
        """Generate contextual, domain-aware mitigation suggestions using Gemini.

        Falls back to the deterministic threshold rules when Gemini is unavailable.
        """
        fallback = self._fallback_mitigation_suggestions(
            dp_diff=dp_diff,
            eo_diff=eo_diff,
            bias_detected=bias_detected,
        )

        if not self.client:
            return fallback

        top_proxies = [str(p.get("feature", "")) for p in proxy_features[:3]]
        sorted_groups = sorted(
            group_metrics.items(),
            key=lambda item: item[1].get("selection_rate", 0.0),
        )

        if sorted_groups:
            least_favoured = sorted_groups[0][0]
            most_favoured = sorted_groups[-1][0]
            rate_low = round(sorted_groups[0][1].get("selection_rate", 0.0) * 100, 1)
            rate_high = round(sorted_groups[-1][1].get("selection_rate", 0.0) * 100, 1)
            gap_pct = round((rate_high - rate_low), 1)
        else:
            least_favoured = most_favoured = "unknown"
            rate_low = rate_high = gap_pct = 0

        proxy_str = ", ".join(top_proxies) if top_proxies else "none identified"
        weight_note = (
            f"Sample weights from column '{weight_column}' were used in this audit — "
            "recommend re-auditing with real model predictions to confirm improvement."
            if weight_column else ""
        )

        prompt = (
            f"You are a senior fairness consultant writing a bias remediation plan for a {domain} system. "
            f"The model shows a {round(abs(dp_diff)*100,1)}% demographic parity gap and a "
            f"{round(abs(eo_diff)*100,1)}% equalized odds gap (Bias Severity Index: {bsi_score}/100). "
            f"The least-favoured group is '{least_favoured}' ({rate_low}% selection rate) vs "
            f"'{most_favoured}' ({rate_high}% selection rate) — a {gap_pct} percentage-point gap. "
            f"Proxy features driving bias: {proxy_str}. "
            f"Audit mode: {audit_mode} ({'no real predictions — proxy model used' if audit_mode == 'proxy' else 'real model predictions audited'}). "
            + weight_note + "\n\n"
            "Generate exactly 4 specific, actionable mitigation recommendations. RULES:\n"
            "1. EVERY recommendation MUST name the specific groups, numbers, or features above — "
            "no generic advice.\n"
            "2. Do NOT use these generic phrases: 'Apply reweighting by sensitive group', "
            "'Adjust decision thresholds per group', 'Use post-processing equalized odds optimization', "
            "'Resample underrepresented groups'. Rephrase these ideas using the actual data.\n"
            "3. At least one recommendation must address the specific proxy feature(s) found.\n"
            "4. At least one recommendation must mention the specific domain context.\n"
            "5. Return strict JSON: {\"suggestions\": [\"...\", \"...\", \"...\", \"...\"]}"
        )

        try:
            content = self._call_gemini(
                system_instruction="Output only valid JSON. All suggestions must be specific and reference real data values — no generic template phrases.",
                prompt=prompt,
                temperature=0.5,
                force_json=True
            )
            payload = self._parse_json(content)
            suggestions = [str(s) for s in payload.get("suggestions", []) if str(s).strip()]
            # Verify Gemini actually personalised the output (shouldn't match static fallback exactly)
            if suggestions and suggestions != fallback:
                return suggestions
            return fallback
        except Exception as e:
            print(f"[LLM SERVICE ERROR] generate_mitigation_suggestions failed: {type(e)} {e}")
            return fallback

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
            warning_text = f" Notes: {'; '.join(warnings)}"

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

    @staticmethod
    def _fallback_mitigation_suggestions(
        dp_diff: float,
        eo_diff: float,
        bias_detected: bool,
    ) -> list[str]:
        if not bias_detected:
            return ["No severe bias detected. Continue monitoring with periodic fairness checks."]
        suggestions: list[str] = []
        if abs(dp_diff) > 0.10:
            suggestions.append("Apply reweighting by sensitive group to reduce demographic parity disparity.")
            suggestions.append("Adjust decision thresholds per group after legal and policy review.")
        if abs(eo_diff) > 0.10:
            suggestions.append("Use post-processing equalized odds optimization to align error rates across groups.")
        if abs(dp_diff) > 0.20 or abs(eo_diff) > 0.20:
            suggestions.append("Resample underrepresented groups and retrain with fairness constraints.")
        if not suggestions:
            suggestions.append("Review feature engineering choices and collect more balanced data.")
        return suggestions

    def _fallback_combined_report(
        self,
        query: str,
        dataset_context: dict[str, Any],
        dp_diff: float,
        eo_diff: float,
        bsi_score: float,
        bias_detected: bool,
        group_metrics: dict[str, dict[str, float]],
        proxy_features: list[dict[str, Any]],
        temporal_drift: dict[str, Any],
        text_bias: dict[str, Any],
        raw_warnings: list[str],
        raw_info_notes: list[str],
        audit_mode: str,
        dataset_rows: int,
        weight_column: str | None = None,
        reliability_note: str = "",
        target_transformation: dict[str, Any] | None = None,
        risk_tier: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        intent_payload = self._fallback_intent(query)
        legal_context = self._fallback_legal_context(query, dataset_context)
        domain = legal_context.get("domain", "general")
        
        suggestions = self._fallback_mitigation_suggestions(
            dp_diff=dp_diff,
            eo_diff=eo_diff,
            bias_detected=bias_detected,
        )
        
        explanation_payload = self._deterministic_explanation(
            dp_diff=dp_diff,
            eo_diff=eo_diff,
            bsi_score=bsi_score,
            suggestions=suggestions,
            group_metrics=group_metrics,
            audit_mode=audit_mode,
            reliability_note=reliability_note,
            target_transformation=target_transformation or {},
            warnings=raw_warnings,
        )
        
        counterfactuals = self._fallback_counterfactuals(group_metrics, domain)
        
        resolved_risk_tier = risk_tier or {"label": "Unknown", "action": "None"}
        audit_narrative = self._fallback_audit_narrative(
            legal_context=legal_context,
            bsi_score=bsi_score,
            risk_tier=resolved_risk_tier,
            proxy_features=proxy_features,
            temporal_drift=temporal_drift,
            text_bias=text_bias,
            suggestions=suggestions,
            counterfactuals=counterfactuals,
        )
        
        return {
            "intent": intent_payload["intent"],
            "domain": domain,
            "legal_context": legal_context,
            "warnings": raw_warnings,
            "info_notes": raw_info_notes,
            "suggestions": suggestions,
            "explanation": explanation_payload["explanation"],
            "summary": explanation_payload["summary"],
            "report_text": explanation_payload["report_text"],
            "counterfactuals": counterfactuals,
            "audit_narrative": audit_narrative,
        }

    def generate_combined_audit_report(
        self,
        query: str,
        dataset_context: dict[str, Any],
        dp_diff: float,
        eo_diff: float,
        bsi_score: float,
        bias_detected: bool,
        group_metrics: dict[str, dict[str, float]],
        proxy_features: list[dict[str, Any]],
        temporal_drift: dict[str, Any],
        text_bias: dict[str, Any],
        raw_warnings: list[str],
        raw_info_notes: list[str],
        audit_mode: str,
        dataset_rows: int,
        weight_column: str | None = None,
        reliability_note: str = "",
        target_transformation: dict[str, Any] | None = None,
        risk_tier: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete consolidated audit report in a single Gemini call to save tokens and quota."""
        fallback_combined = self._fallback_combined_report(
            query=query,
            dataset_context=dataset_context,
            dp_diff=dp_diff,
            eo_diff=eo_diff,
            bsi_score=bsi_score,
            bias_detected=bias_detected,
            group_metrics=group_metrics,
            proxy_features=proxy_features,
            temporal_drift=temporal_drift,
            text_bias=text_bias,
            raw_warnings=raw_warnings,
            raw_info_notes=raw_info_notes,
            audit_mode=audit_mode,
            dataset_rows=dataset_rows,
            weight_column=weight_column,
            reliability_note=reliability_note,
            target_transformation=target_transformation,
            risk_tier=risk_tier,
        )

        if not self.client:
            return fallback_combined

        # Limit group metrics for prompt if there are too many (e.g., intersectional)
        sorted_groups = sorted(
            group_metrics.items(),
            key=lambda item: item[1].get('selection_rate', 0.0)
        )
        if len(sorted_groups) > 10:
            prompt_groups = dict(sorted_groups[:5] + sorted_groups[-5:])
            group_note = f" (showing 10 extreme groups out of {len(sorted_groups)})"
        else:
            prompt_groups = group_metrics
            group_note = ""

        # ── Rich group table for prompt ──────────────────────────────────────
        group_rows = []
        for g, m in prompt_groups.items():
            row = (
                f"  {g}: selection={round(m.get('selection_rate',0)*100,1)}%, "
                f"accuracy={round(m.get('accuracy',0)*100,1)}%"
            )
            if 'fpr' in m:
                row += f", FPR={round(m['fpr']*100,1)}%, FNR={round(m.get('fnr',0)*100,1)}%"
            group_rows.append(row)
        group_table = "\n".join(group_rows) + group_note

        # ── Proxy variable table with direction and strength ─────────────────
        proxy_rows = []
        for p in proxy_features[:8]:
            corr = p.get('correlation', 0)
            strength = "strong" if corr > 0.5 else "moderate" if corr > 0.25 else "weak"
            proxy_rows.append(
                f"  [{strength}] {p.get('feature','?')} — correlation={corr:.3f} ({p.get('direction','?')})"
            )
        proxy_table = "\n".join(proxy_rows) if proxy_rows else "  none identified"

        # ── BSI interpretation band ──────────────────────────────────────────
        if bsi_score < 20:
            bsi_band = "LOW (0-20): model is largely fair, minor monitoring sufficient"
        elif bsi_score < 40:
            bsi_band = "MODERATE (20-40): notable disparity, review and improve data pipeline"
        elif bsi_score < 65:
            bsi_band = "HIGH (40-65): significant bias present, mitigation required before deployment"
        else:
            bsi_band = "CRITICAL (65-100): severe bias, immediate halt and remediation required"

        # ── Temporal drift summary ───────────────────────────────────────────
        drift_status = temporal_drift.get('status', 'not_available') if isinstance(temporal_drift, dict) else 'not_available'
        drift_detail = ""
        if isinstance(temporal_drift, dict) and temporal_drift.get('early_bsi') is not None:
            drift_detail = (
                f" Early BSI={temporal_drift.get('early_bsi')}, "
                f"Late BSI={temporal_drift.get('late_bsi')}, "
                f"Delta={temporal_drift.get('delta')}"
            )

        # ── Text bias summary ────────────────────────────────────────────────
        text_bias_summary = ""
        if isinstance(text_bias, dict) and text_bias.get('has_text_columns'):
            gaps = text_bias.get('sentiment_gaps', [])
            if gaps:
                g0 = gaps[0]
                text_bias_summary = (
                    f"Sentiment gap between '{g0.get('group_a')}' and '{g0.get('group_b')}': "
                    f"{round(g0.get('gap',0),3)} in column '{g0.get('column')}'"
                )

        prompt = (
            "You are a senior AI fairness auditor and legal compliance specialist with deep expertise in "
            "algorithmic bias detection, EU AI Act, EEOC Title VII, FCRA, FHA, and ACA frameworks. "
            "Analyse the computed fairness metrics below and produce a precise, evidence-based audit report.\n\n"

            "═══════════════════ DATASET FACTS ═══════════════════\n"
            f"Query            : {query}\n"
            f"Domain Context   : {json.dumps(dataset_context.get('column_names', []))}\n"
            f"Total Rows       : {dataset_rows:,}\n"
            f"Audit Mode       : {audit_mode} "
            f"({'no predictions — proxy logistic regression trained' if audit_mode == 'proxy' else 'real model predictions audited'})\n"
            + (f"Sample Weights   : column '{weight_column}' applied\n" if weight_column else "")
            + f"Target Encoding  : {json.dumps(target_transformation or {})}\n"
            f"Reliability      : {reliability_note}\n\n"

            "═══════════════════ FAIRNESS METRICS ═══════════════════\n"
            f"Demographic Parity Gap  : {round(abs(dp_diff)*100, 2)}%  "
            f"({'EXCEEDS' if abs(dp_diff) > 0.10 else 'within'} the 10% warning threshold)\n"
            f"Equalized Odds Gap      : {round(abs(eo_diff)*100, 2)}%  "
            f"({'EXCEEDS' if abs(eo_diff) > 0.10 else 'within'} the 10% warning threshold)\n"
            f"Bias Severity Index     : {bsi_score}/100  →  {bsi_band}\n"
            f"Bias Flag               : {'YES — bias is present' if bias_detected else 'NO — no significant bias detected'}\n\n"

            "═══════════════════ GROUP BREAKDOWN ═══════════════════\n"
            f"{group_table}\n\n"

            "═══════════════════ PROXY VARIABLE ANALYSIS ═══════════════════\n"
            "These features are statistically correlated with the sensitive attribute and may be "
            "transmitting discrimination indirectly (even without being a protected attribute themselves):\n"
            f"{proxy_table}\n\n"

            "═══════════════════ TEMPORAL DRIFT ═══════════════════\n"
            f"Status: {drift_status}{drift_detail}\n\n"

            + (f"═══════════════════ TEXT BIAS ═══════════════════\n{text_bias_summary}\n\n" if text_bias_summary else "")

            + "═══════════════════ DATA QUALITY WARNINGS ═══════════════════\n"
            f"{json.dumps(raw_warnings)}\n\n"
            f"Info Notes: {json.dumps(raw_info_notes)}\n\n"

            + (f"Risk Tier: {json.dumps(risk_tier)}\n\n" if risk_tier else "")

            + "═══════════════════ YOUR TASK ═══════════════════\n"
            "Return ONE valid JSON object with ALL of the following fields. Be precise and reference the "
            "actual numbers, group names, and feature names from the context above — never be generic.\n\n"

            "FIELD DEFINITIONS:\n"
            "1. 'intent' — string: 'bias_detection' or 'mitigation' based on user query.\n"

            "2. 'domain' — string: one of 'hiring', 'credit', 'housing', 'healthcare', 'general'. "
            "Infer from column names and query.\n"

            "3. 'legal_context' — object: { domain, framework (cite the specific law/rule that applies), "
            "notes (1 sentence on legal risk) }.\n"

            "4. 'bias_types' — list of strings: Identify every type of bias present from this taxonomy — "
            "use ONLY these labels and only include those that actually apply:\n"
            "   - 'Selection Bias' (certain groups systematically excluded)\n"
            "   - 'Historical Bias' (training data reflects past discrimination)\n"
            "   - 'Proxy Discrimination' (neutral feature transmits protected-class disadvantage)\n"
            "   - 'Measurement Bias' (features measured differently across groups)\n"
            "   - 'Label Bias' (ground-truth labels themselves reflect human prejudice)\n"
            "   - 'Representation Bias' (severe group size imbalance in training data)\n"
            "   - 'Algorithmic Amplification' (model amplifies existing small gaps)\n"
            "   - 'Temporal Drift Bias' (bias has worsened over time)\n"
            "   - 'Text/Language Bias' (sentiment or terminology differs by group)\n"
            "   If no bias is detected, return an empty list [].\n"

            "5. 'proxy_analysis' — list of objects, one per proxy feature (up to 8): "
            "{ feature, correlation, direction, strength ('strong'/'moderate'/'weak'), "
            "why_it_matters (1 sentence), how_to_fix (1 specific action) }.\n"

            "6. 'bsi_interpretation' — object: { score, band, meaning (1 plain-English sentence "
            "explaining what BSI={bsi_score} means for this specific dataset), "
            "trend (if temporal drift available, state whether BSI is worsening or stable) }.\n"

            "7. 'warnings' — list of strings: Rewrite each raw_warning for a business executive. "
            "No jargon. Name specific groups and numbers. EXACT same count as raw_warnings.\n"

            "8. 'info_notes' — list of strings: Rewrite each raw info note. EXACT same count as raw_info_notes.\n"

            "9. 'explanation' — string: 3-4 sentence plain-English explanation covering: "
            "(a) what bias was found and which groups are affected, "
            "(b) what the BSI score means in context, "
            "(c) which proxy features are the main drivers, "
            "(d) what this means for real people impacted by this system.\n"

            "10. 'summary' — string: One sentence verdict. Do NOT claim disadvantage if both parity gaps < 1%.\n"

            "11. 'report_text' — string: 2-3 sentence compliance-grade summary for a legal/audit report.\n"

            "12. 'counterfactuals' — list of strings: For each disadvantaged group, write a specific "
            "counterfactual sentence: 'A person in group X with similar qualifications to someone in group Y "
            "is [N]% less/more likely to receive [outcome].' If no gap, confirm equity.\n"

            "13. 'rectification_plan' — list of objects, exactly 4 steps in priority order: "
            "{ priority (1-4), action (short title), description (1-2 sentences — cite specific features "
            "or groups), expected_impact ('Reduces DP gap by ~X%' or similar), timeline ('Immediate', "
            "'1-3 months', '3-6 months', '6-12 months') }.\n"
            "Steps MUST address: proxy feature removal/adjustment, data rebalancing, "
            "model retraining with fairness constraints, and ongoing monitoring.\n"

            "14. 'suggestions' — list of strings: The 4 actions from rectification_plan as plain sentences. "
            "Must name specific groups/features from context. No generic templates.\n"

            "15. 'audit_narrative' — object:\n"
            "    - 'executive_summary': 2-3 sentence overview for C-suite, naming the domain, BSI, and worst-affected group.\n"
            "    - 'findings': List of bullet strings — one per key finding (proxy features, drift, text bias, group gaps).\n"
            "    - 'risk_assessment': 1 sentence naming the risk tier and what it means operationally.\n"
            "    - 'legal_exposure': 1 sentence identifying the specific legal risk under the applicable framework.\n"
            "    - 'recommended_actions': Mirror of rectification_plan as plain action strings.\n\n"

            "═══════════════════ JSON COMPLIANCE RULES ═══════════════════\n"
            "RULE 1: Do NOT use double quotes inside string values. Use single quotes if quoting within text.\n"
            "RULE 2: Every string must be a valid JSON string (escaped properly).\n"
            "RULE 3: Return ONLY the JSON object — no markdown fences, no preamble, no explanation.\n"
            "RULE 4: All list fields must be actual JSON arrays [ ... ], never a string.\n"
            "RULE 5: All object fields must be actual JSON objects { ... }, never a string.\n\n"

            "═══════════════════ REQUIRED OUTPUT SCHEMA ═══════════════════\n"
            "{\n"
            "  \"intent\": \"bias_detection\" | \"mitigation\",\n"
            "  \"domain\": \"hiring\" | \"credit\" | \"housing\" | \"healthcare\" | \"general\",\n"
            "  \"legal_context\": { \"domain\": \"...\", \"framework\": \"...\", \"notes\": \"...\" },\n"
            "  \"bias_types\": [ \"Selection Bias\", \"Proxy Discrimination\", ... ],\n"
            "  \"proxy_analysis\": [\n"
            "    { \"feature\": \"...\", \"correlation\": 0.00, \"direction\": \"positive|negative\",\n"
            "      \"strength\": \"strong|moderate|weak\", \"why_it_matters\": \"...\", \"how_to_fix\": \"...\" }\n"
            "  ],\n"
            "  \"bsi_interpretation\": { \"score\": 0.0, \"band\": \"...\", \"meaning\": \"...\", \"trend\": \"...\" },\n"
            "  \"warnings\": [ ... ],\n"
            "  \"info_notes\": [ ... ],\n"
            "  \"explanation\": \"...\",\n"
            "  \"summary\": \"...\",\n"
            "  \"report_text\": \"...\",\n"
            "  \"counterfactuals\": [ ... ],\n"
            "  \"rectification_plan\": [\n"
            "    { \"priority\": 1, \"action\": \"...\", \"description\": \"...\",\n"
            "      \"expected_impact\": \"...\", \"timeline\": \"...\" }\n"
            "  ],\n"
            "  \"suggestions\": [ ... ],\n"
            "  \"audit_narrative\": {\n"
            "    \"executive_summary\": \"...\",\n"
            "    \"findings\": [ ... ],\n"
            "    \"risk_assessment\": \"...\",\n"
            "    \"legal_exposure\": \"...\",\n"
            "    \"recommended_actions\": [ ... ]\n"
            "  }\n"
            "}"
        )

        try:
            content = self._call_gemini(
                system_instruction="Output only valid JSON conforming to the schema. Keep text values brief and concise to save tokens. Do not use double quotes inside string values under any circumstances.",
                prompt=prompt,
                temperature=0.3,
                force_json=True
            )
            payload = self._parse_json(content)

            # ── Standard fields ──────────────────────────────────────────
            intent = payload.get("intent", fallback_combined["intent"])
            domain = payload.get("domain", fallback_combined["domain"])

            legal_ctx = payload.get("legal_context", {})
            if not isinstance(legal_ctx, dict):
                legal_ctx = fallback_combined["legal_context"]
            else:
                legal_ctx = {
                    "domain": str(legal_ctx.get("domain", domain)),
                    "framework": str(legal_ctx.get("framework", fallback_combined["legal_context"]["framework"])),
                    "notes": str(legal_ctx.get("notes", fallback_combined["legal_context"]["notes"])),
                }

            warnings_out = payload.get("warnings")
            if isinstance(warnings_out, list):
                warnings_out = [str(w) for w in warnings_out]
            else:
                warnings_out = raw_warnings

            info_notes_out = payload.get("info_notes")
            if isinstance(info_notes_out, list):
                info_notes_out = [str(n) for n in info_notes_out]
            else:
                info_notes_out = raw_info_notes

            suggestions_out = payload.get("suggestions")
            if isinstance(suggestions_out, list) and suggestions_out:
                suggestions_out = [str(s) for s in suggestions_out]
            else:
                suggestions_out = fallback_combined["suggestions"]

            explanation = str(payload.get("explanation", fallback_combined["explanation"]))
            summary = self._sanitize_summary(
                summary=str(payload.get("summary", fallback_combined["summary"])),
                dp_diff=dp_diff,
                eo_diff=eo_diff,
            )
            report_text = str(payload.get("report_text", fallback_combined["report_text"]))

            counterfactuals_out = payload.get("counterfactuals")
            if isinstance(counterfactuals_out, list) and counterfactuals_out:
                counterfactuals_out = [str(c) for c in counterfactuals_out]
            else:
                counterfactuals_out = fallback_combined["counterfactuals"]

            narrative_out = payload.get("audit_narrative")
            if not isinstance(narrative_out, dict):
                narrative_out = fallback_combined["audit_narrative"]
            else:
                narrative_out = {
                    "executive_summary": str(narrative_out.get("executive_summary", fallback_combined["audit_narrative"]["executive_summary"])),
                    "findings": [str(f) for f in narrative_out.get("findings", [])] if isinstance(narrative_out.get("findings"), list) else fallback_combined["audit_narrative"]["findings"],
                    "risk_assessment": str(narrative_out.get("risk_assessment", fallback_combined["audit_narrative"]["risk_assessment"])),
                    "legal_exposure": str(narrative_out.get("legal_exposure", fallback_combined["audit_narrative"]["legal_exposure"])),
                    "recommended_actions": [str(a) for a in narrative_out.get("recommended_actions", [])] if isinstance(narrative_out.get("recommended_actions"), list) else fallback_combined["audit_narrative"]["recommended_actions"],
                }

            # ── NEW: bias type classification ────────────────────────────────
            _VALID_BIAS_TYPES = {
                "Selection Bias", "Historical Bias", "Proxy Discrimination",
                "Measurement Bias", "Label Bias", "Representation Bias",
                "Algorithmic Amplification", "Temporal Drift Bias", "Text/Language Bias",
            }
            bias_types_raw = payload.get("bias_types", [])
            bias_types_out: list[str] = (
                [str(b) for b in bias_types_raw if str(b) in _VALID_BIAS_TYPES]
                if isinstance(bias_types_raw, list)
                else []
            )

            # ── NEW: proxy analysis detail ─────────────────────────────────
            proxy_analysis_raw = payload.get("proxy_analysis", [])
            proxy_analysis_out: list[dict[str, object]] = []
            if isinstance(proxy_analysis_raw, list):
                for item in proxy_analysis_raw:
                    if isinstance(item, dict):
                        proxy_analysis_out.append({
                            "feature": str(item.get("feature", "")),
                            "correlation": float(item.get("correlation", 0.0)),
                            "direction": str(item.get("direction", "")),
                            "strength": str(item.get("strength", "")),
                            "why_it_matters": str(item.get("why_it_matters", "")),
                            "how_to_fix": str(item.get("how_to_fix", "")),
                        })

            # ── NEW: BSI interpretation ───────────────────────────────────
            bsi_interp_raw = payload.get("bsi_interpretation", {})
            bsi_interp_out: dict[str, object] = {
                "score": bsi_score,
                "band": bsi_band,
                "meaning": "",
                "trend": drift_status,
            }
            if isinstance(bsi_interp_raw, dict):
                bsi_interp_out["meaning"] = str(bsi_interp_raw.get("meaning", ""))
                if bsi_interp_raw.get("trend"):
                    bsi_interp_out["trend"] = str(bsi_interp_raw["trend"])

            # ── NEW: rectification plan ───────────────────────────────────
            rect_plan_raw = payload.get("rectification_plan", [])
            rect_plan_out: list[dict[str, object]] = []
            if isinstance(rect_plan_raw, list):
                for step in rect_plan_raw:
                    if isinstance(step, dict):
                        rect_plan_out.append({
                            "priority": int(step.get("priority", 0)),
                            "action": str(step.get("action", "")),
                            "description": str(step.get("description", "")),
                            "expected_impact": str(step.get("expected_impact", "")),
                            "timeline": str(step.get("timeline", "")),
                        })

            return {
                "intent": intent,
                "domain": domain,
                "legal_context": legal_ctx,
                # ── new enriched fields ──
                "bias_types": bias_types_out,
                "proxy_analysis": proxy_analysis_out,
                "bsi_interpretation": bsi_interp_out,
                "rectification_plan": rect_plan_out,
                # ── standard fields ──
                "warnings": warnings_out,
                "info_notes": info_notes_out,
                "suggestions": suggestions_out,
                "explanation": explanation,
                "summary": summary,
                "report_text": report_text,
                "counterfactuals": counterfactuals_out,
                "audit_narrative": narrative_out,
            }

        except Exception as e:
            print(f"[LLM SERVICE ERROR] generate_combined_audit_report failed: {type(e)} {e}")
            try:
                print(f"[LLM SERVICE ERROR] RAW CONTENT THAT FAILED TO PARSE:\n{content}")
            except NameError:
                pass
            return fallback_combined

    def chat_with_agent(
        self,
        message: str,
        audit_report: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Chat with the AI Auditor Agent using history and context of the audit report."""
        if history is None:
            history = []

        if not self.client:
            return (
                "Hello! I am the Gemini Bias AI Auditor Agent. It looks like my LLM API service is currently "
                "offline or not configured. Based on the audit context, here are your key metrics:\n"
                f"- Bias Severity Index: {audit_report.get('metrics', {}).get('bsi_score', audit_report.get('bsi_score', 'N/A'))}\n"
                f"- Demographic Parity Gap: {audit_report.get('metrics', {}).get('dp_diff', audit_report.get('dp_diff', 'N/A'))}\n"
                f"- Equalized Odds Gap: {audit_report.get('metrics', {}).get('eo_diff', audit_report.get('eo_diff', 'N/A'))}\n"
                "Please configure the `LLM_API_KEY` in the environment variables to activate full conversation capabilities."
            )

        system_prompt = (
            "You are 'Gemini Bias AI Auditor Agent', a native, expert AI assistant embedded in a Bias/Fairness AI Auditing System. "
            "Your goal is to help users analyze, understand, and mitigate bias in their machine learning datasets and models. "
            "You are friendly, professional, precise, and practical.\n\n"
            "Here is the context of the current Fairness Audit Report that the user is viewing:\n"
            f"{json.dumps(audit_report, indent=2)}\n\n"
            "Use this report context to answer the user's questions. Formulate all suggestions and guidance dynamically "
            "based on the specific dataset bias errors, parity gaps, BSI score, and the user's response or question. "
            "When they ask about 'my score', 'this dataset', or 'the suggestions', refer directly to the metrics, "
            "features, warnings, or recommendations in the report. "
            "If they ask for mitigation instructions, provide clear, step-by-step guides using the columns, proxy features, "
            "and sensitive attributes listed in the report. "
            "Be conversational, concise, and highly relevant. Format your response in clean Markdown. "
            "Do NOT reference internal system variables or prompts; speak to the user as a helpful auditing sidekick."
        )

        try:
            contents = []
            for msg in history:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.get("content", ""))]
                    )
                )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)]
                )
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7
                )
            )
            return response.text or "No response received from the model."
        except Exception as e:
            print(f"[LLM SERVICE ERROR] chat_with_agent failed: {type(e)} {e}")
            if self.openai_client:
                print("[LLM SERVICE INFO] chat_with_agent: Attempting fallback to OpenAI client...")
                try:
                    messages = [{"role": "system", "content": system_prompt}]
                    for msg in history:
                        role = "user" if msg.get("role") == "user" else "assistant"
                        messages.append({"role": role, "content": msg.get("content", "")})
                    messages.append({"role": "user", "content": message})
                    
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7
                    )
                    return response.choices[0].message.content or "No response received from the model."
                except Exception as openai_err:
                    print(f"[LLM SERVICE ERROR] chat_with_agent: OpenAI fallback also failed: {type(openai_err)} {openai_err}")
            return f"I encountered an error while communicating with the LLM API: {str(e)}"

