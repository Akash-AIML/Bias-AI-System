from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.fairness import compute_fairness
from app.services.llm_service import LLMService
from app.services.mitigation import suggest_mitigations
from app.services.preprocessing import dataset_profile, load_and_preprocess

router = APIRouter(tags=["analysis"])
llm_service = LLMService()


class AnalyzeResponse(BaseModel):
    bias: bool
    dp_diff: float
    eo_diff: float
    group_metrics: dict[str, dict[str, float]]
    suggestions: list[str]
    explanation: str
    summary: str
    report_text: str
    intent: str
    domain: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    sensitive: str = Form(...),
    query: str = Form("check bias"),
) -> AnalyzeResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/csv"}:
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        preprocessed = load_and_preprocess(file_bytes, target=target, sensitive=sensitive)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    intent_payload = llm_service.infer_intent(
        query=query,
        dataset_context=dataset_profile(preprocessed.dataframe, target=target, sensitive=sensitive),
    )

    try:
        fairness_result = compute_fairness(
            features=preprocessed.features,
            labels=preprocessed.labels,
            sensitive_values=preprocessed.sensitive_values,
            numeric_features=preprocessed.numeric_features,
            categorical_features=preprocessed.categorical_features,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    suggestions = suggest_mitigations(
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        bias_detected=fairness_result.bias,
    )

    if intent_payload["intent"] == "mitigation" and len(suggestions) > 1:
        suggestions = [
            "Prioritize mitigation rollout in this order:",
            *suggestions,
        ]

    explanation_payload = llm_service.generate_explanation(
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        suggestions=suggestions,
        group_metrics=fairness_result.group_metrics,
    )

    return AnalyzeResponse(
        bias=fairness_result.bias,
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        group_metrics=fairness_result.group_metrics,
        suggestions=suggestions,
        explanation=explanation_payload["explanation"],
        summary=explanation_payload["summary"],
        report_text=explanation_payload["report_text"],
        intent=intent_payload["intent"],
        domain=intent_payload["domain"],
    )
