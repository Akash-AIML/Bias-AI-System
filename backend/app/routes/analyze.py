from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.fairness import compute_fairness
from app.services.llm_service import LLMService
from app.services.mitigation import suggest_mitigations
from app.services.preprocessing import _fill_missing_values, dataset_profile, load_and_preprocess
from app.services.reporting import build_fairness_report
from app.services.reweighting import apply_group_reweighting

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
    audit_mode: str
    warnings: list[str]
    audit_report: dict[str, Any]


class ReportRequest(BaseModel):
    dataset_summary: dict[str, Any]
    verdict: str
    dp_diff: float
    eo_diff: float
    group_metrics: dict[str, dict[str, float]]
    suggestions: list[str]
    audit_mode: str


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    sensitive: str = Form(...),
    prediction_column: str | None = Form(None),
    query: str = Form("check bias"),
) -> AnalyzeResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/csv"}:
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        preprocessed = load_and_preprocess(
            file_bytes,
            target=target,
            sensitive=sensitive,
            prediction_column=prediction_column,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    intent_payload = llm_service.infer_intent(
        query=query,
        dataset_context=dataset_profile(
            preprocessed.dataframe,
            target=target,
            sensitive=sensitive,
            prediction_column=prediction_column,
        ),
    )

    try:
        fairness_result = compute_fairness(
            features=preprocessed.features,
            labels=preprocessed.labels,
            predictions=preprocessed.prediction_values,
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

    audit_report = {
        "mode": fairness_result.audit_mode,
        "verdict": "bias_detected" if fairness_result.bias else "no_severe_bias",
        "metrics": {
            "dp_diff": fairness_result.dp_diff,
            "eo_diff": fairness_result.eo_diff,
        },
        "group_metrics": fairness_result.group_metrics,
        "warnings": fairness_result.warnings,
        "reliability_note": fairness_result.reliability_note,
    }

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
        audit_mode=fairness_result.audit_mode,
        warnings=fairness_result.warnings,
        audit_report=audit_report,
    )


@router.post("/reweighted-csv")
async def reweighted_csv(
    file: UploadFile = File(...),
    sensitive: str = Form(...),
    target: str = Form(...),
) -> StreamingResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/csv"}:
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        preprocessed = load_and_preprocess(file_bytes, target=target, sensitive=sensitive)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    reweighted = apply_group_reweighting(preprocessed.dataframe, sensitive_column=sensitive)
    output = BytesIO()
    reweighted.to_csv(output, index=False)
    output.seek(0)

    filename = f"reweighted_{file.filename or 'dataset'}.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/report")
async def generate_report(payload: ReportRequest) -> StreamingResponse:
    pdf_bytes = build_fairness_report(payload.model_dump())
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fairness_report.pdf"},
    )


@router.post("/intersectional")
async def intersectional_analysis(
    file: UploadFile = File(...),
    target: str = Form(...),
    sensitive_a: str = Form(...),
    sensitive_b: str = Form(...),
    prediction_column: str | None = Form(None),
) -> AnalyzeResponse:
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/csv"}:
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as error:
        raise HTTPException(status_code=400, detail="Failed to parse CSV file") from error

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in dataset")
    if sensitive_a not in df.columns or sensitive_b not in df.columns:
        raise HTTPException(status_code=400, detail="Both sensitive columns must exist in dataset")
    if prediction_column and prediction_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Prediction column '{prediction_column}' not found in dataset")

    cleaned_df = _fill_missing_values(df)
    labels = cleaned_df[target]
    predictions = cleaned_df[prediction_column] if prediction_column else None
    combined_sensitive = (
        cleaned_df[sensitive_a].astype(str) + "|" + cleaned_df[sensitive_b].astype(str)
    )

    features = cleaned_df.drop(columns=[target, sensitive_a, sensitive_b])
    if prediction_column and prediction_column in features.columns:
        features = features.drop(columns=[prediction_column])

    numeric_features = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    fairness_result = compute_fairness(
        features=features,
        labels=labels,
        predictions=predictions,
        sensitive_values=combined_sensitive,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    suggestions = suggest_mitigations(
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        bias_detected=fairness_result.bias,
    )

    explanation_payload = llm_service.generate_explanation(
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        suggestions=suggestions,
        group_metrics=fairness_result.group_metrics,
    )

    audit_report = {
        "mode": fairness_result.audit_mode,
        "verdict": "bias_detected" if fairness_result.bias else "no_severe_bias",
        "metrics": {
            "dp_diff": fairness_result.dp_diff,
            "eo_diff": fairness_result.eo_diff,
        },
        "group_metrics": fairness_result.group_metrics,
        "warnings": fairness_result.warnings,
        "reliability_note": fairness_result.reliability_note,
    }

    return AnalyzeResponse(
        bias=fairness_result.bias,
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        group_metrics=fairness_result.group_metrics,
        suggestions=suggestions,
        explanation=explanation_payload["explanation"],
        summary=explanation_payload["summary"],
        report_text=explanation_payload["report_text"],
        intent="intersectional_analysis",
        domain="general",
        audit_mode=fairness_result.audit_mode,
        warnings=fairness_result.warnings,
        audit_report=audit_report,
    )
