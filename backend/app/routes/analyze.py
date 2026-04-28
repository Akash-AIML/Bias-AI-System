from __future__ import annotations

from io import BytesIO
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.bias_tracer import trace_bias_proxies
from app.services.fairness import compute_fairness
from app.services.llm_service import LLMService
from app.services.mitigation import suggest_mitigations
from app.services.mitigation_simulator import simulate_mitigation
from app.services.preprocessing import (
    _fill_missing_values,
    dataset_profile,
    load_and_preprocess,
    suggest_column_roles,
)
from app.services.reporting import build_fairness_report
from app.services.risk_classifier import classify_risk
from app.services.temporal_analyzer import analyze_temporal_drift
from app.services.text_bias_analyzer import analyze_text_bias
from app.services.text_encoder import encode_text_columns
from app.services.reweighting import apply_group_reweighting

router = APIRouter(tags=["analysis"])
llm_service = LLMService()


def _is_supported_csv_upload(file: UploadFile) -> bool:
    allowed_types = {"text/csv", "application/vnd.ms-excel", "application/csv", "application/octet-stream", ""}
    if file.content_type in allowed_types:
        return True
    filename = (file.filename or "").lower()
    return filename.endswith(".csv")


class AnalyzeResponse(BaseModel):
    bias: bool
    audit_id: str
    dp_diff: float
    eo_diff: float
    bsi_score: float
    risk_tier: dict[str, str]
    proxy_features: list[dict[str, object]]
    temporal_drift: dict[str, object]
    text_bias: dict[str, object]
    legal_context: dict[str, str]
    audit_narrative: dict[str, object]
    group_metrics: dict[str, dict[str, float]]
    suggestions: list[str]
    explanation: str
    summary: str
    report_text: str
    intent: str
    domain: str
    audit_mode: str
    warnings: list[str]
    target_transformation: dict[str, Any]
    audit_report: dict[str, Any]


class ReportRequest(BaseModel):
    dataset_summary: dict[str, Any]
    accountability_summary: dict[str, Any]
    verdict: str
    dp_diff: float
    eo_diff: float
    bsi_score: float
    risk_tier: dict[str, str]
    proxy_features: list[dict[str, object]]
    temporal_drift: dict[str, object]
    text_bias: dict[str, object]
    legal_context: dict[str, str]
    audit_narrative: dict[str, object]
    group_metrics: dict[str, dict[str, float]]
    suggestions: list[str]
    audit_mode: str


class ColumnSuggestionResponse(BaseModel):
    columns: list[str]
    target: str | None
    sensitive: str | None
    prediction_column: str | None
    time_column: str | None
    method: str
    notes: list[str]


@router.post("/column-suggestions", response_model=ColumnSuggestionResponse)
async def column_suggestions(file: UploadFile = File(...)) -> ColumnSuggestionResponse:
    if not _is_supported_csv_upload(file):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        payload = suggest_column_roles(file_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return ColumnSuggestionResponse(**payload)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    sensitive: str = Form(...),
    prediction_column: str | None = Form(None),
    org_name: str | None = Form(None),
    dataset_name: str | None = Form(None),
    time_column: str | None = Form(None),
    text_columns: str | None = Form(None),
    query: str = Form("check bias"),
    target_binarization_threshold: float | None = Form(None),
) -> AnalyzeResponse:
    if not _is_supported_csv_upload(file):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        text_column_list = (
            [column.strip() for column in text_columns.split(",") if column.strip()]
            if text_columns
            else None
        )
        preprocessed = load_and_preprocess(
            file_bytes,
            target=target,
            sensitive=sensitive,
            prediction_column=prediction_column,
            time_column=time_column,
            text_columns=text_column_list,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    audit_id = f"AUD-{uuid.uuid4().hex[:10]}"

    intent_payload = llm_service.infer_intent(
        query=query,
        dataset_context=dataset_profile(
            preprocessed.dataframe,
            target=target,
            sensitive=sensitive,
            prediction_column=prediction_column,
        ),
    )

    legal_context = llm_service.classify_legal_context(
        query=query,
        dataset_context=dataset_profile(
            preprocessed.dataframe,
            target=target,
            sensitive=sensitive,
            prediction_column=prediction_column,
        ),
    )

    base_features = preprocessed.features.drop(columns=preprocessed.text_columns, errors="ignore")
    if preprocessed.time_column and preprocessed.time_column in base_features.columns:
        base_features = base_features.drop(columns=[preprocessed.time_column])
    base_numeric_features = [
        column for column in preprocessed.numeric_features if column in base_features.columns
    ]
    base_categorical_features = [
        column for column in preprocessed.categorical_features if column in base_features.columns
    ]

    temporal_numeric_features = [
        column
        for column in base_numeric_features
        if preprocessed.time_column is None or column != preprocessed.time_column
    ]
    temporal_categorical_features = [
        column
        for column in base_categorical_features
        if preprocessed.time_column is None or column != preprocessed.time_column
    ]

    analysis_features = base_features
    analysis_numeric_features = list(base_numeric_features)
    analysis_categorical_features = list(base_categorical_features)
    analysis_warnings: list[str] = []
    if preprocessed.text_columns:
        try:
            text_embeddings = encode_text_columns(preprocessed.dataframe, preprocessed.text_columns)
            analysis_features = pd.concat([analysis_features, text_embeddings], axis=1)
            analysis_numeric_features.extend(text_embeddings.columns.tolist())
        except Exception:
            analysis_warnings.append(
                "Text embeddings could not be generated; text columns were excluded from fairness scoring."
            )

    if analysis_features.empty:
        raise HTTPException(
            status_code=400,
            detail="No usable feature columns after preprocessing and text encoding.",
        )

    try:
        fairness_result = compute_fairness(
            features=analysis_features,
            labels=preprocessed.labels,
            predictions=preprocessed.prediction_values,
            sensitive_values=preprocessed.sensitive_values,
            numeric_features=analysis_numeric_features,
            categorical_features=analysis_categorical_features,
            target_binarization_threshold=target_binarization_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    fairness_result.warnings.extend(analysis_warnings)

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
        bsi_score=fairness_result.bsi_score,
        suggestions=suggestions,
        group_metrics=fairness_result.group_metrics,
        audit_mode=fairness_result.audit_mode,
        reliability_note=fairness_result.reliability_note,
        target_transformation=fairness_result.target_transformation,
        warnings=fairness_result.warnings,
    )

    risk_tier = classify_risk(fairness_result.bsi_score)
    proxy_features = trace_bias_proxies(base_features, preprocessed.sensitive_values)
    temporal_drift = analyze_temporal_drift(
        dataframe=preprocessed.dataframe,
        time_column=preprocessed.time_column,
        target=target,
        sensitive=sensitive,
        prediction_column=prediction_column,
        numeric_features=temporal_numeric_features,
        categorical_features=temporal_categorical_features,
        target_binarization_threshold=target_binarization_threshold,
    )
    try:
        text_bias = analyze_text_bias(
            dataframe=preprocessed.dataframe,
            text_columns=preprocessed.text_columns,
            sensitive_values=preprocessed.sensitive_values,
        )
    except Exception:
        text_bias = analyze_text_bias(
            dataframe=preprocessed.dataframe,
            text_columns=[],
            sensitive_values=preprocessed.sensitive_values,
        )
        fairness_result.warnings.append(
            "Text bias analysis failed; language findings are unavailable for this dataset."
        )

    counterfactuals = llm_service.generate_counterfactuals(
        group_metrics=fairness_result.group_metrics,
        domain=legal_context.get("domain", "general"),
    )

    audit_narrative = llm_service.generate_audit_narrative(
        legal_context=legal_context,
        bsi_score=fairness_result.bsi_score,
        risk_tier=risk_tier.__dict__,
        proxy_features=[proxy.__dict__ for proxy in proxy_features],
        temporal_drift=temporal_drift.__dict__,
        text_bias=text_bias.__dict__,
        suggestions=suggestions,
        counterfactuals=counterfactuals,
    )

    simulations = simulate_mitigation(
        dataframe=preprocessed.dataframe,
        target=target,
        sensitive=sensitive,
        prediction_column=prediction_column,
        numeric_features=temporal_numeric_features,
        categorical_features=temporal_categorical_features,
        target_binarization_threshold=target_binarization_threshold,
    )

    audit_report = {
        "audit_id": audit_id,
        "mode": fairness_result.audit_mode,
        "verdict": "bias_detected" if fairness_result.bias else "no_severe_bias",
        "metrics": {
            "dp_diff": fairness_result.dp_diff,
            "eo_diff": fairness_result.eo_diff,
            "bsi_score": fairness_result.bsi_score,
        },
        "group_metrics": fairness_result.group_metrics,
        "warnings": fairness_result.warnings,
        "target_transformation": fairness_result.target_transformation,
        "reliability_note": fairness_result.reliability_note,
        "risk_tier": risk_tier.__dict__,
        "proxy_features": [proxy.__dict__ for proxy in proxy_features],
        "temporal_drift": temporal_drift.__dict__,
        "text_bias": text_bias.__dict__,
        "legal_context": legal_context,
        "audit_narrative": audit_narrative,
        "mitigation_simulation": [simulation.__dict__ for simulation in simulations],
    }

    return AnalyzeResponse(
        bias=fairness_result.bias,
        audit_id=audit_id,
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        bsi_score=fairness_result.bsi_score,
        risk_tier=risk_tier.__dict__,
        proxy_features=[proxy.__dict__ for proxy in proxy_features],
        temporal_drift=temporal_drift.__dict__,
        text_bias=text_bias.__dict__,
        legal_context=legal_context,
        audit_narrative={
            **audit_narrative,
            "counterfactuals": counterfactuals,
        },
        group_metrics=fairness_result.group_metrics,
        suggestions=suggestions,
        explanation=explanation_payload["explanation"],
        summary=explanation_payload["summary"],
        report_text=explanation_payload["report_text"],
        intent=intent_payload["intent"],
        domain=legal_context.get("domain", intent_payload["domain"]),
        audit_mode=fairness_result.audit_mode,
        warnings=fairness_result.warnings,
        target_transformation=fairness_result.target_transformation,
        audit_report=audit_report,
    )


@router.post("/reweighted-csv")
async def reweighted_csv(
    file: UploadFile = File(...),
    sensitive: str = Form(...),
    target: str = Form(...),
) -> StreamingResponse:
    if not _is_supported_csv_upload(file):
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
    target_binarization_threshold: float | None = Form(None),
) -> AnalyzeResponse:
    if not _is_supported_csv_upload(file):
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
        target_binarization_threshold=target_binarization_threshold,
    )

    audit_id = f"AUD-{uuid.uuid4().hex[:10]}"

    suggestions = suggest_mitigations(
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        bias_detected=fairness_result.bias,
    )

    explanation_payload = llm_service.generate_explanation(
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        bsi_score=fairness_result.bsi_score,
        suggestions=suggestions,
        group_metrics=fairness_result.group_metrics,
        audit_mode=fairness_result.audit_mode,
        reliability_note=fairness_result.reliability_note,
        target_transformation=fairness_result.target_transformation,
        warnings=fairness_result.warnings,
    )

    risk_tier = classify_risk(fairness_result.bsi_score)
    proxy_features = trace_bias_proxies(features, combined_sensitive)
    temporal_drift = {
        "status": "not_available",
        "time_column": None,
        "early_bsi": None,
        "late_bsi": None,
        "delta": None,
    }
    text_bias = {
        "has_text_columns": False,
        "columns": [],
        "sentiment_gaps": [],
        "top_terms": [],
        "summary": "Text bias analysis not run for intersectional audits.",
    }
    legal_context = llm_service.classify_legal_context(
        query="intersectional_analysis",
        dataset_context=dataset_profile(
            cleaned_df,
            target=target,
            sensitive=f"{sensitive_a}|{sensitive_b}",
            prediction_column=prediction_column,
        ),
    )
    counterfactuals = llm_service.generate_counterfactuals(
        group_metrics=fairness_result.group_metrics,
        domain=legal_context.get("domain", "general"),
    )
    audit_narrative = llm_service.generate_audit_narrative(
        legal_context=legal_context,
        bsi_score=fairness_result.bsi_score,
        risk_tier=risk_tier.__dict__,
        proxy_features=[proxy.__dict__ for proxy in proxy_features],
        temporal_drift=temporal_drift,
        text_bias=text_bias,
        suggestions=suggestions,
        counterfactuals=counterfactuals,
    )

    audit_report = {
        "audit_id": audit_id,
        "mode": fairness_result.audit_mode,
        "verdict": "bias_detected" if fairness_result.bias else "no_severe_bias",
        "metrics": {
            "dp_diff": fairness_result.dp_diff,
            "eo_diff": fairness_result.eo_diff,
            "bsi_score": fairness_result.bsi_score,
        },
        "group_metrics": fairness_result.group_metrics,
        "warnings": fairness_result.warnings,
        "target_transformation": fairness_result.target_transformation,
        "reliability_note": fairness_result.reliability_note,
        "risk_tier": risk_tier.__dict__,
        "proxy_features": [proxy.__dict__ for proxy in proxy_features],
        "temporal_drift": temporal_drift,
        "text_bias": text_bias,
        "legal_context": legal_context,
        "audit_narrative": audit_narrative,
    }

    return AnalyzeResponse(
        bias=fairness_result.bias,
        audit_id=audit_id,
        dp_diff=fairness_result.dp_diff,
        eo_diff=fairness_result.eo_diff,
        bsi_score=fairness_result.bsi_score,
        risk_tier=risk_tier.__dict__,
        proxy_features=[proxy.__dict__ for proxy in proxy_features],
        temporal_drift=temporal_drift,
        text_bias=text_bias,
        legal_context=legal_context,
        audit_narrative={
            **audit_narrative,
            "counterfactuals": counterfactuals,
        },
        group_metrics=fairness_result.group_metrics,
        suggestions=suggestions,
        explanation=explanation_payload["explanation"],
        summary=explanation_payload["summary"],
        report_text=explanation_payload["report_text"],
        intent="intersectional_analysis",
        domain=legal_context.get("domain", "general"),
        audit_mode=fairness_result.audit_mode,
        warnings=fairness_result.warnings,
        target_transformation=fairness_result.target_transformation,
        audit_report=audit_report,
    )
