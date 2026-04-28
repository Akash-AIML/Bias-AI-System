from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_fairness_report(payload: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER, title="Algorithmic Accountability Certificate")
    styles = getSampleStyleSheet()

    elements: list[Any] = []
    elements.append(Paragraph("Algorithmic Accountability Certificate", styles["Title"]))
    elements.append(Spacer(1, 12))

    dataset = payload.get("dataset_summary", {})
    accountability = payload.get("accountability_summary", {})
    elements.append(Paragraph("Dataset Summary", styles["Heading2"]))
    dataset_table = Table(
        [
            ["Organization", dataset.get("org_name", "-")],
            ["Dataset", dataset.get("dataset_name", "-")],
            ["File", dataset.get("file", "-")],
            ["Target", dataset.get("target", "-")],
            ["Sensitive", dataset.get("sensitive", "-")],
            ["Prediction Column", dataset.get("prediction_column", "-")],
            ["Audit Mode", payload.get("audit_mode", "-")],
            ["Audit ID", accountability.get("audit_id", "-")],
            ["Audit Date", dataset.get("uploaded_at", "-")],
        ]
    )
    dataset_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    elements.append(dataset_table)
    elements.append(Spacer(1, 12))

    verdict = payload.get("verdict", "unknown")
    elements.append(Paragraph(f"Audit Verdict: <b>{verdict}</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    dp_diff = payload.get("dp_diff", 0)
    eo_diff = payload.get("eo_diff", 0)
    bsi_score = payload.get("bsi_score", 0)
    risk_tier = payload.get("risk_tier", {})
    elements.append(Paragraph(f"Bias Severity Index: {bsi_score}", styles["BodyText"]))
    elements.append(Paragraph(
        f"Risk Tier: {risk_tier.get('label', '-')}",
        styles["BodyText"],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Demographic Parity Difference: {dp_diff}", styles["BodyText"]))
    elements.append(Paragraph("Meaning: difference in selection rates across groups.", styles["BodyText"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Equalized Odds Difference: {eo_diff}", styles["BodyText"]))
    elements.append(Paragraph("Meaning: gap in error rates across groups.", styles["BodyText"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Proxy Discrimination Findings", styles["Heading2"]))
    proxy_features = payload.get("proxy_features", [])
    proxy_table_data = [["Feature", "Correlation", "Direction"]]
    for proxy in proxy_features:
        proxy_table_data.append([
            str(proxy.get("feature", "-")),
            str(proxy.get("correlation", "-")),
            str(proxy.get("direction", "-")),
        ])
    proxy_table = Table(proxy_table_data, hAlign="LEFT")
    proxy_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    elements.append(proxy_table)
    elements.append(Spacer(1, 12))

    temporal_drift = payload.get("temporal_drift", {})
    elements.append(Paragraph("Temporal Drift", styles["Heading2"]))
    elements.append(
        Paragraph(
            f"Status: {temporal_drift.get('status', '-')}",
            styles["BodyText"],
        )
    )
    elements.append(
        Paragraph(
            f"Early BSI: {temporal_drift.get('early_bsi', '-')}, Late BSI: {temporal_drift.get('late_bsi', '-')}",
            styles["BodyText"],
        )
    )
    elements.append(Spacer(1, 12))

    text_bias = payload.get("text_bias", {})
    elements.append(Paragraph("Language Bias Scan", styles["Heading2"]))
    elements.append(Paragraph(text_bias.get("summary", "-"), styles["BodyText"]))
    elements.append(Spacer(1, 12))

    legal_context = payload.get("legal_context", {})
    elements.append(Paragraph("Legal Context", styles["Heading2"]))
    elements.append(Paragraph(legal_context.get("framework", "-"), styles["BodyText"]))
    elements.append(Paragraph(legal_context.get("notes", "-"), styles["BodyText"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Group Metrics", styles["Heading2"]))
    group_metrics = payload.get("group_metrics", {})
    table_data = [["Group", "Accuracy", "Selection Rate"]]
    for group, metrics in group_metrics.items():
        table_data.append([
            str(group),
            str(metrics.get("accuracy", "-")),
            str(metrics.get("selection_rate", "-")),
        ])
    metrics_table = Table(table_data, hAlign="LEFT")
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    elements.append(metrics_table)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Mitigation Suggestions", styles["Heading2"]))
    suggestions = payload.get("suggestions", [])
    for suggestion in suggestions:
        elements.append(Paragraph(f"- {suggestion}", styles["BodyText"]))
    elements.append(Spacer(1, 16))

    audit_narrative = payload.get("audit_narrative", {})
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    elements.append(Paragraph(audit_narrative.get("executive_summary", "-"), styles["BodyText"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Risk Assessment", styles["Heading2"]))
    elements.append(Paragraph(audit_narrative.get("risk_assessment", "-"), styles["BodyText"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Legal Exposure", styles["Heading2"]))
    elements.append(Paragraph(audit_narrative.get("legal_exposure", "-"), styles["BodyText"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Generated by Accountability Audit System", styles["Italic"]))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
