from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class TextBiasResult:
    has_text_columns: bool
    columns: list[str]
    sentiment_gaps: list[dict[str, object]]
    top_terms: list[dict[str, object]]
    summary: str


def analyze_text_bias(
    dataframe: pd.DataFrame,
    text_columns: list[str],
    sensitive_values: pd.Series,
    max_terms: int = 6,
) -> TextBiasResult:
    if not text_columns:
        return TextBiasResult(
            has_text_columns=False,
            columns=[],
            sentiment_gaps=[],
            top_terms=[],
            summary="No free-form text columns detected.",
        )

    analyzer = SentimentIntensityAnalyzer()
    sentiment_gaps: list[dict[str, object]] = []
    top_terms: list[dict[str, object]] = []

    group_counts = sensitive_values.astype(str).value_counts()
    top_groups = group_counts.index.tolist()[:2]

    for column in text_columns:
        text_series = dataframe[column].fillna("").astype(str)
        sentiments = text_series.apply(lambda text: analyzer.polarity_scores(text)["compound"])

        if len(top_groups) >= 2:
            group_a, group_b = top_groups[0], top_groups[1]
            mean_a = sentiments[sensitive_values.astype(str) == group_a].mean()
            mean_b = sentiments[sensitive_values.astype(str) == group_b].mean()
            gap = float(mean_a - mean_b)
            sentiment_gaps.append(
                {
                    "column": column,
                    "group_a": str(group_a),
                    "group_b": str(group_b),
                    "gap": round(gap, 4),
                }
            )

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(text_series)
        except ValueError:
            continue

        feature_names = vectorizer.get_feature_names_out()
        for group in top_groups:
            group_mask = sensitive_values.astype(str) == group
            if group_mask.sum() == 0:
                continue
            group_mean = tfidf_matrix[group_mask].mean(axis=0)
            group_scores = group_mean.A1
            top_indices = group_scores.argsort()[-max_terms:][::-1]
            terms = [feature_names[idx] for idx in top_indices if group_scores[idx] > 0]
            top_terms.append(
                {
                    "column": column,
                    "group": str(group),
                    "terms": terms,
                }
            )

    summary = "Text bias scan completed across free-form columns."
    return TextBiasResult(
        has_text_columns=True,
        columns=text_columns,
        sentiment_gaps=sentiment_gaps,
        top_terms=top_terms,
        summary=summary,
    )
