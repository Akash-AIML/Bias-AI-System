from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def encode_text_columns(df: pd.DataFrame, text_columns: list[str]) -> pd.DataFrame:
    if not text_columns:
        return pd.DataFrame(index=df.index)

    merged_text = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    model = _get_model()
    embeddings = model.encode(merged_text, show_progress_bar=False, convert_to_numpy=True)

    embedding_columns = [f"text_emb_{idx}" for idx in range(embeddings.shape[1])]
    return pd.DataFrame(embeddings, columns=embedding_columns, index=df.index)
