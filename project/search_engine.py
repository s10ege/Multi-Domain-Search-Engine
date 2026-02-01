# search_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import txtai


@dataclass(frozen=True)
class SearchResources:
    titles: list
    texts: list
    domains: list
    sources: list
    embeddings: txtai.Embeddings


def load_resources(
    csv_path: str = "final_data.csv",
    embeddings_path: str = "embeddings.tar.gz",
    encoding: str = "latin1",
) -> SearchResources:
    df = pd.read_csv(csv_path, encoding=encoding)

    # Drop Unnamed columns
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Drop rows with missing core fields (adjust as needed)
    df = df.dropna(subset=["title", "text", "domain", "source"])

    titles = df["title"].tolist()
    texts = df["text"].tolist()
    domains = df["domain"].tolist()
    sources = df["source"].tolist()

    embeddings = txtai.Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    embeddings.load(embeddings_path)

    return SearchResources(titles, texts, domains, sources, embeddings)


def search(
    resources: SearchResources,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    results = resources.embeddings.search(query, top_k)

    # results entries typically like: [(idx, score), ...]
    output: List[Dict[str, Any]] = []
    for idx, score in results:
        output.append(
            {
                "title": resources.titles[idx],
                "text": resources.texts[idx],
                "domain": resources.domains[idx],
                "source": resources.sources[idx],
                "score": float(score),
                "index": int(idx),
            }
        )
    return output
