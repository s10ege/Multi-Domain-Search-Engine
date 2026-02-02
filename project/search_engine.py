# search_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from txtai import Embeddings


@dataclass(frozen=True)
class SearchResources:
    titles: list
    texts: list
    domains: list
    sources: list
    embeddings: Embeddings

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
    df = df.dropna(subset=["title", "text", "domain", "source"]).reset_index(drop=True)

    titles = df["title"].tolist()
    texts = df["text"].tolist()
    domains = df["domain"].tolist()
    sources = df["source"].tolist()

    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    embeddings.load(embeddings_path)

    return SearchResources(titles, texts, domains, sources, embeddings)
    


def search(
    resources: SearchResources,
    query: str,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
):
    # Pull more candidates than needed
    candidate_k = max(top_k * 30, 50)
    raw = resources.embeddings.search(query, candidate_k)  # [(idx, score)]

    # Normalise domain filter
    domain_set = None
    if domain is not None:
        if isinstance(domain, str):
            domain_set = {domain.strip().lower()}
        else:
            domain_set = {d.strip().lower() for d in domain}

    results = []
    for idx, score in raw:
        idx = int(idx)
        doc_domain = str(resources.domains[idx]).strip().lower()

        if domain_set is not None and doc_domain not in domain_set:
            continue

        results.append(
            {
                "title": resources.titles[idx],
                "text": resources.texts[idx],
                "domain": resources.domains[idx],
                "source": resources.sources[idx],
                "score": float(score),
                "index": idx,
            }
        )

        if len(results) >= top_k:
            break

    return results

