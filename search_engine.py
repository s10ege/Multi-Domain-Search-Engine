# search_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import pandas as pd
from txtai import Embeddings, RAG
from txtai.pipeline import Summary, Similarity


@dataclass(frozen=True)
class SearchResources:
    titles: list
    texts: list
    domains: list
    sources: list
    embeddings: Embeddings

def load_resources(
    csv_path: str = "final_data.csv",
    embeddings_path: str = "embeddings",
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
    domains = df["domain"].astype(str).str.strip().str.lower().tolist()
    sources = df["source"].tolist()

    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "content": True,
            "scoring": {"method": "bm25", "terms": True},
            "function": "sentence",
        }
    )
    path = Path(embeddings_path)
    if not path.exists() and embeddings_path == "embeddings":
        tar = Path("embeddings.tar.gz")
        if tar.exists():
            path = tar

    embeddings.load(str(path))

    return SearchResources(titles, texts, domains, sources, embeddings)


def close_resources(resources: SearchResources) -> None:
    resources.embeddings.close()
    


def _domain_clause(domain: Optional[Union[str, list[str]]]) -> Optional[str]:
    if domain is None:
        return None

    if isinstance(domain, str):
        domains = [domain]
    else:
        domains = list(domain)

    normalized = [d.strip().lower() for d in domains if str(d).strip()]
    if not normalized:
        return None

    escaped = [d.replace("'", "''") for d in normalized]
    if len(escaped) == 1:
        return f"domain = '{escaped[0]}'"
    quoted = ", ".join(f"'{d}'" for d in escaped)
    return f"domain IN ({quoted})"


def _search_candidates(
    resources: SearchResources,
    query: str,
    candidate_k: int,
    domain: Optional[Union[str, list[str]]] = None,
):
    candidate_k = int(candidate_k)
    sql = f"SELECT id, score FROM txtai WHERE similar(:query, {candidate_k})"
    clause = _domain_clause(domain)
    if clause:
        sql += f" AND {clause}"
    sql += f" LIMIT {candidate_k}"

    raw = resources.embeddings.search(
        sql,
        parameters={"query": query},
    )

    results = []
    for row in raw:
        if isinstance(row, dict):
            idx = int(row["id"])
            score = float(row.get("score", 0.0))
        else:
            idx, score = row
            idx = int(idx)
            score = float(score)

        results.append(
            {
                "title": resources.titles[idx],
                "text": resources.texts[idx],
                "domain": resources.domains[idx],
                "source": resources.sources[idx],
                "score": score,
                "index": idx,
            }
        )

    return results


def _normalize_rag_answer(output: Any) -> str:
    if isinstance(output, list):
        if not output:
            return ""
        output = output[0]

    if isinstance(output, tuple):
        if len(output) >= 2:
            return str(output[1])
        return str(output)

    if isinstance(output, dict):
        if "answer" in output:
            return str(output["answer"])
        if "text" in output:
            return str(output["text"])
        return str(output)

    return str(output)


def search(
    resources: SearchResources,
    query: str,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
):
    # Pull more candidates than needed
    candidate_k = max(top_k * 30, 50)

    results = _search_candidates(resources, query, candidate_k, domain=domain)
    return results[:top_k]


def rag_answer(
    resources: SearchResources,
    question: str,
    top_k: int = 3,
    domain: Optional[Union[str, list[str]]] = None,
    model_path: str = "Qwen/Qwen3-0.6B",
    template: Optional[str] = None,
    max_context_chars: int = 1500,
    max_new_tokens: int = 256,
    max_length: int = 2048,
):
    results = search(resources, question, top_k=top_k, domain=domain)
    if not results:
        return {"answer": "", "results": []}

    context_texts = [r["text"][:max_context_chars] for r in results]
    prompt = template or (
        "Answer the following question using the provided context.\n\n"
        "Question:\n{question}\n\n"
        "Context:\n{context}"
    )

    rag = RAG(resources.embeddings, model_path, template=prompt, context=top_k)
    answer = rag(
        question,
        texts=context_texts,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )

    return {"answer": _normalize_rag_answer(answer), "results": results}


def summarize_results(
    resources: SearchResources,
    query: str,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
    model_path: Optional[str] = None,
    minlength: Optional[int] = None,
    maxlength: Optional[int] = None,
):
    results = search(resources, query, top_k=top_k, domain=domain)
    if not results:
        return {"summary": "", "results": []}

    combined = "\n\n".join(r["text"] for r in results)
    try:
        summarizer = Summary(path=model_path) if model_path else Summary()
        summary = summarizer(combined, minlength=minlength, maxlength=maxlength)
    except Exception:
        text = combined.strip()
        if not text:
            summary = ""
        else:
            sentences = text.replace("\n", " ").split(".")
            summary = ". ".join(s.strip() for s in sentences if s.strip())
            summary = summary[:1000].rstrip()
            if summary and not summary.endswith("."):
                summary += "."

    return {"summary": summary, "results": results}


def more_like_this(
    resources: SearchResources,
    index: int,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
):
    seed_text = resources.texts[index]
    results = search(resources, seed_text, top_k=top_k + 1, domain=domain)
    filtered = [r for r in results if r["index"] != index]
    return filtered[:top_k]


def rerank_search(
    resources: SearchResources,
    query: str,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    factor: int = 10,
):
    candidate_k = max(top_k * factor, top_k)
    candidates = _search_candidates(resources, query, candidate_k, domain=domain)
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    reranker = Similarity(path=rerank_model, crossencode=True)
    scores = reranker(query, texts, multilabel=None)

    scored = []
    for idx, score in scores:
        item = dict(candidates[idx])
        item["rerank_score"] = float(score)
        scored.append(item)

    scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return scored[:top_k]


def cluster_results(
    resources: SearchResources,
    query: str,
    top_k: int = 10,
    domain: Optional[Union[str, list[str]]] = None,
    clusters: int = 3,
):
    results = search(resources, query, top_k=top_k, domain=domain)
    if not results:
        return []

    texts = [r["text"] for r in results]

    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for clustering") from exc

    vectors = resources.embeddings.batchtransform(texts)
    k = min(max(1, clusters), len(results))

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(vectors)

    grouped = {}
    for label, item in zip(labels, results):
        grouped.setdefault(int(label), []).append(item)

    return [
        {"cluster": cluster_id, "items": items}
        for cluster_id, items in sorted(grouped.items())
    ]

