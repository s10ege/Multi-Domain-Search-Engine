# search_engine.py
# Core search logic, RAG pipeline, and resource management for the application.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import pandas as pd
from txtai import Embeddings, RAG
from txtai.pipeline import Summary, Similarity

SIMILARITY_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _get_similarity_pipeline(model_path: str) -> Similarity:
    return Similarity(model_path, crossencode=True)

# Frozen dataclass acts as an immutable container for global application state
# (index, database, and content lists).
@dataclass(frozen=True)
class SearchResources:
    titles: list
    texts: list
    domains: list
    sources: list
    embeddings: Embeddings
    rag: RAG

# Loads the CSV dataset and the pre-built txtai index from disk.
# Handles data cleaning and ensures the embeddings model is initialized.
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

    # Load embeddings — config must match build_embeddings.py
    embeddings = Embeddings(
        {
            "path": "BAAI/bge-small-en-v1.5",
            "content": True,
            "scoring": {"method": "bm25", "terms": True},
            "instructions": {
                "query": "Represent this sentence for searching relevant passages: ",
            },
        }
    )
    path = Path(embeddings_path)
    if not path.exists() and embeddings_path == "embeddings":
        tar = Path("embeddings.tar.gz")
        if tar.exists():
            path = tar

    embeddings.load(str(path))

    # Initialize RAG once to avoid reloading the model on each call
    # Use a clear template for consistent outputs
    template = (
        "You are a helpful assistant. Answer the question below using only the information provided in the context.\n\n"
        "Instructions:\n"
        "- Write in a clear, professional tone\n"
        "- Provide a clear, informative explanation\n"
        "- Write at least 3-4 complete sentences ending with proper punctuation\n"
        "- Always finish your sentences completely - never cut off mid-sentence\n"
        "- Do not include reasoning steps or thought process\n"
        "- Stay factual and based on the context\n"
        "- If the context does not contain enough information to answer the question, state that clearly\n"
        "- When making specific claims, reference the title of the source in parentheses\n\n"
        "Question: {question}\n\n"
        "Context: {context}\n\n"
        "Answer:"
    )
    rag = RAG(
        embeddings,
        "Qwen/Qwen3-0.6B",
        template=template,
    )

    return SearchResources(titles, texts, domains, sources, embeddings, rag)


# Closes the embeddings index to release memory and file handles.
def close_resources(resources: SearchResources) -> None:
    resources.embeddings.close()
    

# Helper: Constructs SQL WHERE clauses for filtering by domain.
# Only allows selecting ONE domain at a time. If multiple domains are provided,
# only the first one will be used.
def _domain_clause(domain: Optional[Union[str, list[str]]]) -> Optional[str]:
    if domain is None:
        return None

    # Convert to list if string, then take only the first domain
    if isinstance(domain, str):
        selected_domain = domain
    else:
        domains = list(domain)
        if not domains:
            return None
        # Only use the first domain from the list
        selected_domain = domains[0]

    # Normalize and validate the single domain
    normalized = str(selected_domain).strip().lower()
    if not normalized:
        return None

    # Escape single quotes for SQL safety
    escaped = normalized.replace("'", "''")
    return f"domain = '{escaped}'"


# Helper: Executes the low-level txtai SQL query.
# Combines semantic search ("similar()") with structured metadata filtering.
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

    selected_domain = None
    if domain is not None:
        if isinstance(domain, str):
            selected_domain = domain
        else:
            domains = list(domain)
            if domains:
                selected_domain = domains[0]

    if selected_domain is not None:
        selected_domain = str(selected_domain).strip().lower()
        if not selected_domain:
            selected_domain = None

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

# 1. Standard Semantic Search
# Retrieves candidates via bi-encoder, then re-ranks with cross-encoder for
# better top-k precision (typically +5–15% MRR@10).
def search(
    resources: SearchResources,
    query: str,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
):
    # Retrieve a modest candidate pool, then let the cross-encoder sort them.
    RERANK_BUDGET = max(top_k * 4, 20)
    candidates = _search_candidates(resources, query, RERANK_BUDGET, domain=domain)
    if not candidates:
        return []

    similarity = _get_similarity_pipeline(SIMILARITY_MODEL_PATH)
    MAX_LEN = 500  # stay within the cross-encoder's 512-token limit
    truncated = [r["text"][:MAX_LEN] for r in candidates]
    scores = similarity(query[:MAX_LEN], truncated)

    reranked = []
    for candidate_idx, score in scores:
        item = candidates[int(candidate_idx)]
        reranked.append({**item, "score": float(score)})

    return reranked[:top_k]

# Normalizes the diverse output formats of RAG pipelines into a single string.
def _normalize_rag_answer(output: Any) -> str:
    if isinstance(output, list):
        if not output:
            return ""
        output = output[0]

    if isinstance(output, tuple):
        if len(output) >= 2:
            return str(output[1])
        elif len(output) == 1:
            return str(output[0])
        return str(output)

    if isinstance(output, dict):
        if "answer" in output:
            return str(output["answer"])
        if "text" in output:
            return str(output["text"])
        return str(output)

    return str(output)

# 2. RAG (Retrieval Augmented Generation)
# Retrieves context docs first, then feeds them to an LLM to answer the question.
def rag_answer(
    resources: SearchResources,
    question: str,
    top_k: int = 3,
    domain: Optional[Union[str, list[str]]] = None,
    model_path: str = "Qwen/Qwen3-0.6B",
    template: Optional[str] = None,
    max_context_chars: int = 3000,
    max_new_tokens: int = 512,
):
    results = search(resources, question, top_k=top_k, domain=domain)
    if not results:
        return {"answer": "", "results": []}

    context_texts = [
        f"Title: {r['title']}\n{r['text'][:max_context_chars]}"
        for r in results
    ]

    # Use pre-initialized RAG instance with sampling for more natural outputs.
    # Qwen3 model card recommends temperature=0.7, top_p=0.8 for non-thinking mode.
    answer = resources.rag(
        question,
        texts=context_texts,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

    return {"answer": _normalize_rag_answer(answer), "results": results}



# 3. Summarization
# Combines top search results and generates a concise summary.
# Currently Not used.
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

# 4. More Like This
# Uses the text of an existing search result as a query to find similar items.
def more_like_this(
    resources: SearchResources,
    index: int,
    top_k: int = 5,
    domain: Optional[Union[str, list[str]]] = None,
):
    seed_text = resources.texts[index]
    candidate_k = max(top_k * 10, top_k + 10)
    candidates = search(resources, seed_text, top_k=candidate_k, domain=domain)
    candidates = [r for r in candidates if r["index"] != index]

    enriched_candidates = []
    for item in candidates:
        candidate = dict(item)
        if "text" not in candidate or candidate["text"] is None:
            candidate_idx = candidate.get("index")
            if candidate_idx is not None:
                candidate_idx = int(candidate_idx)
                if 0 <= candidate_idx < len(resources.texts):
                    candidate["text"] = resources.texts[candidate_idx]
                else:
                    candidate["text"] = ""
            else:
                candidate["text"] = ""
        enriched_candidates.append(candidate)
    candidates = enriched_candidates

    if not candidates:
        return []

    similarity = _get_similarity_pipeline(SIMILARITY_MODEL_PATH)

    # Truncate texts to avoid model errors with max sequence length
    MAX_TEXT_LENGTH = 500  # Keep it under the 512 token limit
    truncated_seed = seed_text[:MAX_TEXT_LENGTH]
    truncated_candidates = [r["text"][:MAX_TEXT_LENGTH] for r in candidates]

    scores = similarity(truncated_seed, truncated_candidates)

    results = []
    for candidate_idx, score in scores:
        item = candidates[int(candidate_idx)]
        results.append({**item, "score": float(score)})

    return results[:top_k]