"""
eval_rag.py
RAG quality evaluation for SemantikML.

Runs 10 selected queries through rag_answer() and scores each response on:
  - Faithfulness (0-2): is the answer grounded in retrieved context?
  - Relevance   (0-2): does the answer address the query?

Usage (from project root):
    python evaluation/eval_rag.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, os.path.join(BACKEND_DIR, "src"))
os.chdir(BACKEND_DIR)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from search_engine import load_resources, rag_answer  # noqa: E402


# ── 10 queries for RAG evaluation (mix of domains) ────────────────────────────
RAG_QUERIES = [
    # Research (5)
    {"id": "rag_r01", "query": "transformer attention mechanism self-attention", "domain": "research"},
    {"id": "rag_r02", "query": "reinforcement learning policy gradient reward", "domain": "research"},
    {"id": "rag_r03", "query": "graph neural network node classification", "domain": "research"},
    {"id": "rag_r04", "query": "federated learning privacy distributed training", "domain": "research"},
    {"id": "rag_r05", "query": "diffusion model score matching denoising", "domain": "research"},
    # Blog (3)
    {"id": "rag_b01", "query": "how to build a neural network from scratch Python", "domain": "blog"},
    {"id": "rag_b02", "query": "gradient descent explained step by step beginner", "domain": "blog"},
    {"id": "rag_b03", "query": "overfitting regularisation dropout tutorial", "domain": "blog"},
    # Documentation (2)
    {"id": "rag_d01", "query": "scikit-learn Pipeline fit transform parameters", "domain": "documentation"},
    {"id": "rag_d02", "query": "PyTorch nn.Module forward method layer definition", "domain": "documentation"},
]


def score_rag_response(
    query: str,
    answer: str,
    context_titles: list[str],
    context_snippets: list[str],
) -> tuple[int, int, str]:
    """
    Heuristic-based scoring of a RAG response.
    Returns (faithfulness, relevance, reasoning).

    Faithfulness (0-2):
      0 = answer contradicts context or contains fabricated facts not in context
      1 = answer is partially grounded; some claims lack context support
      2 = answer is clearly grounded in context; specific details match

    Relevance (0-2):
      0 = answer does not address the query at all
      1 = answer partially addresses the query
      2 = answer directly and fully addresses the query
    """
    answer_lower = answer.lower().strip()
    query_lower = query.lower()

    # Empty / error answers
    if not answer_lower or len(answer_lower) < 30:
        return 0, 0, "No answer generated (empty or too short)."

    if "does not contain enough information" in answer_lower or "not enough information" in answer_lower:
        return 1, 0, "Model correctly acknowledged insufficient context; relevance=0 because query unanswered."

    # Extract key terms from query (non-stopwords)
    stopwords = {"how", "to", "a", "the", "is", "in", "for", "of", "and", "or", "does", "an", "what",
                 "explain", "explained", "tutorial", "guide", "step", "by", "from", "vs", "on"}
    query_terms = [t for t in query_lower.split() if t not in stopwords and len(t) > 3]

    # Relevance scoring
    terms_in_answer = sum(1 for t in query_terms if t in answer_lower)
    relevance_ratio = terms_in_answer / len(query_terms) if query_terms else 0.0

    if relevance_ratio >= 0.5:
        relevance = 2
    elif relevance_ratio >= 0.25:
        relevance = 1
    else:
        relevance = 0

    # Faithfulness scoring: check if answer references context content
    all_context = " ".join(context_snippets).lower()
    context_title_words = set()
    for t in context_titles:
        context_title_words.update(t.lower().split())

    # Count unique words from context that appear in answer
    answer_words = set(answer_lower.split())
    context_words = set(all_context.split())
    overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0.0

    if overlap >= 0.15:
        faithfulness = 2
    elif overlap >= 0.08:
        faithfulness = 1
    else:
        faithfulness = 0

    reasoning = (
        f"Relevance: {terms_in_answer}/{len(query_terms)} query terms in answer (ratio={relevance_ratio:.2f}). "
        f"Faithfulness: {overlap:.2%} answer words overlap with context."
    )
    return faithfulness, relevance, reasoning


def main() -> None:
    print("Loading search engine...")
    resources = load_resources()
    print(f"  {len(resources.titles)} documents loaded.")

    results = []

    for q in RAG_QUERIES:
        print(f"\n[{q['id']}] {q['query'][:60]}")

        t0 = time.perf_counter()
        output = rag_answer(resources, q["query"], top_k=3, domain=q["domain"], max_new_tokens=150)
        latency = time.perf_counter() - t0

        answer = output.get("answer", "") or ""
        retrieved = output.get("results", [])
        context_titles = [r.get("title", "") for r in retrieved]
        context_snippets = [r.get("text", "")[:300] for r in retrieved]

        faithfulness, relevance, reasoning = score_rag_response(
            q["query"], answer, context_titles, context_snippets
        )

        print(f"  Answer ({len(answer)} chars): {answer[:200]}...")
        print(f"  Context sources: {context_titles}")
        print(f"  Faithfulness={faithfulness}/2  Relevance={relevance}/2")
        print(f"  Scoring: {reasoning}")
        print(f"  Latency: {latency:.2f}s")

        results.append(
            {
                "id": q["id"],
                "query": q["query"],
                "domain": q["domain"],
                "answer": answer,
                "answer_length": len(answer),
                "context_titles": context_titles,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "scoring_reasoning": reasoning,
                "latency_s": round(latency, 3),
            }
        )

    # Aggregate
    n = len(results)
    mean_faith = sum(r["faithfulness"] for r in results) / n
    mean_rel = sum(r["relevance"] for r in results) / n
    mean_lat = sum(r["latency_s"] for r in results) / n

    print("\n" + "=" * 70)
    print(f" RAG Evaluation Summary ({n} queries)")
    print("=" * 70)
    print(f"  Mean Faithfulness : {mean_faith:.2f} / 2.0")
    print(f"  Mean Relevance    : {mean_rel:.2f} / 2.0")
    print(f"  Mean Latency      : {mean_lat:.1f} s")

    by_domain: dict[str, list] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r)
    for dom, dom_results in sorted(by_domain.items()):
        dm_f = sum(r["faithfulness"] for r in dom_results) / len(dom_results)
        dm_r = sum(r["relevance"] for r in dom_results) / len(dom_results)
        print(f"  {dom:<16}: Faithfulness={dm_f:.2f}  Relevance={dm_r:.2f}  (n={len(dom_results)})")

    output_data = {
        "aggregate": {
            "n_queries": n,
            "mean_faithfulness": round(mean_faith, 4),
            "mean_relevance": round(mean_rel, 4),
            "mean_latency_s": round(mean_lat, 3),
            "scale": "0 = poor, 1 = partial, 2 = good",
        },
        "by_domain": {
            dom: {
                "mean_faithfulness": round(
                    sum(r["faithfulness"] for r in dom_results) / len(dom_results), 4
                ),
                "mean_relevance": round(
                    sum(r["relevance"] for r in dom_results) / len(dom_results), 4
                ),
            }
            for dom, dom_results in by_domain.items()
        },
        "per_query": results,
    }

    out_path = os.path.join(SCRIPT_DIR, "rag_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nRAG results saved to: {out_path}")


if __name__ == "__main__":
    main()
