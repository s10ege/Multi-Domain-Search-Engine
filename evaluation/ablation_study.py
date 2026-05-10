"""
ablation_study.py
Ablation study for SemantikML — evaluates four retrieval configurations.

Configurations
--------------
1. BM25-only        — sparse keyword retrieval only, no reranking
2. Dense-only       — dense bi-encoder (ANN) retrieval only, no reranking
3. Hybrid (no CE)   — BM25 + dense hybrid retrieval, no cross-encoder reranking
4. Full system      — BM25 + dense + cross-encoder reranking (production baseline)

All four configurations use domain-filtered retrieval (top_k=10) over the same
25 queries and relevance labels as the main eval_search.py evaluation.

Usage (from project root):
    python evaluation/ablation_study.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Path & working-directory setup (mirrors eval_search.py) ───────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, os.path.join(BACKEND_DIR, "src"))
sys.path.insert(0, SCRIPT_DIR)  # so we can import from eval_search
os.chdir(BACKEND_DIR)

# Metric helpers — imported from eval_search to avoid duplication.
# eval_search also wraps sys.stdout for UTF-8 at import time; do not wrap again.
from eval_search import aggregate_metrics, compute_metrics  # noqa: E402

from search_engine import (  # noqa: E402
    SIMILARITY_MODEL_PATH,
    SearchResources,
    _domain_clause,
    _get_similarity_pipeline,
    load_resources,
    search,
)


# ── Shared candidate retrieval with configurable hybrid weight ─────────────────


def _candidates(
    resources: SearchResources,
    query: str,
    candidate_k: int,
    domain: str | None,
    weights: float,
) -> list[dict]:
    """
    Retrieve candidates via txtai SQL search with a given hybrid weight.

    weights=0.0 → BM25-only (sparse)
    weights=1.0 → Dense-only (ANN)
    weights=0.5 → Hybrid (default balance)
    """
    candidate_k = int(candidate_k)
    sql = f"SELECT id, score FROM txtai WHERE similar(:query, {candidate_k})"
    clause = _domain_clause(domain)
    if clause:
        sql += f" AND {clause}"
    sql += f" LIMIT {candidate_k}"

    raw = resources.embeddings.search(
        sql,
        parameters={"query": query},
        weights=weights,
    )

    results = []
    for row in raw:
        if isinstance(row, dict):
            idx = int(row["id"])
            score = float(row.get("score", 0.0))
        else:
            idx, score = int(row[0]), float(row[1])

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


# ── Three ablation search functions ───────────────────────────────────────────


def search_bm25_only(
    resources: SearchResources,
    query: str,
    top_k: int = 10,
    domain: str | None = None,
) -> list[dict]:
    """BM25-only retrieval; no cross-encoder reranking."""
    RERANK_BUDGET = max(top_k * 4, 20)
    candidates = _candidates(resources, query, RERANK_BUDGET, domain, weights=0.0)
    return candidates[:top_k]


def search_dense_only(
    resources: SearchResources,
    query: str,
    top_k: int = 10,
    domain: str | None = None,
) -> list[dict]:
    """Dense (ANN) retrieval only; no cross-encoder reranking."""
    RERANK_BUDGET = max(top_k * 4, 20)
    candidates = _candidates(resources, query, RERANK_BUDGET, domain, weights=1.0)
    return candidates[:top_k]


def search_hybrid_no_rerank(
    resources: SearchResources,
    query: str,
    top_k: int = 10,
    domain: str | None = None,
) -> list[dict]:
    """Hybrid BM25+dense retrieval; bi-encoder order preserved, no cross-encoder."""
    RERANK_BUDGET = max(top_k * 4, 20)
    candidates = _candidates(resources, query, RERANK_BUDGET, domain, weights=0.5)
    return candidates[:top_k]


# ── Evaluation loop ────────────────────────────────────────────────────────────


def run_ablation_eval(
    resources: SearchResources,
    queries: list[dict],
    search_fn,
    top_k: int = 10,
) -> list[dict]:
    """Run all queries through a given search function with domain filter applied."""
    results = []
    for q in queries:
        domain = q["domain"]
        relevant = set(q["relevant_indices"])

        t0 = time.perf_counter()
        hits = search_fn(resources, q["query"], top_k=top_k, domain=domain)
        latency = time.perf_counter() - t0

        retrieved = [r["index"] for r in hits]
        metrics = compute_metrics(retrieved, relevant, latency)

        results.append(
            {
                "id": q["id"],
                "query": q["query"],
                "domain": q["domain"],
                "num_relevant_gt": len(relevant),
                "num_retrieved": len(retrieved),
                "retrieved_indices": retrieved,
                **metrics,
            }
        )
    return results


# ── Summary table ──────────────────────────────────────────────────────────────

_CONFIGS = [
    "BM25-only",
    "Dense-only",
    "Hybrid (no reranking)",
    "Full system (Hybrid + CE)",
]


def print_summary_table(aggregates: dict[str, dict]) -> None:
    """Print the dissertation-ready summary table to stdout."""
    header = (
        f"{'Configuration':<28} | {'P@5':>5} | {'P@10':>5} | "
        f"{'R@10':>5} | {'MRR':>5} | {'nDCG@10':>7} | {'Latency(ms)':>11}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name in _CONFIGS:
        agg = aggregates[name]
        latency_ms = agg["mean_latency_s"] * 1000
        print(
            f"{name:<28} | "
            f"{agg['mean_p_at_5']:>5.3f} | "
            f"{agg['mean_p_at_10']:>5.3f} | "
            f"{agg['mean_r_at_10']:>5.3f} | "
            f"{agg['mean_mrr']:>5.3f} | "
            f"{agg['mean_ndcg_at_10']:>7.3f} | "
            f"{latency_ms:>11.1f}"
        )
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    gt_path = os.path.join(SCRIPT_DIR, "eval_queries.json")
    with open(gt_path, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} queries.")
    print("Loading search engine resources (this may take a moment)...")
    resources = load_resources()
    print(f"  {len(resources.titles)} documents loaded.\n")

    # Pre-warm the cross-encoder so its load time does not inflate Full system latency
    print("Pre-warming cross-encoder...")
    _get_similarity_pipeline(SIMILARITY_MODEL_PATH)
    print("  Cross-encoder ready.\n")

    configs: list[tuple[str, object]] = [
        ("BM25-only", search_bm25_only),
        ("Dense-only", search_dense_only),
        ("Hybrid (no reranking)", search_hybrid_no_rerank),
        ("Full system (Hybrid + CE)", search),
    ]

    per_query_results: dict[str, list[dict]] = {}
    aggregates: dict[str, dict] = {}

    for name, fn in configs:
        print(f"Running: {name} ...")
        results = run_ablation_eval(resources, queries, fn, top_k=10)
        agg = aggregate_metrics(results)
        per_query_results[name] = results
        aggregates[name] = agg
        latency_ms = agg["mean_latency_s"] * 1000
        print(
            f"  P@5={agg['mean_p_at_5']:.3f}  P@10={agg['mean_p_at_10']:.3f}"
            f"  R@10={agg['mean_r_at_10']:.3f}  MRR={agg['mean_mrr']:.3f}"
            f"  nDCG@10={agg['mean_ndcg_at_10']:.3f}  latency={latency_ms:.1f}ms"
        )

    # ── Print final summary table ──────────────────────────────────────────────
    print("\n\n=== ABLATION STUDY — SUMMARY TABLE (domain-filtered, top_k=10) ===")
    print_summary_table(aggregates)

    # ── Save full per-query results ────────────────────────────────────────────
    output = {
        name: {
            "aggregate": aggregates[name],
            "per_query": per_query_results[name],
        }
        for name in aggregates
    }
    out_path = os.path.join(SCRIPT_DIR, "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull per-query results saved to: {out_path}")


if __name__ == "__main__":
    main()
