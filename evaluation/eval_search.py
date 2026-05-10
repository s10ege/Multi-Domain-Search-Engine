"""
eval_search.py
Retrieval-quality evaluation for SemantikML.

Computes per-query and aggregate metrics:
  Precision@5, Precision@10, Recall@5, Recall@10, MRR, nDCG@10, latency

Also compares domain-filtered vs cross-domain (domain=None) retrieval.

Usage (from project root):
    python evaluation/eval_search.py
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Path & working-directory setup ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, os.path.join(BACKEND_DIR, "src"))
os.chdir(BACKEND_DIR)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from search_engine import load_resources, search  # noqa: E402


# ── Metric helpers ─────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for idx in top_k if idx in relevant)
    return hits / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant docs found in top-k. 0.0 when relevant set is empty."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for idx in top_k if idx in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """1 / rank of the first relevant result (0 if none found)."""
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance)."""
    if not relevant:
        return 0.0

    def dcg(ranked: list[int]) -> float:
        total = 0.0
        for i, idx in enumerate(ranked[:k], 1):
            if idx in relevant:
                total += 1.0 / math.log2(i + 1)
        return total

    actual_dcg = dcg(retrieved)
    ideal_dcg = dcg(list(relevant)[:k])
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_metrics(
    retrieved: list[int],
    relevant: set[int],
    latency: float,
) -> dict:
    return {
        "p_at_5": precision_at_k(retrieved, relevant, 5),
        "p_at_10": precision_at_k(retrieved, relevant, 10),
        "r_at_5": recall_at_k(retrieved, relevant, 5),
        "r_at_10": recall_at_k(retrieved, relevant, 10),
        "mrr": mean_reciprocal_rank(retrieved, relevant),
        "ndcg_at_10": ndcg_at_k(retrieved, relevant, 10),
        "latency_s": round(latency, 4),
    }


def aggregate_metrics(per_query: list[dict]) -> dict:
    keys = ["p_at_5", "p_at_10", "r_at_5", "r_at_10", "mrr", "ndcg_at_10", "latency_s"]
    result = {}
    for k in keys:
        vals = [q[k] for q in per_query]
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n else 0.0
        result[f"mean_{k}"] = round(mean, 4)
        result[f"std_{k}"] = round(std, 4)
    return result


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_eval(
    resources,
    queries: list[dict],
    use_domain_filter: bool,
    top_k: int = 10,
) -> list[dict]:
    results = []
    for q in queries:
        domain = q["domain"] if use_domain_filter else None
        relevant = set(q["relevant_indices"])

        t0 = time.perf_counter()
        hits = search(resources, q["query"], top_k=top_k, domain=domain)
        latency = time.perf_counter() - t0

        retrieved = [r["index"] for r in hits]
        metrics = compute_metrics(retrieved, relevant, latency)

        results.append(
            {
                "id": q["id"],
                "query": q["query"],
                "domain": q["domain"],
                "filter_applied": use_domain_filter,
                "num_relevant_gt": len(relevant),
                "num_retrieved": len(retrieved),
                "retrieved_indices": retrieved,
                **metrics,
            }
        )
    return results


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{'=' * 90}")
    print(f" {title}")
    print(f"{'=' * 90}")
    hdr = f"{'ID':<6} {'Domain':<14} {'P@5':>6} {'P@10':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8} {'ms':>7}"
    print(hdr)
    print("-" * 90)
    for r in rows:
        latency_ms = r["latency_s"] * 1000
        print(
            f"{r['id']:<6} {r['domain']:<14} "
            f"{r['p_at_5']:>6.3f} {r['p_at_10']:>6.3f} "
            f"{r['r_at_5']:>6.3f} {r['r_at_10']:>6.3f} "
            f"{r['mrr']:>6.3f} {r['ndcg_at_10']:>8.3f} "
            f"{latency_ms:>7.1f}"
        )


def print_aggregate(agg: dict, label: str) -> None:
    print(f"\n--- {label} ---")
    print(
        f"  mean P@5={agg['mean_p_at_5']:.3f} (sd {agg['std_p_at_5']:.3f})"
        f"  P@10={agg['mean_p_at_10']:.3f} (sd {agg['std_p_at_10']:.3f})"
    )
    print(
        f"  mean R@5={agg['mean_r_at_5']:.3f} (sd {agg['std_r_at_5']:.3f})"
        f"  R@10={agg['mean_r_at_10']:.3f} (sd {agg['std_r_at_10']:.3f})"
    )
    print(
        f"  mean MRR={agg['mean_mrr']:.3f} (sd {agg['std_mrr']:.3f})"
        f"  nDCG@10={agg['mean_ndcg_at_10']:.3f} (sd {agg['std_ndcg_at_10']:.3f})"
    )
    print(f"  mean latency={agg['mean_latency_s'] * 1000:.1f} ms (sd {agg['std_latency_s'] * 1000:.1f} ms)")


def main() -> None:
    # Load ground truth
    gt_path = os.path.join(SCRIPT_DIR, "eval_queries.json")
    with open(gt_path, encoding="utf-8") as f:
        queries = json.load(f)

    print("Loading search engine...")
    resources = load_resources()
    print(f"  {len(resources.titles)} documents loaded.")

    # ── Filtered evaluation ────────────────────────────────────────────────────
    print("\nRunning domain-filtered evaluation (top_k=10)...")
    filtered_results = run_eval(resources, queries, use_domain_filter=True, top_k=10)
    print_table(filtered_results, "Filtered Retrieval (domain filter ON)")
    filtered_agg = aggregate_metrics(filtered_results)
    print_aggregate(filtered_agg, "Filtered — Aggregate")

    # ── Per-domain breakdown ───────────────────────────────────────────────────
    for dom in ("research", "blog", "documentation"):
        dom_rows = [r for r in filtered_results if r["domain"] == dom]
        dom_agg = aggregate_metrics(dom_rows)
        print_aggregate(dom_agg, f"Filtered — {dom} only ({len(dom_rows)} queries)")

    # ── Cross-domain evaluation ────────────────────────────────────────────────
    print("\nRunning cross-domain evaluation (domain filter OFF)...")
    unfiltered_results = run_eval(resources, queries, use_domain_filter=False, top_k=10)
    print_table(unfiltered_results, "Unfiltered Retrieval (no domain filter)")
    unfiltered_agg = aggregate_metrics(unfiltered_results)
    print_aggregate(unfiltered_agg, "Unfiltered — Aggregate")

    # ── Comparison summary ─────────────────────────────────────────────────────
    print("\n\n--- Filtered vs Unfiltered Comparison ---")
    metrics_to_compare = ["mean_p_at_5", "mean_p_at_10", "mean_r_at_10", "mean_mrr", "mean_ndcg_at_10"]
    print(f"{'Metric':<20} {'Filtered':>10} {'Unfiltered':>12} {'Delta':>10}")
    print("-" * 55)
    for m in metrics_to_compare:
        f_val = filtered_agg[m]
        u_val = unfiltered_agg[m]
        delta = f_val - u_val
        sign = "+" if delta >= 0 else ""
        print(f"{m:<20} {f_val:>10.4f} {u_val:>12.4f} {sign}{delta:>9.4f}")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "filtered": {
            "per_query": filtered_results,
            "aggregate": filtered_agg,
            "by_domain": {
                dom: aggregate_metrics([r for r in filtered_results if r["domain"] == dom])
                for dom in ("research", "blog", "documentation")
            },
        },
        "unfiltered": {
            "per_query": unfiltered_results,
            "aggregate": unfiltered_agg,
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
