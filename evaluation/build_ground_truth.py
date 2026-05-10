"""
build_ground_truth.py
Runs candidate queries against the live search engine and saves raw results
so we can make manual relevance judgements.
Run from the project root:
    python evaluation/build_ground_truth.py
"""

from __future__ import annotations

import json
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Locate backend/src regardless of where the script is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
sys.path.insert(0, os.path.join(BACKEND_DIR, "src"))

os.chdir(BACKEND_DIR)  # make relative paths (final_data.csv, embeddings/) work

from search_engine import load_resources, search  # noqa: E402

CANDIDATE_QUERIES: list[dict] = [
    # ── research domain (~10) ──────────────────────────────────────────────
    {"id": "r01", "query": "transformer attention mechanism self-attention", "domain": "research"},
    {"id": "r02", "query": "convolutional neural network image classification", "domain": "research"},
    {"id": "r03", "query": "reinforcement learning policy gradient reward", "domain": "research"},
    {"id": "r04", "query": "BERT language model pre-training fine-tuning NLP", "domain": "research"},
    {"id": "r05", "query": "generative adversarial network image synthesis", "domain": "research"},
    {"id": "r06", "query": "graph neural network node classification", "domain": "research"},
    {"id": "r07", "query": "knowledge distillation model compression", "domain": "research"},
    {"id": "r08", "query": "federated learning privacy distributed training", "domain": "research"},
    {"id": "r09", "query": "diffusion model score matching denoising", "domain": "research"},
    {"id": "r10", "query": "contrastive learning self-supervised representation", "domain": "research"},
    # ── blog domain (~8) ───────────────────────────────────────────────────
    {"id": "b01", "query": "how to build a neural network from scratch Python", "domain": "blog"},
    {"id": "b02", "query": "gradient descent explained step by step beginner", "domain": "blog"},
    {"id": "b03", "query": "overfitting regularisation dropout tutorial", "domain": "blog"},
    {"id": "b04", "query": "how does LSTM work sequence modelling explained", "domain": "blog"},
    {"id": "b05", "query": "pandas data preprocessing machine learning guide", "domain": "blog"},
    {"id": "b06", "query": "transfer learning practical guide fine-tuning pretrained models", "domain": "blog"},
    {"id": "b07", "query": "random forest vs gradient boosting comparison tutorial", "domain": "blog"},
    {"id": "b08", "query": "visualising neural network weights and activations", "domain": "blog"},
    # ── documentation domain (~7) ──────────────────────────────────────────
    {"id": "d01", "query": "PyTorch tensor operations autograd backward", "domain": "documentation"},
    {"id": "d02", "query": "scikit-learn Pipeline fit transform parameters", "domain": "documentation"},
    {"id": "d03", "query": "NumPy array broadcasting reshape indexing", "domain": "documentation"},
    {"id": "d04", "query": "PyTorch DataLoader Dataset batch size shuffle", "domain": "documentation"},
    {"id": "d05", "query": "scikit-learn cross_val_score GridSearchCV hyperparameter", "domain": "documentation"},
    {"id": "d06", "query": "NumPy linalg linear algebra matrix operations", "domain": "documentation"},
    {"id": "d07", "query": "PyTorch nn.Module forward method layer definition", "domain": "documentation"},
]

TOP_K = 10


def main() -> None:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("Loading search engine...")
    resources = load_resources()
    print(f"  {len(resources.titles)} documents loaded.")

    output: list[dict] = []

    for q in CANDIDATE_QUERIES:
        print(f"\n-- {q['id']} [{q['domain']}] --")
        print(f"   Query: {q['query']}")
        results = search(resources, q["query"], top_k=TOP_K, domain=q["domain"])
        hits = []
        for rank, r in enumerate(results, 1):
            snippet = r["text"][:150].replace("\n", " ")
            title_safe = r["title"][:70]
            print(f"  {rank:2d}. [{r['index']:5d}] {r['domain']:<14} {r['score']:.4f}  {title_safe}")
            print(f"       {snippet}")
            hits.append({
                "rank": rank,
                "index": r["index"],
                "title": r["title"],
                "domain": r["domain"],
                "score": round(r["score"], 6),
                "snippet": snippet,
            })
        output.append({"id": q["id"], "query": q["query"], "domain": q["domain"], "hits": hits})

    out_path = os.path.join(SCRIPT_DIR, "raw_query_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\nSaved raw results → {out_path}")


if __name__ == "__main__":
    main()
