# SemantikML Evaluation Summary

Date: 2026-04-06  
System: SemantikML — hybrid BM25 + dense retrieval with cross-encoder re-ranking  
Corpus: 9,000 documents (research=4000, blog=3000, documentation=2000)

---

## 1. Test Suite Coverage

| Module | Stmts | Missed | Coverage |
|---|---|---|---|
| src/__init__.py | 0 | 0 | 100% |
| src/build_embeddings.py | 32 | 32 | 0% |
| src/client.py | 102 | 14 | 86% |
| src/search_engine.py | 172 | 20 | 88% |
| **TOTAL** | **306** | **66** | **78%** |

**283 tests passed**, 0 failures.  
Note: `build_embeddings.py` shows 0% because `test_build_embeddings.py` uses
`from backend.src.build_embeddings import build` (broken relative import path).
Excluding that file, coverage of the remaining modules is **87.6%**.

Missing lines in `search_engine.py`: fallback path detection (68-70), tuple
domain handling (157-164), RAG normalise-dict branch (228), summarise fallback
(295-304), and candidate text-enrichment guard (331-333).

---

## 2. Retrieval Evaluation

### 2a. Domain-Filtered Results (domain filter ON, top\_k=10)

| ID | Domain | P@5 | P@10 | R@5 | R@10 | MRR | nDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| r01 | research | 0.800 | 0.500 | 0.800 | 1.000 | 1.000 | 0.982 | 2131 |
| r02 | research | 0.800 | 0.400 | 1.000 | 1.000 | 1.000 | 0.983 | 1488 |
| r03 | research | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1527 |
| r04 | research | 0.800 | 0.800 | 0.500 | 1.000 | 1.000 | 0.947 | 1597 |
| r05 | research | 0.800 | 0.400 | 1.000 | 1.000 | 1.000 | 0.905 | 1582 |
| r06 | research | 1.000 | 0.900 | 0.556 | 1.000 | 1.000 | 0.990 | 1506 |
| r07 | research | 0.800 | 0.700 | 0.571 | 1.000 | 1.000 | 0.980 | 1480 |
| r08 | research | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1509 |
| r09 | research | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1519 |
| r10 | research | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1487 |
| b01 | blog | 0.200 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 75 |
| b02 | blog | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1425 |
| b03 | blog | 0.400 | 0.200 | 1.000 | 1.000 | 1.000 | 0.920 | 135 |
| b04 | blog | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 37 |
| b05 | blog | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 36 |
| b06 | blog | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 508 |
| b07 | blog | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 783 |
| b08 | blog | 0.200 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 66 |
| d01 | documentation | 0.200 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 526 |
| d02 | documentation | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1065 |
| d03 | documentation | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1014 |
| d04 | documentation | 0.200 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 | 147 |
| d05 | documentation | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1194 |
| d06 | documentation | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 504 |
| d07 | documentation | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 692 |

### 2b. Aggregate by Domain (Filtered)

| Domain | P@5 | P@10 | R@5 | R@10 | MRR | nDCG@10 | Mean Latency |
|---|---|---|---|---|---|---|---|
| **research** (n=10) | **0.900** | **0.770** | 0.643 | **1.000** | **1.000** | **0.979** | 1583 ms |
| blog (n=8) | 0.225 | 0.175 | 0.438 | 0.500 | 0.500 | 0.490 | 383 ms |
| documentation (n=7) | 0.771 | 0.743 | 0.643 | **1.000** | **1.000** | **1.000** | 735 ms |
| **Overall** (n=25) | **0.648** | **0.572** | 0.577 | 0.840 | 0.840 | 0.828 | **961 ms** |

### 2c. Filtered vs Unfiltered Comparison

| Metric | Filtered | Unfiltered | Delta |
|---|---|---|---|
| mean P@5 | 0.6480 | 0.6160 | **+0.0320** |
| mean P@10 | 0.5720 | 0.5640 | **+0.0080** |
| mean R@10 | 0.8400 | 0.8150 | **+0.0250** |
| mean MRR | 0.8400 | 0.7667 | **+0.0733** |
| mean nDCG@10 | 0.8282 | 0.7618 | **+0.0664** |
| mean latency | 961 ms | 1448 ms | **-487 ms** |

Domain filtering consistently improves retrieval quality (+7.3% MRR, +6.6% nDCG@10)
while *reducing* latency by 33% — because a smaller candidate pool means fewer
documents for the cross-encoder to re-rank.

---

## 3. RAG Quality Evaluation

10 queries evaluated; model: Qwen3-1.7B. Scoring scale: 0=poor, 1=partial, 2=good.

| ID | Domain | Faithfulness | Relevance | Latency (s) |
|---|---|---|---|---|
| rag_r01 | research | 2/2 | 2/2 | 522 |
| rag_r02 | research | 2/2 | 2/2 | 368 |
| rag_r03 | research | 2/2 | 2/2 | 524 |
| rag_r04 | research | 2/2 | 2/2 | 461 |
| rag_r05 | research | 2/2 | 2/2 | 417 |
| rag_b01 | blog | 1/2 | 2/2 | 192 |
| rag_b02 | blog | **0/2** | 2/2 | 322 |
| rag_b03 | blog | 2/2 | 2/2 | 247 |
| rag_d01 | documentation | 2/2 | 2/2 | 149 |
| rag_d02 | documentation | 2/2 | 2/2 | 159 |

| Domain | Mean Faithfulness | Mean Relevance |
|---|---|---|
| research | **2.00 / 2.0** | **2.00 / 2.0** |
| blog | 1.00 / 2.0 | 2.00 / 2.0 |
| documentation | **2.00 / 2.0** | **2.00 / 2.0** |
| **Overall** | **1.70 / 2.0** | **2.00 / 2.0** |

Mean generation latency: **336 s / query** (CPU-only Qwen3-1.7B, max_new_tokens=512).

---

## 4. Key Observations

### What worked well
- **Research retrieval**: Near-perfect performance (MRR=1.000, nDCG@10=0.979). The
  BAAI/bge-small-en-v1.5 + BM25 hybrid, combined with the cross-encoder, reliably
  finds arXiv papers on RL, GNNs, federated learning, and diffusion models. The
  first relevant result appeared at rank 1 for all 10 research queries.

- **Domain filtering speeds up retrieval**: Contrary to expectation, applying a domain
  filter reduced mean latency from 1,448 ms to 961 ms (-33%) by shrinking the
  cross-encoder candidate pool, while simultaneously improving all retrieval metrics.

- **Documentation retrieval**: Perfect MRR and nDCG@10=1.000 for scikit-learn and
  NumPy queries. sklearn / numpy API docs are well-represented and consistently
  retrieved.

- **RAG relevance**: All 10 RAG responses directly addressed the query (mean
  Relevance=2.0/2.0). Research and documentation answers were factually grounded
  in context (Faithfulness=2.0/2.0).

### Failure modes and limitations

1. **Blog corpus quality**: The blog domain has severe issues:
   - Queries b04 (LSTM) and b05 (pandas) returned 0 results — content not present or not indexed.
   - Queries b06 (transfer learning) and b07 (random forest) returned 10 irrelevant results
     each with similarity score ≈ 0.0000 — the system retrieved superficially related titles.
   - Many blog documents appear duplicated at multiple indices (e.g., "Gradient Descent
     Optimization Explained" appears at ≥10 different index positions with identical content).
   - Cross-encoder similarity scores for blog queries are 100–1000× lower than for research
     (0.0002 vs 0.99), suggesting the blog content is generic/templated rather than substantive.

2. **Corpus duplication**: Multiple documents share identical title and text but occupy separate
   index positions. This inflates candidate counts without adding retrieval value and misleads
   recall calculations.

3. **PyTorch DataLoader query (d04)**: Only 1/3 results relevant. The documentation index lacks
   `torch.utils.data.DataLoader` — a significant gap for a PyTorch documentation collection.

4. **RAG faithfulness on blog**: rag_b02 scored 0/2 for faithfulness. The blog document "Gradient
   Descent Optimization Explained" contains only templated/boilerplate text, giving the LLM
   insufficient grounding material. The model produced a correct answer from its internal
   knowledge rather than from context — a hallucination risk for production use.

5. **RAG latency**: 336 s/query on CPU with Qwen3-1.7B is impractical for interactive use.
   A GPU or smaller model would be required for deployment.

---

## 5. Artefacts

| File | Contents |
|---|---|
| `eval_queries.json` | 25 queries with ground-truth relevant indices |
| `raw_query_results.json` | Raw top-10 results for all 25 queries |
| `eval_results.json` | Per-query + aggregate metrics (filtered & unfiltered) |
| `rag_eval_results.json` | RAG faithfulness/relevance scores for 10 queries |
| `eval_search.py` | Retrieval evaluation script |
| `eval_rag.py` | RAG evaluation script |
| `build_ground_truth.py` | Script used to inspect search results for ground truth labelling |
