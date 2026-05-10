# SemantikML — an AI-Driven Multi-Domain Search Engine

A hybrid semantic search engine and RAG system for machine-learning and AI literature. SemantikML indexes **9,000 documents** across three source types — arXiv research papers, Medium / Towards Data Science blog posts, and library documentation — and serves them through a two-stage retrieve-and-rerank pipeline with an optional local Retrieval-Augmented Generation (RAG) answer mode.

Built as a final-year BEng project at Brunel University London (April 2026).

## Why this exists

Practitioner knowledge about ML/AI is fragmented across academic preprints, technical blogs, and API documentation — sources that traditional, keyword-based engines don't combine in one place. Lexical search also breaks down under **vocabulary mismatch**: the same idea ("self-supervised learning", "contrastive pre-training", "pre-training without labels") shows up under different terms across venues and years.

SemantikML addresses both problems with a single domain-specific engine that combines lexical (BM25) and dense (sentence-transformer) retrieval, reranks with a cross-encoder, and optionally grounds answers in retrieved passages via a local RAG pipeline.

## Features

- **Hybrid semantic search** — BM25 + FAISS dense retrieval, scores combined at query time
- **Cross-encoder reranking** — final ranking refined by `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Local RAG answer mode** — grounded natural-language answers via Qwen3-0.6B, no API calls
- **Domain filtering** — restrict to papers, blogs, or library docs
- **More Like This** — exploratory retrieval from an example document
- **Streamlit web app** — local authentication (PBKDF2-HMAC-SHA256), search history, RAG response caching, anonymous user limits
- **Reproducible evaluation suite** — 25-query benchmark with ablation, per-domain metrics, and a generated PDF report
- **258-test automated suite** — unit + integration coverage, all passing

## Architecture

```
            ┌──────────────────────────┐
   query →  │  Hybrid first stage      │
            │  BM25  +  FAISS (IVF230) │
            └────────────┬─────────────┘
                         │  top-k candidates
                         ▼
            ┌──────────────────────────┐
            │  Cross-encoder reranker  │   ms-marco-MiniLM-L-6-v2
            └────────────┬─────────────┘
                         │  reranked results
                         ▼
            ┌──────────────────────────┐
            │  Streamlit UI            │  → search results, More Like This
            │                          │  → optional RAG answer (Qwen3-0.6B)
            └──────────────────────────┘
```

- **Embedding model:** `BAAI/bge-small-en-v1.5` (33M params)
- **Dense index:** FAISS IVF230 (approximate nearest neighbour)
- **Sparse index:** BM25 inverted index
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generator:** `Qwen3-0.6B`, CPU-only
- **Framework:** [txtai](https://github.com/neuml/txtai)

## Dataset

| Source                                    | Count  |
| ----------------------------------------- | -----: |
| arXiv research papers (cs.LG, stat.ML, …) | 4,000  |
| Medium / Towards Data Science blog posts  | 3,000  |
| NumPy / PyTorch / scikit-learn docs       | 2,000  |
| **Total**                                 | **9,000** |

The pre-processed corpus lives at `backend/final_data.csv`. The pre-built txtai index is generated into `backend/embeddings/` (gitignored — regenerate locally).

## Quick start

### 1. Set up the virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Build the embeddings index

```bash
cd backend
python src/build_embeddings.py
```

### 4. Run the app

```bash
# Streamlit web app (from project root)
python -m streamlit run backend/streamlit_app/main.py

# Or the CLI client
cd backend && python -m src.client
```

## Evaluation

A 25-query benchmark with graded relevance judgements measures each architectural component's contribution. **Ablation results (nDCG@10):**

| Configuration                  | nDCG@10 | Δ vs. previous |
| ------------------------------ | ------: | -------------: |
| BM25 only                      |   0.355 |              — |
| + Dense retrieval              |   0.531 |        +49.6 % |
| + Hybrid (BM25 + dense)        |   0.565 |         +6.4 % |
| + Cross-encoder rerank         | **0.828** |        +46.5 % |

The cross-encoder reranker is the dominant quality lever, at roughly **37×** the latency of the first stage.

**Per-domain retrieval performance:**

| Domain         | nDCG@10 | MRR   |
| -------------- | ------: | ----: |
| Research papers | 0.979  | 1.000 |
| Library docs    | 1.000  | 1.000 |
| Blog posts      | 0.490  | —     |

The lower blog score reflects corpus-coverage gaps in that domain rather than a retrieval-algorithm failure.

**RAG pipeline:** answers are evaluated for faithfulness and relevance across domains. The main practical constraint is a mean uncached generation latency of ~336 s on CPU, mitigated in practice by a query-level response cache.

To reproduce locally:

```bash
python evaluation/eval_search.py        # ablation + per-domain
python evaluation/eval_rag.py           # RAG faithfulness / relevance
python evaluation/ablation_study.py     # component ablation
python evaluation/generate_evaluation_pdf.py   # full PDF report
```

Raw JSON results and the generated `EVALUATION_REPORT.pdf` are committed under `evaluation/`.

## Project structure

```
.
├── backend/
│   ├── src/
│   │   ├── search_engine.py        # Core hybrid + rerank pipeline
│   │   ├── build_embeddings.py     # FAISS / BM25 index builder
│   │   └── client.py               # CLI client
│   ├── streamlit_app/
│   │   ├── main.py                 # Entry point
│   │   ├── search.py / similar.py  # Search + More Like This UI
│   │   ├── auth_store.py           # PBKDF2 local auth
│   │   ├── history_store.py        # Per-user search history
│   │   ├── login.py / signup.py / admin.py
│   │   └── app_core.py             # Session state / shared services
│   ├── embeddings/                 # Pre-built index (gitignored)
│   ├── final_data.csv              # Source corpus
│   └── test/                       # 258 pytest tests
├── evaluation/                     # Eval scripts + results + report PDF
├── figures/                        # Result and UI figures (PNG)
├── pdf/                            # Manual test cases, improvement plan
├── scripts/                        # Figure generation, screenshots, health check
└── .venv/                          # Virtual environment (gitignored)
```

## Testing

```bash
pytest backend/test/                                   # full suite
pytest backend/test/ --cov=backend --cov-report=html   # with coverage
pytest backend/test/test_search_engine.py -v           # one module
```

258 tests cover the search engine, embeddings builder, RAG pipeline, Streamlit views, auth store, and history store.

## Tech stack

- **Python 3.8+**, virtualenv at `.venv/`
- **[txtai](https://github.com/neuml/txtai)** — semantic-search / RAG orchestration
- **sentence-transformers** — `BAAI/bge-small-en-v1.5` embeddings
- **FAISS** (CPU) — approximate nearest-neighbour search
  *(on Windows, `faiss-cpu` is excluded — txtai falls back to its built-in index)*
- **transformers** — cross-encoder reranker and Qwen3 generator
- **Streamlit** — web UI
- **SQLite** — local auth (`auth.db`) and search history (`search_history.db`)
- **pytest** — test framework

## Limitations and threats to validity

- **RAG latency** — uncached answers average ~336 s on CPU; the response cache hides this for repeated queries, but cold queries are slow.
- **Blog-domain coverage gaps** — the 3,000-document blog slice under-covers some topics, which depresses retrieval scores in that domain.
- **Evaluation scale** — 25 queries with manual ground truth; larger benchmarks would tighten the confidence intervals.
- **No GPU acceleration assumed** — everything runs CPU-first by design.

## Acknowledgements

Final-year project at Brunel University London, College of Engineering, Design and Physical Sciences (Department of Engineering), FHEQ Level 6 BEng, April 2026.

Open-source components: txtai (NeuML), Sentence-Transformers and `bge-small-en-v1.5` (BAAI), `ms-marco-MiniLM-L-6-v2` cross-encoder, Qwen3-0.6B, FAISS, Streamlit.
