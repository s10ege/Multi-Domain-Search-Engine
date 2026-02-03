# AI-Driven Multi-Domain-Search-Engine

Features

Hybrid semantic search with BM25 + vectors.
Domain filtering at query time.
RAG answer mode with context retrieval.
Summarize top‑k results.
“More like this” similarity by selected result.
Optional reranked search with cross‑encoder scores.
Optional topic clustering of results.
Results include title, domain, source, score, and index.
CLI Menu

Search — standard semantic search.
RAG / Answer mode — returns an answer plus context results.
Summarize top‑k — summarizes retrieved results.
Find similar to a result — selects a prior result and finds related docs.
Search (reranked) — re‑scores candidates with a cross‑encoder.
Show clusters (optional) — groups results by topic.
Change domain filter — restricts results to one domain.
Show current domain — displays the active domain filter.
Change Top‑K — controls number of results.
Clear domain filter — searches across all domains.
Quit — exits cleanly.
