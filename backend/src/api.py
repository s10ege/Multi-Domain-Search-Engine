from __future__ import annotations

import os
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .search_engine import (
    SearchResources,
    close_resources,
    load_resources,
    more_like_this,
    rag_answer,
    search,
)


class SearchRequest(BaseModel):
    query: str = Field("", description="Query text")
    domain: Optional[Union[str, List[str]]] = None
    offset: int = 0
    limit: int = 10


class SimilarRequest(BaseModel):
    seed_id: Union[int, str]
    domain: Optional[Union[str, List[str]]] = None
    offset: int = 0
    limit: int = 10


class SearchResult(BaseModel):
    id: int
    index: int
    title: str
    domain: str
    source: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: Optional[int] = None
    answer: Optional[str] = None


class DocResponse(BaseModel):
    id: int
    title: str
    domain: str
    source: str
    content: str


app = FastAPI(title="SemantikML API", version="0.1.0")

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
origins = [origin.strip() for origin in origins if origin.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
def _startup() -> None:
    csv_path = os.getenv("SEMANTIKML_CSV", "final_data.csv")
    embeddings_path = os.getenv("SEMANTIKML_EMBEDDINGS", "embeddings")
    app.state.resources = load_resources(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    resources: SearchResources = app.state.resources
    close_resources(resources)


def _slice_results(items: List[dict], offset: int, limit: int) -> List[dict]:
    start = max(offset, 0)
    end = start + max(limit, 0)
    return items[start:end]


def _to_result(item: dict) -> SearchResult:
    preview = item.get("text", "")
    snippet = preview[:240] + "..." if len(preview) > 240 else preview
    return SearchResult(
        id=int(item["index"]),
        index=int(item["index"]),
        title=str(item.get("title") or "Untitled"),
        domain=str(item.get("domain") or ""),
        source=str(item.get("source") or ""),
        score=float(item.get("score") or 0.0),
        snippet=snippet,
    )


@app.post("/search", response_model=SearchResponse)
async def search_api(payload: SearchRequest) -> SearchResponse:
    resources: SearchResources = app.state.resources
    top_k = max(payload.offset + payload.limit, 1)
    results = search(
        resources,
        payload.query,
        top_k=top_k,
        domain=payload.domain,
    )
    sliced = _slice_results(results, payload.offset, payload.limit)
    return SearchResponse(results=[_to_result(item) for item in sliced])


@app.post("/deep-research", response_model=SearchResponse)
async def deep_research_api(payload: SearchRequest) -> SearchResponse:
    resources: SearchResources = app.state.resources
    top_k = max(payload.offset + payload.limit, 1)
    output = rag_answer(
        resources,
        payload.query,
        top_k=top_k,
        domain=payload.domain,
    )
    results = output.get("results", [])
    sliced = _slice_results(results, payload.offset, payload.limit)
    return SearchResponse(
        results=[_to_result(item) for item in sliced],
        answer=output.get("answer") or "",
    )


@app.post("/similar", response_model=SearchResponse)
async def similar_api(payload: SimilarRequest) -> SearchResponse:
    resources: SearchResources = app.state.resources
    try:
        seed_index = int(payload.seed_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="seed_id must be an integer") from exc

    top_k = max(payload.offset + payload.limit, 1)
    results = more_like_this(
        resources,
        seed_index,
        top_k=top_k,
        domain=payload.domain,
    )
    sliced = _slice_results(results, payload.offset, payload.limit)
    return SearchResponse(results=[_to_result(item) for item in sliced])


@app.get("/doc/{doc_id}", response_model=DocResponse)
async def doc_api(doc_id: str) -> DocResponse:
    resources: SearchResources = app.state.resources
    try:
        idx = int(doc_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="doc id must be an integer") from exc

    if idx < 0 or idx >= len(resources.texts):
        raise HTTPException(status_code=404, detail="Document not found")

    return DocResponse(
        id=idx,
        title=str(resources.titles[idx]),
        domain=str(resources.domains[idx]),
        source=str(resources.sources[idx]),
        content=str(resources.texts[idx]),
    )
