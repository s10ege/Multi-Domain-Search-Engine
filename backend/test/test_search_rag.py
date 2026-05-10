# test_search_rag.py
# Comprehensive dissertation test suite for the search and RAG pipeline.
#
# Coverage:
#   - search()       : cross-encoder re-ranking, score ordering, text/query truncation
#   - domain filter  : end-to-end propagation, SQL clause correctness
#   - top-K          : boundary conditions, candidate budget formula
#   - more_like_this : seed exclusion, domain scoping, candidate budget, enrichment
#   - rag_answer()   : empty-context guard, domain scoping
#   - history_store  : RAG cache round-trip, save_search UPSERT, list_recent_searches
#
# All ML models are mocked — no GPU or internet connection required.

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
TEST_DIR = Path(__file__).parent
BACKEND_DIR = TEST_DIR.parent
STREAMLIT_APP_DIR = BACKEND_DIR / "streamlit_app"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(STREAMLIT_APP_DIR))

from src.search_engine import (
    SearchResources,
    _domain_clause,
    _search_candidates,
    more_like_this,
    rag_answer,
    search,
)
from history_store import (
    clear_history,
    get_rag_cache,
    init_db,
    list_recent_searches,
    save_rag_cache,
    save_search,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_resources(n: int = 5) -> SearchResources:
    """Return a SearchResources with n fake documents."""
    return SearchResources(
        titles=[f"Title {i}" for i in range(n)],
        texts=[f"Text content for document {i}" for i in range(n)],
        domains=["research" if i % 2 == 0 else "blog" for i in range(n)],
        sources=[f"source{i}.com" for i in range(n)],
        embeddings=MagicMock(),
        rag=MagicMock(),
    )


def _make_candidate(idx: int, score: float = 0.9, domain: str = "research") -> dict:
    return {
        "title": f"Title {idx}",
        "text": f"Text content for document {idx}",
        "domain": domain,
        "source": f"source{idx}.com",
        "score": score,
        "index": idx,
    }


@pytest.fixture
def history_db(tmp_path):
    """Fresh SQLite history DB for each test."""
    db = str(tmp_path / "history.db")
    init_db(db)
    return db


# ══════════════════════════════════════════════════════════════════════════════
# TestSearchScoreOrdering
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchScoreOrdering:
    """
    Verifies that the cross-encoder re-ranking pipeline genuinely reorders
    candidates relative to the original bi-encoder scores.
    Priority: CRITICAL — key dissertation evidence that re-ranking adds value.
    """

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_cross_encoder_reorders_candidates(
        self, mock_candidates, mock_get_sim
    ):
        """
        When the cross-encoder disagrees with bi-encoder ordering, the final
        result list reflects the cross-encoder order, not the original order.
        """
        # Bi-encoder returns [0, 1, 2] with descending raw scores
        mock_candidates.return_value = [
            _make_candidate(0, score=0.90),
            _make_candidate(1, score=0.80),
            _make_candidate(2, score=0.70),
        ]
        # Cross-encoder reverses the order: candidate index 2 wins
        mock_sim = Mock(return_value=[(2, 0.95), (0, 0.60), (1, 0.50)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(3), "query", top_k=3)

        assert results[0]["title"] == "Title 2"
        assert results[1]["title"] == "Title 0"
        assert results[2]["title"] == "Title 1"

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_results_sorted_by_cross_encoder_score_descending(
        self, mock_candidates, mock_get_sim
    ):
        """
        Final scores in the returned list are in strictly descending order.
        """
        mock_candidates.return_value = [_make_candidate(i) for i in range(4)]
        mock_sim = Mock(return_value=[(1, 0.88), (3, 0.75), (0, 0.60), (2, 0.45)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(4), "query", top_k=4)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_top_result_has_highest_cross_encoder_score(
        self, mock_candidates, mock_get_sim
    ):
        """
        The first returned result always carries the highest cross-encoder score.
        """
        mock_candidates.return_value = [_make_candidate(i) for i in range(5)]
        mock_sim = Mock(return_value=[(4, 0.99), (0, 0.80), (2, 0.75), (1, 0.60), (3, 0.55)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(5), "query", top_k=5)

        assert results[0]["score"] == max(r["score"] for r in results)

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_cross_encoder_score_replaces_biencoder_score(
        self, mock_candidates, mock_get_sim
    ):
        """
        The score in each returned result comes from the cross-encoder,
        not from the original bi-encoder score stored in candidates.
        """
        original_biencoder_score = 0.55
        mock_candidates.return_value = [_make_candidate(0, score=original_biencoder_score)]
        cross_encoder_score = 0.93
        mock_sim = Mock(return_value=[(0, cross_encoder_score)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(1), "query", top_k=1)

        assert results[0]["score"] == cross_encoder_score
        assert results[0]["score"] != original_biencoder_score

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_single_candidate_returned_as_only_result(
        self, mock_candidates, mock_get_sim
    ):
        """
        With a single candidate, re-ranking still works and returns that document.
        """
        mock_candidates.return_value = [_make_candidate(0, score=0.75)]
        mock_sim = Mock(return_value=[(0, 0.82)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(1), "query", top_k=1)

        assert len(results) == 1
        assert results[0]["title"] == "Title 0"
        assert results[0]["score"] == 0.82

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_text_truncated_to_500_chars_before_cross_encoder(
        self, mock_candidates, mock_get_sim
    ):
        """
        Candidate texts longer than 500 characters are truncated before being
        passed to the cross-encoder to avoid exceeding token limits.
        """
        long_text = "A" * 1000
        resources = _make_resources(1)
        object.__setattr__(resources, "texts", [long_text])

        candidate = {
            "title": "Long Doc",
            "text": long_text,
            "domain": "research",
            "source": "src.com",
            "score": 0.8,
            "index": 0,
        }
        mock_candidates.return_value = [candidate]
        mock_sim = Mock(return_value=[(0, 0.90)])
        mock_get_sim.return_value = mock_sim

        search(resources, "query", top_k=1)

        call_args = mock_sim.call_args
        passed_texts = call_args[0][1]  # second positional arg is list of truncated texts
        assert len(passed_texts[0]) == 500

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_query_truncated_to_500_chars_before_cross_encoder(
        self, mock_candidates, mock_get_sim
    ):
        """
        A query longer than 500 characters is truncated before being passed
        to the cross-encoder similarity pipeline.
        """
        long_query = "Q" * 1000
        mock_candidates.return_value = [_make_candidate(0)]
        mock_sim = Mock(return_value=[(0, 0.88)])
        mock_get_sim.return_value = mock_sim

        search(_make_resources(1), long_query, top_k=1)

        call_args = mock_sim.call_args
        passed_query = call_args[0][0]  # first positional arg is the query
        assert len(passed_query) == 500

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_reranking_uses_same_similarity_model_constant(
        self, mock_candidates, mock_get_sim
    ):
        """
        search() always loads the cross-encoder using SIMILARITY_MODEL_PATH.
        """
        from src.search_engine import SIMILARITY_MODEL_PATH

        mock_candidates.return_value = [_make_candidate(0)]
        mock_sim = Mock(return_value=[(0, 0.9)])
        mock_get_sim.return_value = mock_sim

        search(_make_resources(1), "query", top_k=1)

        mock_get_sim.assert_called_once_with(SIMILARITY_MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# TestDomainFilteringIntegrity
# ══════════════════════════════════════════════════════════════════════════════

class TestDomainFilteringIntegrity:
    """
    Verifies that the domain parameter flows correctly from search() through
    to the SQL query executed by _search_candidates(), preventing cross-domain
    result leakage.
    Priority: CRITICAL
    """

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_search_passes_domain_kwarg_to_candidates(
        self, mock_candidates, mock_get_sim
    ):
        """
        search() forwards the domain parameter to _search_candidates().
        """
        mock_candidates.return_value = []

        search(_make_resources(), "query", top_k=5, domain="research")

        call_kwargs = mock_candidates.call_args[1]
        assert call_kwargs["domain"] == "research"

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_search_passes_none_domain_when_unspecified(
        self, mock_candidates, mock_get_sim
    ):
        """
        search() passes domain=None to _search_candidates() when no domain
        is specified, resulting in no SQL filter.
        """
        mock_candidates.return_value = []

        search(_make_resources(), "query", top_k=5)

        call_kwargs = mock_candidates.call_args[1]
        assert call_kwargs["domain"] is None

    def test_sql_contains_domain_clause_for_research(self):
        """
        _search_candidates embeds a domain WHERE clause for domain='research'.
        """
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = []
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=mock_embeddings, rag=MagicMock(),
        )

        _search_candidates(resources, "test", candidate_k=5, domain="research")

        sql = mock_embeddings.search.call_args[0][0]
        assert "domain = 'research'" in sql

    def test_sql_contains_domain_clause_for_blog(self):
        """
        _search_candidates embeds the correct WHERE clause for domain='blog'.
        """
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = []
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=mock_embeddings, rag=MagicMock(),
        )

        _search_candidates(resources, "test", candidate_k=5, domain="blog")

        sql = mock_embeddings.search.call_args[0][0]
        assert "domain = 'blog'" in sql

    def test_sql_contains_domain_clause_for_documentation(self):
        """
        _search_candidates embeds the correct WHERE clause for domain='documentation'.
        """
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = []
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=mock_embeddings, rag=MagicMock(),
        )

        _search_candidates(resources, "test", candidate_k=5, domain="documentation")

        sql = mock_embeddings.search.call_args[0][0]
        assert "domain = 'documentation'" in sql

    def test_sql_has_no_domain_clause_when_domain_is_none(self):
        """
        With domain=None, the SQL query contains no AND domain clause.
        """
        mock_embeddings = MagicMock()
        mock_embeddings.search.return_value = []
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=mock_embeddings, rag=MagicMock(),
        )

        _search_candidates(resources, "test", candidate_k=5, domain=None)

        sql = mock_embeddings.search.call_args[0][0]
        assert "domain" not in sql.lower()

    def test_domain_clause_normalises_uppercase_input(self):
        """
        _domain_clause lowercases the domain string before embedding in SQL.
        """
        clause = _domain_clause("RESEARCH")
        assert clause == "domain = 'research'"

    def test_domain_clause_strips_leading_trailing_whitespace(self):
        """
        _domain_clause strips whitespace around the domain name.
        """
        clause = _domain_clause("  blog  ")
        assert clause == "domain = 'blog'"

    def test_domain_clause_list_takes_first_element_only(self):
        """
        When a list is passed, only the first domain is used.
        """
        clause = _domain_clause(["research", "blog", "documentation"])
        assert clause == "domain = 'research'"
        assert "blog" not in clause
        assert "documentation" not in clause

    @patch("src.search_engine.search")
    def test_rag_answer_propagates_domain_to_search(self, mock_search):
        """
        rag_answer() passes its domain argument through to search().
        """
        mock_search.return_value = []

        rag_answer(_make_resources(), "question", domain="documentation")

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["domain"] == "documentation"

    @patch("src.search_engine.search")
    def test_more_like_this_propagates_domain_to_search(self, mock_search):
        """
        more_like_this() passes its domain argument through to search().
        """
        mock_search.return_value = []
        resources = _make_resources(3)

        more_like_this(resources, index=0, top_k=3, domain="blog")

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["domain"] == "blog"


# ══════════════════════════════════════════════════════════════════════════════
# TestTopKBoundaries
# ══════════════════════════════════════════════════════════════════════════════

class TestTopKBoundaries:
    """
    Tests boundary conditions for the top_k parameter and the candidate
    budget formula: RERANK_BUDGET = max(top_k * 4, 20).
    Priority: CRITICAL
    """

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_top_k_1_returns_exactly_one_result(
        self, mock_candidates, mock_get_sim
    ):
        """top_k=1 returns exactly one document."""
        mock_candidates.return_value = [_make_candidate(i) for i in range(20)]
        mock_sim = Mock(return_value=[(i, 1.0 - i * 0.04) for i in range(20)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(20), "query", top_k=1)

        assert len(results) == 1

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_top_k_20_returns_at_most_20_results(
        self, mock_candidates, mock_get_sim
    ):
        """top_k=20 returns at most 20 documents (the UI maximum)."""
        mock_candidates.return_value = [_make_candidate(i) for i in range(80)]
        mock_sim = Mock(return_value=[(i, 1.0 - i * 0.01) for i in range(80)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(80), "query", top_k=20)

        assert len(results) <= 20

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine._search_candidates")
    def test_top_k_larger_than_available_returns_all_available(
        self, mock_candidates, mock_get_sim
    ):
        """
        When fewer candidates exist than top_k, all available results
        are returned without raising an error.
        """
        n_available = 3
        mock_candidates.return_value = [_make_candidate(i) for i in range(n_available)]
        mock_sim = Mock(return_value=[(i, 0.9 - i * 0.1) for i in range(n_available)])
        mock_get_sim.return_value = mock_sim

        results = search(_make_resources(n_available), "query", top_k=10)

        assert len(results) == n_available

    @patch("src.search_engine._search_candidates")
    def test_candidate_budget_is_4x_top_k_above_threshold(
        self, mock_candidates
    ):
        """
        For top_k=10, candidate_k should be 40 (4 × 10).
        """
        mock_candidates.return_value = []

        search(_make_resources(), "query", top_k=10)

        candidate_k_used = mock_candidates.call_args[0][2]
        assert candidate_k_used == 40

    @patch("src.search_engine._search_candidates")
    def test_candidate_budget_minimum_is_20_for_small_top_k(
        self, mock_candidates
    ):
        """
        For top_k=1, candidate_k must be at least 20 (floor prevents too few candidates).
        """
        mock_candidates.return_value = []

        search(_make_resources(), "query", top_k=1)

        candidate_k_used = mock_candidates.call_args[0][2]
        assert candidate_k_used >= 20

    @patch("src.search_engine._search_candidates")
    def test_candidate_budget_minimum_20_for_top_k_4(
        self, mock_candidates
    ):
        """
        For top_k=4, 4×4=16 < 20, so candidate_k should still be 20.
        """
        mock_candidates.return_value = []

        search(_make_resources(), "query", top_k=4)

        candidate_k_used = mock_candidates.call_args[0][2]
        assert candidate_k_used == 20

    @patch("src.search_engine._search_candidates")
    def test_candidate_budget_24_for_top_k_6(self, mock_candidates):
        """
        For top_k=6, 4×6=24 > 20, so candidate_k should be 24.
        """
        mock_candidates.return_value = []

        search(_make_resources(), "query", top_k=6)

        candidate_k_used = mock_candidates.call_args[0][2]
        assert candidate_k_used == 24

    @patch("src.search_engine._search_candidates")
    def test_empty_candidates_returns_empty_list(self, mock_candidates):
        """
        When _search_candidates returns an empty list, search returns [].
        """
        mock_candidates.return_value = []

        results = search(_make_resources(), "query", top_k=5)

        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# TestMoreLikeThisExtended
# ══════════════════════════════════════════════════════════════════════════════

class TestMoreLikeThisExtended:
    """
    Extended tests for more_like_this() beyond the basic suite.
    Focuses on candidate budget formula, enrichment, and domain scoping.
    Priority: IMPORTANT
    """

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_candidate_budget_is_10x_top_k(
        self, mock_search, mock_get_sim
    ):
        """
        more_like_this requests a large candidate pool: max(top_k * 10, top_k + 10).
        For top_k=5, this is max(50, 15) = 50.
        """
        mock_search.return_value = []
        resources = _make_resources(3)

        more_like_this(resources, index=0, top_k=5)

        # top_k is passed as a keyword argument to search()
        call_kwargs = mock_search.call_args[1]
        candidate_k_requested = call_kwargs["top_k"]
        assert candidate_k_requested == 50

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_uses_seed_index_zero(
        self, mock_search, mock_get_sim
    ):
        """
        more_like_this uses the text of the seed document (index=0) as the query.
        """
        mock_search.return_value = []
        resources = _make_resources(3)
        seed_text = resources.texts[0]

        more_like_this(resources, index=0, top_k=3)

        query_used = mock_search.call_args[0][1]
        assert query_used == seed_text

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_uses_seed_index_last(
        self, mock_search, mock_get_sim
    ):
        """
        more_like_this correctly uses the text of the last document as seed.
        """
        mock_search.return_value = []
        resources = _make_resources(5)
        last_index = 4
        seed_text = resources.texts[last_index]

        more_like_this(resources, index=last_index, top_k=3)

        query_used = mock_search.call_args[0][1]
        assert query_used == seed_text

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_excludes_seed_from_results(
        self, mock_search, mock_get_sim
    ):
        """
        The seed document (index=2) must not appear in the returned results.
        more_like_this filters the seed BEFORE re-ranking, so the similarity
        mock must return exactly (n_candidates - 1) tuples.
        """
        seed_index = 2
        all_candidates = [_make_candidate(i) for i in range(5)]
        mock_search.return_value = all_candidates
        # After filtering seed (index=2), 4 candidates remain — mock must return 4 tuples
        mock_sim = Mock(return_value=[(i, 0.9 - i * 0.1) for i in range(4)])
        mock_get_sim.return_value = mock_sim

        resources = _make_resources(5)
        results = more_like_this(resources, index=seed_index, top_k=4)

        for r in results:
            assert r["index"] != seed_index

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_returns_at_most_top_k(
        self, mock_search, mock_get_sim
    ):
        """
        more_like_this never returns more than top_k documents.
        Seed is NOT included in mock_search so no filtering offset occurs.
        """
        # Indices 1..20 — seed (index=0) deliberately excluded from search results
        candidates = [_make_candidate(i) for i in range(1, 21)]  # 20 non-seed candidates
        mock_search.return_value = candidates
        mock_sim = Mock(return_value=[(i, 1.0 - i * 0.05) for i in range(20)])
        mock_get_sim.return_value = mock_sim

        resources = _make_resources(21)
        results = more_like_this(resources, index=0, top_k=4)

        assert len(results) <= 4

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_returns_empty_when_no_candidates(
        self, mock_search, mock_get_sim
    ):
        """
        When search finds no candidates, more_like_this returns [].
        """
        mock_search.return_value = []
        resources = _make_resources(3)

        results = more_like_this(resources, index=0, top_k=5)

        assert results == []

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_domain_none_returns_cross_domain_results(
        self, mock_search, mock_get_sim
    ):
        """
        With domain=None, more_like_this uses no domain filter.
        """
        mock_search.return_value = []
        resources = _make_resources(3)

        more_like_this(resources, index=0, top_k=3, domain=None)

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["domain"] is None

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_reranks_remaining_candidates_with_cross_encoder(
        self, mock_search, mock_get_sim
    ):
        """
        After filtering the seed, the remaining candidates are re-ranked by
        the cross-encoder (similarity pipeline is called).
        """
        seed_index = 0
        other_candidates = [_make_candidate(i) for i in range(1, 4)]
        mock_search.return_value = [_make_candidate(seed_index)] + other_candidates
        mock_sim = Mock(return_value=[(i, 0.9 - i * 0.1) for i in range(len(other_candidates))])
        mock_get_sim.return_value = mock_sim

        resources = _make_resources(4)
        more_like_this(resources, index=seed_index, top_k=3)

        mock_sim.assert_called_once()

    @patch("src.search_engine._get_similarity_pipeline")
    @patch("src.search_engine.search")
    def test_more_like_this_seed_only_result_returns_empty(
        self, mock_search, mock_get_sim
    ):
        """
        If search returns only the seed document and nothing else,
        more_like_this returns [] after filtering.
        """
        seed_index = 0
        mock_search.return_value = [_make_candidate(seed_index)]
        mock_get_sim.return_value = Mock(return_value=[])

        resources = _make_resources(1)
        results = more_like_this(resources, index=seed_index, top_k=5)

        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# TestRAGEmptyContextGuard
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGEmptyContextGuard:
    """
    Verifies that rag_answer() returns an empty response without calling the
    LLM whenever the search step returns no results.
    Priority: CRITICAL — prevents unnecessary/expensive LLM calls.
    """

    @patch("src.search_engine.search")
    def test_empty_search_results_returns_empty_answer(self, mock_search):
        """
        Zero search results → answer is an empty string.
        """
        mock_search.return_value = []
        resources = _make_resources()

        result = rag_answer(resources, "what is attention?")

        assert result["answer"] == ""

    @patch("src.search_engine.search")
    def test_empty_search_results_returns_empty_results_list(self, mock_search):
        """
        Zero search results → results key is an empty list.
        """
        mock_search.return_value = []
        resources = _make_resources()

        result = rag_answer(resources, "what is BERT?")

        assert result["results"] == []

    @patch("src.search_engine.search")
    def test_llm_not_called_when_search_returns_empty(self, mock_search):
        """
        The RAG pipeline (LLM) must not be invoked when search finds nothing.
        """
        mock_search.return_value = []
        mock_rag = MagicMock()
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=MagicMock(), rag=mock_rag,
        )

        rag_answer(resources, "some question")

        mock_rag.assert_not_called()

    @patch("src.search_engine.search")
    def test_empty_guard_applies_regardless_of_domain(self, mock_search):
        """
        The empty-context guard fires even when a domain filter is set.
        """
        mock_search.return_value = []
        mock_rag = MagicMock()
        resources = SearchResources(
            titles=[], texts=[], domains=[], sources=[],
            embeddings=MagicMock(), rag=mock_rag,
        )

        result = rag_answer(resources, "question", domain="research")

        mock_rag.assert_not_called()
        assert result == {"answer": "", "results": []}

    @patch("src.search_engine.search")
    def test_empty_guard_applies_for_top_k_zero(self, mock_search):
        """
        With top_k=0 (edge case), no candidates exist and guard fires.
        """
        mock_search.return_value = []
        resources = _make_resources()

        result = rag_answer(resources, "question", top_k=0)

        assert result["answer"] == ""
        assert result["results"] == []

    @patch("src.search_engine.search")
    def test_rag_returns_dict_with_correct_keys(self, mock_search):
        """
        The return value always has 'answer' and 'results' keys, even when empty.
        """
        mock_search.return_value = []
        resources = _make_resources()

        result = rag_answer(resources, "question")

        assert "answer" in result
        assert "results" in result

    @patch("src.search_engine.search")
    def test_rag_calls_llm_when_results_are_present(self, mock_search):
        """
        Positive control: when results exist, the LLM IS called.
        """
        mock_search.return_value = [_make_candidate(0)]
        mock_rag = MagicMock(return_value="Generated answer text.")
        resources = SearchResources(
            titles=["T"], texts=["Doc text"], domains=["research"], sources=["src"],
            embeddings=MagicMock(), rag=mock_rag,
        )

        rag_answer(resources, "question with results")

        mock_rag.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# TestRAGDomainScoping
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGDomainScoping:
    """
    Verifies that the domain filter propagates correctly through the full
    RAG pipeline: rag_answer() → search() → _search_candidates() → SQL.
    Priority: CRITICAL
    """

    @patch("src.search_engine.search")
    def test_rag_passes_research_domain_to_search(self, mock_search):
        """rag_answer passes domain='research' to search()."""
        mock_search.return_value = []
        rag_answer(_make_resources(), "question", domain="research")
        assert mock_search.call_args[1]["domain"] == "research"

    @patch("src.search_engine.search")
    def test_rag_passes_blog_domain_to_search(self, mock_search):
        """rag_answer passes domain='blog' to search()."""
        mock_search.return_value = []
        rag_answer(_make_resources(), "question", domain="blog")
        assert mock_search.call_args[1]["domain"] == "blog"

    @patch("src.search_engine.search")
    def test_rag_passes_documentation_domain_to_search(self, mock_search):
        """rag_answer passes domain='documentation' to search()."""
        mock_search.return_value = []
        rag_answer(_make_resources(), "question", domain="documentation")
        assert mock_search.call_args[1]["domain"] == "documentation"

    @patch("src.search_engine.search")
    def test_rag_passes_none_domain_when_unspecified(self, mock_search):
        """rag_answer passes domain=None when no domain is given."""
        mock_search.return_value = []
        rag_answer(_make_resources(), "question")
        assert mock_search.call_args[1]["domain"] is None

    @patch("src.search_engine.search")
    def test_rag_top_k_propagated_to_search(self, mock_search):
        """rag_answer forwards the top_k value to search()."""
        mock_search.return_value = []
        rag_answer(_make_resources(), "question", top_k=7)
        assert mock_search.call_args[1]["top_k"] == 7

    @patch("src.search_engine.search")
    def test_rag_results_are_returned_in_output(self, mock_search):
        """rag_answer includes the search results in its return value."""
        result_doc = _make_candidate(0)
        mock_search.return_value = [result_doc]
        mock_rag = MagicMock(return_value="An answer.")
        resources = SearchResources(
            titles=["T"], texts=["Doc"], domains=["research"], sources=["src"],
            embeddings=MagicMock(), rag=mock_rag,
        )

        result = rag_answer(resources, "question")

        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == result_doc["title"]

    @patch("src.search_engine.search")
    def test_rag_different_domains_produce_different_sql_filters(
        self, mock_search
    ):
        """
        Calling rag_answer with different domain values results in different
        domain kwargs being passed to search — domains are not cached or reused.
        """
        mock_search.return_value = []

        rag_answer(_make_resources(), "q", domain="research")
        rag_answer(_make_resources(), "q", domain="blog")

        calls = mock_search.call_args_list
        assert calls[0][1]["domain"] == "research"
        assert calls[1][1]["domain"] == "blog"

    @patch("src.search_engine.search")
    def test_rag_context_texts_include_title_prefix(self, mock_search):
        """
        Each context text passed to the LLM is prefixed with 'Title: <title>'.
        """
        mock_search.return_value = [{"title": "Paper X", "text": "Body text"}]
        mock_rag = MagicMock(return_value="answer")
        resources = SearchResources(
            titles=["Paper X"], texts=["Body text"], domains=["research"], sources=["src"],
            embeddings=MagicMock(), rag=mock_rag,
        )

        rag_answer(resources, "question")

        rag_call_kwargs = mock_rag.call_args[1]
        assert any("Title: Paper X" in t for t in rag_call_kwargs["texts"])


# ══════════════════════════════════════════════════════════════════════════════
# TestHistoryStoreRAGCache
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryStoreRAGCache:
    """
    Tests the SQLite RAG cache round-trip (save → retrieve) for correctness
    of every stored field.
    Priority: CRITICAL — caching is a key dissertation feature.
    """

    def test_cache_miss_returns_none(self, history_db):
        """
        get_rag_cache returns None when no entry exists for the given key.
        """
        result = get_rag_cache(history_db, "user1", "never asked", "research", 5)
        assert result is None

    def test_cache_round_trip_preserves_answer(self, history_db):
        """
        Saved answer is returned unchanged by get_rag_cache.
        """
        save_rag_cache(history_db, "u1", "query", "research", 5, "The answer.", [], 2.5)
        cached = get_rag_cache(history_db, "u1", "query", "research", 5)
        assert cached["answer"] == "The answer."

    def test_cache_round_trip_preserves_results_json(self, history_db):
        """
        Saved results list (stored as JSON) is returned correctly deserialized.
        """
        results = [{"title": "Paper", "score": 0.9}]
        save_rag_cache(history_db, "u1", "query", "research", 5, "answer", results, None)
        cached = get_rag_cache(history_db, "u1", "query", "research", 5)
        assert cached["results"] == results

    def test_cache_round_trip_preserves_elapsed_seconds(self, history_db):
        """
        Elapsed seconds (timing info for the dissertation) is stored and returned.
        """
        save_rag_cache(history_db, "u1", "query", "research", 5, "answer", [], 12.34)
        cached = get_rag_cache(history_db, "u1", "query", "research", 5)
        assert abs(cached["elapsed_seconds"] - 12.34) < 0.001

    def test_cache_null_elapsed_seconds_is_stored(self, history_db):
        """
        elapsed_seconds=None is stored and returned as None.
        """
        save_rag_cache(history_db, "u1", "query", "research", 5, "answer", [], None)
        cached = get_rag_cache(history_db, "u1", "query", "research", 5)
        assert cached["elapsed_seconds"] is None

    def test_cache_is_user_scoped(self, history_db):
        """
        User A's cached answer is not returned for User B's identical query.
        """
        save_rag_cache(history_db, "user_A", "shared query", "research", 5, "A's answer", [], 1.0)
        cached_for_b = get_rag_cache(history_db, "user_B", "shared query", "research", 5)
        assert cached_for_b is None

    def test_cache_different_domain_is_cache_miss(self, history_db):
        """
        The cache key includes domain — different domain means cache miss.
        """
        save_rag_cache(history_db, "u1", "query", "research", 5, "answer", [], 1.0)
        cached = get_rag_cache(history_db, "u1", "query", "blog", 5)
        assert cached is None

    def test_cache_different_top_k_is_cache_miss(self, history_db):
        """
        The cache key includes top_k — different top_k means cache miss.
        """
        save_rag_cache(history_db, "u1", "query", "research", 5, "answer", [], 1.0)
        cached = get_rag_cache(history_db, "u1", "query", "research", 10)
        assert cached is None

    def test_cache_query_lookup_is_case_insensitive(self, history_db):
        """
        'Neural Networks' and 'neural networks' map to the same cache entry.
        """
        save_rag_cache(history_db, "u1", "Neural Networks", "research", 5, "answer", [], 1.0)
        cached = get_rag_cache(history_db, "u1", "neural networks", "research", 5)
        assert cached is not None
        assert cached["answer"] == "answer"

    def test_cache_query_is_normalised_before_save(self, history_db):
        """
        Extra whitespace in the query is collapsed before saving,
        so a clean lookup finds the entry.
        """
        save_rag_cache(history_db, "u1", "neural   networks", "research", 5, "answer", [], 1.0)
        cached = get_rag_cache(history_db, "u1", "neural networks", "research", 5)
        assert cached is not None

    def test_empty_answer_is_not_cached(self, history_db):
        """
        An empty answer string is not saved (guard against caching failures).
        """
        save_rag_cache(history_db, "u1", "question", "research", 5, "", [], 1.0)
        cached = get_rag_cache(history_db, "u1", "question", "research", 5)
        assert cached is None

    def test_empty_query_is_not_cached(self, history_db):
        """
        An empty query string is not saved.
        """
        save_rag_cache(history_db, "u1", "", "research", 5, "some answer", [], 1.0)
        cached = get_rag_cache(history_db, "u1", "", "research", 5)
        assert cached is None

    def test_upsert_overwrites_previous_cache_entry(self, history_db):
        """
        Saving the same cache key twice updates the stored answer (UPSERT).
        """
        save_rag_cache(history_db, "u1", "question", "research", 5, "first", [], 1.0)
        save_rag_cache(history_db, "u1", "question", "research", 5, "updated", [], 2.0)
        cached = get_rag_cache(history_db, "u1", "question", "research", 5)
        assert cached["answer"] == "updated"


# ══════════════════════════════════════════════════════════════════════════════
# TestHistoryStoreSaveSearch
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryStoreSaveSearch:
    """
    Tests for save_search: UPSERT behaviour, query normalisation, and guards.
    Priority: IMPORTANT
    """

    def test_save_search_stores_entry(self, history_db):
        """
        A saved search appears in list_recent_searches for the same user.
        """
        save_search(history_db, "u1", "transformers", "Research", "Semantic Search", 5, 3)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 1
        assert entries[0]["query"] == "transformers"

    def test_save_search_empty_query_is_ignored(self, history_db):
        """
        An empty query string is not persisted.
        """
        save_search(history_db, "u1", "", "Research", "Semantic Search", 5, 0)
        entries = list_recent_searches(history_db, "u1")
        assert entries == []

    def test_save_search_whitespace_only_query_is_ignored(self, history_db):
        """
        A whitespace-only query collapses to empty and is not persisted.
        """
        save_search(history_db, "u1", "   ", "Research", "Semantic Search", 5, 0)
        entries = list_recent_searches(history_db, "u1")
        assert entries == []

    def test_save_search_upsert_same_query_updates_row(self, history_db):
        """
        Saving the same query twice for the same user produces one row (UPSERT).
        """
        save_search(history_db, "u1", "attention", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u1", "attention", "Blog", "RAG Answer", 3, 1)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 1
        assert entries[0]["domain"] == "Blog"  # updated

    def test_save_search_upsert_is_case_insensitive(self, history_db):
        """
        'Attention' and 'attention' are treated as the same query (COLLATE NOCASE).
        """
        save_search(history_db, "u1", "Attention", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u1", "attention", "Blog", "Semantic Search", 5, 1)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 1

    def test_save_search_different_queries_produce_separate_rows(self, history_db):
        """
        Different query strings are stored as separate rows.
        """
        save_search(history_db, "u1", "transformers", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u1", "attention", "Research", "Semantic Search", 5, 2)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 2

    def test_clear_history_removes_all_entries_for_user(self, history_db):
        """
        clear_history() deletes all saved searches for the given user.
        """
        save_search(history_db, "u1", "query1", "Research", "Semantic Search", 5, 1)
        save_search(history_db, "u1", "query2", "Blog", "Semantic Search", 5, 2)
        clear_history(history_db, "u1")
        entries = list_recent_searches(history_db, "u1")
        assert entries == []

    def test_clear_history_does_not_affect_other_users(self, history_db):
        """
        Clearing user A's history does not delete user B's searches.
        """
        save_search(history_db, "user_A", "query", "Research", "Semantic Search", 5, 1)
        save_search(history_db, "user_B", "query", "Research", "Semantic Search", 5, 1)
        clear_history(history_db, "user_A")
        b_entries = list_recent_searches(history_db, "user_B")
        assert len(b_entries) == 1


# ══════════════════════════════════════════════════════════════════════════════
# TestHistoryStoreRecentSearches
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryStoreRecentSearches:
    """
    Tests for list_recent_searches: ordering, limit, and user scoping.
    Priority: IMPORTANT
    """

    def test_empty_on_fresh_db(self, history_db):
        """
        list_recent_searches returns an empty list on a fresh database.
        """
        assert list_recent_searches(history_db, "u1") == []

    def test_returns_entries_for_correct_user(self, history_db):
        """
        Only the queries saved by the requesting user are returned.
        """
        save_search(history_db, "u1", "my query", "Research", "Semantic Search", 5, 2)
        save_search(history_db, "u2", "other query", "Blog", "Semantic Search", 5, 1)

        u1_entries = list_recent_searches(history_db, "u1")
        assert len(u1_entries) == 1
        assert u1_entries[0]["query"] == "my query"

    def test_ordered_newest_first(self, history_db):
        """
        The most recently saved query appears first in the list.
        """
        save_search(history_db, "u1", "first query", "Research", "Semantic Search", 5, 1)
        save_search(history_db, "u1", "second query", "Blog", "Semantic Search", 5, 2)

        entries = list_recent_searches(history_db, "u1")
        assert entries[0]["query"] == "second query"

    def test_default_limit_is_20(self, history_db):
        """
        With 25 unique searches, list_recent_searches returns at most 20 (default).
        """
        for i in range(25):
            save_search(history_db, "u1", f"unique query {i}", "Research", "Semantic Search", 5, i)

        entries = list_recent_searches(history_db, "u1")
        assert len(entries) <= 20

    def test_custom_limit_is_respected(self, history_db):
        """
        A custom limit parameter caps the number of returned entries.
        """
        for i in range(10):
            save_search(history_db, "u1", f"query {i}", "Research", "Semantic Search", 5, i)

        entries = list_recent_searches(history_db, "u1", limit=3)
        assert len(entries) <= 3

    def test_entries_contain_expected_fields(self, history_db):
        """
        Each returned entry contains id, query, domain, method, top_k, result_count.
        """
        save_search(history_db, "u1", "attention mechanism", "Research", "RAG Answer", 7, 4)
        entries = list_recent_searches(history_db, "u1")
        entry = entries[0]

        assert "id" in entry
        assert entry["query"] == "attention mechanism"
        assert entry["domain"] == "Research"
        assert entry["method"] == "RAG Answer"
        assert entry["top_k"] == 7
        assert entry["result_count"] == 4

    def test_searches_from_other_users_not_included(self, history_db):
        """
        Searches from other users are completely excluded.
        """
        save_search(history_db, "other_user", "secret query", "Research", "Semantic Search", 5, 0)
        entries = list_recent_searches(history_db, "u1")
        assert entries == []

    def test_oldest_entry_absent_when_over_limit(self, history_db):
        """
        When more than limit entries exist, the oldest query is not returned.
        """
        for i in range(5):
            save_search(history_db, "u1", f"unique query {i}", "Research", "Semantic Search", 5, i)

        entries = list_recent_searches(history_db, "u1", limit=3)
        returned_queries = [e["query"] for e in entries]
        # The oldest (query 0) should not appear when limit=3
        assert "unique query 0" not in returned_queries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
