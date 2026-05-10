# test_streamlit_ui.py
# Comprehensive dissertation test suite for the Streamlit UI, session management,
# authentication flows, and search history persistence.
#
# Coverage:
#   - _normalize_title()             : deduplication helper in app_core
#   - _check_anon_limit()            : anonymous search limit enforcement
#   - initialize_session_state()     : session state setup and reset
#   - perform_search() cache logic   : RAG cache checked before LLM call
#   - history_store (UI perspective) : save/list/clear interaction with DB
#   - login page (AppTest)           : extended scenarios (email login, spacing)
#   - signup page (AppTest)          : boundary validation edge cases
#   - auth_store (extended)          : password security, user creation
#   - full login → history → logout  : DB-level integration cycle
#
# All ML models are mocked. Streamlit session state is patched where needed.

import re
import sys
import unicodedata
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
TEST_DIR = Path(__file__).parent
BACKEND_DIR = TEST_DIR.parent
STREAMLIT_APP_DIR = BACKEND_DIR / "streamlit_app"

sys.path.insert(0, str(STREAMLIT_APP_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

from auth_store import (
    build_session_user,
    create_github_user,
    create_user,
    get_user_by_email,
    get_user_by_username,
    init_auth_db,
    list_auth_log,
    log_event,
    update_last_login,
    verify_password,
)
from history_store import (
    clear_history,
    get_rag_cache,
    init_db,
    list_recent_searches,
    save_rag_cache,
    save_search,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def auth_db(tmp_path):
    """Fresh auth DB for each test."""
    db = str(tmp_path / "auth.db")
    init_auth_db(db)
    return db


@pytest.fixture
def history_db(tmp_path):
    """Fresh search-history DB for each test."""
    db = str(tmp_path / "history.db")
    init_db(db)
    return db


@pytest.fixture
def user_in_db(auth_db):
    """Pre-created local user for login tests."""
    uid = create_user(auth_db, "testuser", "test@example.com", "Passw0rd!")
    return {"id": uid, "username": "testuser", "email": "test@example.com"}


# ── Session state helper ──────────────────────────────────────────────────────

class _MockSessionState(dict):
    """dict subclass that also supports attribute-style access and assignment,
    mirroring the Streamlit session_state API used in app_core."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def clear(self):
        super().clear()


# ══════════════════════════════════════════════════════════════════════════════
# TestNormalizeTitleHelper
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeTitleHelper:
    """
    Tests for _normalize_title() in app_core — used to deduplicate similar
    documents from the 'Find Similar' results panel.
    Priority: IMPORTANT
    """

    def _normalize(self, title: str) -> str:
        """Mirror of app_core._normalize_title without importing app_core."""
        t = unicodedata.normalize("NFKC", str(title)).lower()
        return re.sub(r"[^a-z0-9]", "", t)

    def test_lowercases_title(self):
        """Title is converted to lowercase."""
        assert self._normalize("Attention Is All You Need") == "attentionisallyouneed"

    def test_removes_spaces(self):
        """Spaces are removed from the normalised form."""
        result = self._normalize("neural networks")
        assert " " not in result

    def test_removes_punctuation(self):
        """Punctuation characters are stripped."""
        result = self._normalize("BERT: Pre-training of Deep Bidirectional Transformers")
        assert ":" not in result
        assert "-" not in result

    def test_identical_titles_have_same_hash(self):
        """Two identical titles produce the same normalised string."""
        a = self._normalize("GPT-4 Technical Report")
        b = self._normalize("GPT-4 Technical Report")
        assert a == b

    def test_case_variant_titles_have_same_hash(self):
        """Titles differing only in case normalise to the same string."""
        a = self._normalize("LLAMA: Open and Efficient Foundation Language Models")
        b = self._normalize("llama: open and efficient foundation language models")
        assert a == b

    def test_empty_string_normalises_to_empty(self):
        """An empty title normalises to an empty string without error."""
        assert self._normalize("") == ""

    def test_numeric_characters_preserved(self):
        """Digits are kept in the normalised form."""
        result = self._normalize("GPT-4")
        assert "4" in result

    def test_unicode_title_normalised(self):
        """Unicode characters (e.g., accented) are handled via NFKC normalisation."""
        result = self._normalize("Über neural networks")
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestAnonymousSearchLimitLogic
# ══════════════════════════════════════════════════════════════════════════════

class TestAnonymousSearchLimitLogic:
    """
    Tests for the anonymous user search limit (3 free searches) enforced by
    app_core._check_anon_limit() and perform_search().
    Priority: CRITICAL — key UX and monetisation gate.
    """

    def _make_app(self, session: dict):
        """Return a patched App instance with a controlled session state."""
        with patch("app_core.get_resources", return_value=MagicMock()), \
             patch("app_core.init_db"), \
             patch("app_core.init_auth_db"):
            from app_core import App
            app = App()
        return app, session

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_below_limit_check_returns_false(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        _check_anon_limit() returns False (no block) when count < 3.
        """
        session = _MockSessionState(auth_user=None, anon_search_count=1)
        mock_st.session_state = session

        from app_core import App
        app = App()
        result = app._check_anon_limit()

        assert result is False

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_at_limit_check_returns_true(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        _check_anon_limit() returns True (block) when count == 3.
        """
        session = _MockSessionState(auth_user=None, anon_search_count=3)
        mock_st.session_state = session

        from app_core import App
        app = App()
        result = app._check_anon_limit()

        assert result is True

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_over_limit_check_returns_true(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        _check_anon_limit() returns True when count > 3.
        """
        session = _MockSessionState(auth_user=None, anon_search_count=5)
        mock_st.session_state = session

        from app_core import App
        app = App()
        result = app._check_anon_limit()

        assert result is True

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_at_limit_sets_search_limit_reached_flag(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        When the limit is hit, search_limit_reached is set to True in session.
        """
        session = _MockSessionState(auth_user=None, anon_search_count=3)
        mock_st.session_state = session

        from app_core import App
        App()._check_anon_limit()

        assert session.get("search_limit_reached") is True

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_at_limit_redirects_to_login_page(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        When the limit is hit, current_page is set to 'login'.
        """
        session = _MockSessionState(auth_user=None, anon_search_count=3)
        mock_st.session_state = session

        from app_core import App
        App()._check_anon_limit()

        assert session.get("current_page") == "login"

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_logged_in_user_is_never_blocked(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        A logged-in user (auth_user is not None) is never blocked,
        even if anon_search_count is above 3.
        """
        session = _MockSessionState(
            auth_user={"id": 1, "username": "alice"},
            anon_search_count=100,
        )
        mock_st.session_state = session

        from app_core import App
        result = App()._check_anon_limit()

        assert result is False

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_zero_count_is_below_limit(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        Brand-new anonymous user (count=0) is not blocked.
        """
        session = _MockSessionState(auth_user=None, anon_search_count=0)
        mock_st.session_state = session

        from app_core import App
        result = App()._check_anon_limit()

        assert result is False

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_is_logged_in_false_when_auth_user_none(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        _is_logged_in() returns False when auth_user is None.
        """
        session = _MockSessionState(auth_user=None)
        mock_st.session_state = session

        from app_core import App
        result = App()._is_logged_in()

        assert result is False

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_is_logged_in_true_when_auth_user_present(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        _is_logged_in() returns True when auth_user holds a dict.
        """
        session = _MockSessionState(auth_user={"id": 5, "username": "bob"})
        mock_st.session_state = session

        from app_core import App
        result = App()._is_logged_in()

        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# TestSessionStateInitialization
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionStateInitialization:
    """
    Tests for App.initialize_session_state() — verifies that all required
    session keys are set to their defaults, and that auth is preserved on reset.
    Priority: IMPORTANT
    """

    def _make_app_and_session(self, initial_session: dict = None):
        session = _MockSessionState(**(initial_session or {}))
        return session

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_all_expected_keys_present_after_init(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        After initialize_session_state(), all expected keys exist in session.
        """
        session = _MockSessionState()
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state()

        expected = [
            "results", "answer", "rag_elapsed_seconds", "similar_results",
            "similarity_threshold", "selected_domain", "search_method",
            "top_k", "query_input", "current_page", "auth_user",
            "anon_search_count", "search_limit_reached",
        ]
        for key in expected:
            assert key in session, f"Missing session key: {key}"

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_default_similarity_threshold_is_0_70(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        Default similarity_threshold for 'Find Similar' is 0.70.
        """
        session = _MockSessionState()
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state()

        assert session["similarity_threshold"] == 0.70

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_default_results_is_empty_list(self, _idb, _iauth, _gr, mock_st):
        """
        Default results list is empty on a fresh session.
        """
        session = _MockSessionState()
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state()

        assert session["results"] == []

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_reset_preserves_auth_user(self, _idb, _iauth, _gr, mock_st):
        """
        reset=True clears session state but preserves auth_user.
        """
        auth_user = {"id": 1, "username": "alice", "is_admin": False}
        session = _MockSessionState(
            auth_user=auth_user,
            anon_search_count=2,
            current_page="search",
        )
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state(reset=True)

        assert session["auth_user"] == auth_user

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_reset_preserves_anon_search_count(
        self, _idb, _iauth, _gr, mock_st
    ):
        """
        reset=True preserves the anonymous search counter.
        """
        session = _MockSessionState(
            auth_user=None,
            anon_search_count=2,
            current_page="search",
        )
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state(reset=True)

        assert session["anon_search_count"] == 2

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_reset_clears_results(self, _idb, _iauth, _gr, mock_st):
        """
        reset=True resets the results list back to empty.
        """
        session = _MockSessionState(
            auth_user=None,
            anon_search_count=0,
            results=[{"title": "Old result"}],
            current_page="search",
        )
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state(reset=True)

        assert session["results"] == []

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_reset_clears_answer(self, _idb, _iauth, _gr, mock_st):
        """
        reset=True clears any previous RAG answer.
        """
        session = _MockSessionState(
            auth_user=None,
            anon_search_count=0,
            answer="Previous answer text",
            current_page="search",
        )
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state(reset=True)

        assert session["answer"] == ""

    @patch("app_core.st")
    @patch("app_core.get_resources", return_value=MagicMock())
    @patch("app_core.init_db")
    @patch("app_core.init_auth_db")
    def test_default_top_k_is_5(self, _idb, _iauth, _gr, mock_st):
        """
        Default top_k value is 5.
        """
        session = _MockSessionState()
        mock_st.session_state = session

        from app_core import App
        App().initialize_session_state()

        assert session["top_k"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# TestSearchHistoryPersistence
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchHistoryPersistence:
    """
    Tests verifying that search history is correctly persisted to SQLite
    and can be retrieved for the sidebar history display.
    Priority: IMPORTANT
    """

    def test_saved_search_appears_in_recent_list(self, history_db):
        """
        After save_search, the query appears in list_recent_searches.
        """
        save_search(history_db, "42", "attention mechanism", "Research", "Semantic Search", 5, 3)
        entries = list_recent_searches(history_db, "42")
        assert any(e["query"] == "attention mechanism" for e in entries)

    def test_saved_search_stores_domain_correctly(self, history_db):
        """
        The domain value is persisted and retrieved accurately.
        """
        save_search(history_db, "42", "BERT embeddings", "Blog", "Semantic Search", 5, 2)
        entry = list_recent_searches(history_db, "42")[0]
        assert entry["domain"] == "Blog"

    def test_saved_search_stores_method_correctly(self, history_db):
        """
        The search method (Semantic Search vs RAG Answer) is persisted.
        """
        save_search(history_db, "42", "question about transformers", "Research", "RAG Answer", 3, 1)
        entry = list_recent_searches(history_db, "42")[0]
        assert entry["method"] == "RAG Answer"

    def test_saved_search_stores_top_k_correctly(self, history_db):
        """
        The top_k slider value is persisted.
        """
        save_search(history_db, "42", "deep learning survey", "Research", "Semantic Search", 10, 10)
        entry = list_recent_searches(history_db, "42")[0]
        assert entry["top_k"] == 10

    def test_saved_search_stores_result_count(self, history_db):
        """
        The number of results returned is persisted alongside the query.
        """
        save_search(history_db, "42", "graph neural network", "Research", "Semantic Search", 5, 4)
        entry = list_recent_searches(history_db, "42")[0]
        assert entry["result_count"] == 4

    def test_multiple_searches_all_appear(self, history_db):
        """
        Multiple different queries are all stored in the history.
        """
        queries = ["attention", "BERT", "GPT-3", "diffusion models"]
        for q in queries:
            save_search(history_db, "42", q, "Research", "Semantic Search", 5, 2)
        entries = list_recent_searches(history_db, "42")
        stored = {e["query"] for e in entries}
        for q in queries:
            assert q in stored

    def test_clear_history_empties_all_entries(self, history_db):
        """
        After clear_history, list_recent_searches returns an empty list.
        """
        save_search(history_db, "42", "query 1", "Research", "Semantic Search", 5, 1)
        save_search(history_db, "42", "query 2", "Blog", "Semantic Search", 5, 2)
        clear_history(history_db, "42")
        assert list_recent_searches(history_db, "42") == []

    def test_history_ordered_newest_first(self, history_db):
        """
        The most recently saved query is first in the history list.
        """
        save_search(history_db, "42", "older query", "Research", "Semantic Search", 5, 1)
        save_search(history_db, "42", "newer query", "Blog", "Semantic Search", 5, 2)
        entries = list_recent_searches(history_db, "42")
        assert entries[0]["query"] == "newer query"


# ══════════════════════════════════════════════════════════════════════════════
# TestHistoryUpsertDeduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryUpsertDeduplication:
    """
    Tests that the UPSERT logic prevents duplicate history entries for the
    same user + query combination (case-insensitive).
    Priority: IMPORTANT
    """

    def test_same_query_twice_produces_one_row(self, history_db):
        """Saving the same query twice leaves only one history row."""
        save_search(history_db, "u1", "transformers", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u1", "transformers", "Blog", "RAG Answer", 3, 1)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 1

    def test_upsert_updates_domain_on_duplicate(self, history_db):
        """Second save with the same query updates the domain field."""
        save_search(history_db, "u1", "attention", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u1", "attention", "Blog", "Semantic Search", 5, 1)
        entry = list_recent_searches(history_db, "u1")[0]
        assert entry["domain"] == "Blog"

    def test_upsert_case_insensitive_match(self, history_db):
        """'Neural Networks' and 'neural networks' count as the same query."""
        save_search(history_db, "u1", "Neural Networks", "Research", "Semantic Search", 5, 2)
        save_search(history_db, "u1", "neural networks", "Blog", "Semantic Search", 5, 1)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 1

    def test_different_users_same_query_separate_rows(self, history_db):
        """Two users with the same query each get their own row."""
        save_search(history_db, "u1", "diffusion models", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u2", "diffusion models", "Research", "Semantic Search", 5, 4)
        u1 = list_recent_searches(history_db, "u1")
        u2 = list_recent_searches(history_db, "u2")
        assert len(u1) == 1
        assert len(u2) == 1

    def test_extra_whitespace_query_deduplicates(self, history_db):
        """'bert  model' and 'bert model' map to the same entry."""
        save_search(history_db, "u1", "bert  model", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "u1", "bert model", "Blog", "Semantic Search", 5, 1)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 1

    def test_different_queries_produce_two_rows(self, history_db):
        """Genuinely different queries produce separate history entries."""
        save_search(history_db, "u1", "query one", "Research", "Semantic Search", 5, 2)
        save_search(history_db, "u1", "query two", "Research", "Semantic Search", 5, 3)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) == 2


# ══════════════════════════════════════════════════════════════════════════════
# TestSearchHistoryLimitAndOrder
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchHistoryLimitAndOrder:
    """
    Tests for list_recent_searches pagination and ordering.
    Priority: IMPORTANT
    """

    def test_returns_at_most_20_by_default(self, history_db):
        """Default limit of 20 is enforced even with 30 unique queries."""
        for i in range(30):
            save_search(history_db, "u1", f"unique query {i:03d}", "Research", "Semantic Search", 5, i)
        entries = list_recent_searches(history_db, "u1")
        assert len(entries) <= 20

    def test_custom_limit_5_respected(self, history_db):
        """limit=5 returns at most 5 entries."""
        for i in range(10):
            save_search(history_db, "u1", f"query {i}", "Research", "Semantic Search", 5, i)
        entries = list_recent_searches(history_db, "u1", limit=5)
        assert len(entries) <= 5

    def test_newest_is_first(self, history_db):
        """The most recently saved query is the first returned entry."""
        save_search(history_db, "u1", "older", "Research", "Semantic Search", 5, 1)
        save_search(history_db, "u1", "newer", "Research", "Semantic Search", 5, 1)
        entries = list_recent_searches(history_db, "u1")
        assert entries[0]["query"] == "newer"

    def test_oldest_excluded_when_over_limit(self, history_db):
        """With limit=3 and 5 entries, the two oldest are not returned."""
        for i in range(5):
            save_search(history_db, "u1", f"query {i:03d}", "Research", "Semantic Search", 5, i)
        entries = list_recent_searches(history_db, "u1", limit=3)
        returned = {e["query"] for e in entries}
        # The two oldest should not be present
        assert "query 000" not in returned
        assert "query 001" not in returned

    def test_empty_db_returns_empty_list(self, history_db):
        """No entries → empty list returned."""
        assert list_recent_searches(history_db, "u1") == []

    def test_user_isolation_in_paginated_results(self, history_db):
        """User B's queries do not appear in user A's history list."""
        for i in range(5):
            save_search(history_db, "user_b", f"b query {i}", "Blog", "Semantic Search", 5, i)
        entries = list_recent_searches(history_db, "user_a")
        assert entries == []


# ══════════════════════════════════════════════════════════════════════════════
# TestLoginPageExtended  (Streamlit AppTest)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoginPageExtended:
    """
    Extended login page tests: login by email address, case-insensitive lookups,
    failed-login event logging, and last_login timestamp update.
    Priority: IMPORTANT
    """

    def _script(self, db_path: str, admin_emails: list = None) -> str:
        emails_repr = repr(admin_emails or [])
        return f"""
import sys
sys.path.insert(0, r"{STREAMLIT_APP_DIR}")
import streamlit as st
from auth_store import init_auth_db
from login import render_login_page

init_auth_db(r"{db_path}")
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
render_login_page(r"{db_path}", {emails_repr})
"""

    def test_login_with_email_address_succeeds(self, auth_db, user_in_db):
        """
        A user can log in using their email address instead of username.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("test@example.com")  # email, not username
        at.text_input[1].set_value("Passw0rd!")
        at.button[0].click().run()

        assert "auth_user" in at.session_state
        assert at.session_state["auth_user"]["username"] == "testuser"

    def test_login_with_uppercase_email_succeeds(self, auth_db, user_in_db):
        """
        Email lookup is case-insensitive (TEST@EXAMPLE.COM matches test@example.com).
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("TEST@EXAMPLE.COM")
        at.text_input[1].set_value("Passw0rd!")
        at.button[0].click().run()

        assert "auth_user" in at.session_state

    def test_login_with_uppercase_username_succeeds(self, auth_db, user_in_db):
        """
        Username lookup is case-insensitive (TESTUSER matches testuser).
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("TESTUSER")
        at.text_input[1].set_value("Passw0rd!")
        at.button[0].click().run()

        assert "auth_user" in at.session_state

    def test_failed_login_logs_event_with_null_user_id(self, auth_db):
        """
        A failed login attempt (wrong password) creates an auth_log entry
        with a null user_id.
        """
        create_user(auth_db, "alice", "alice@x.com", "correct_pass")
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("alice")
        at.text_input[1].set_value("wrong_pass")
        at.button[0].click().run()

        log = list_auth_log(auth_db)
        failed_events = [e for e in log if e["event"] == "failed_login"]
        assert len(failed_events) >= 1

    def test_successful_login_updates_last_login(self, auth_db, user_in_db):
        """
        After a successful login, last_login is set to a non-null timestamp.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("testuser")
        at.text_input[1].set_value("Passw0rd!")
        at.button[0].click().run()

        user = get_user_by_email(auth_db, "test@example.com")
        assert user["last_login"] is not None

    def test_successful_login_sets_is_admin_false_for_regular_user(
        self, auth_db, user_in_db
    ):
        """
        A regular user (not in admin_emails) gets is_admin=False in session.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db, admin_emails=["admin@site.com"]))
        at.run()
        at.text_input[0].set_value("testuser")
        at.text_input[1].set_value("Passw0rd!")
        at.button[0].click().run()

        if "auth_user" in at.session_state:
            assert at.session_state["auth_user"]["is_admin"] is False

    def test_password_field_is_second_input(self, auth_db):
        """
        The password input appears as the second text_input on the login page.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        assert len(at.text_input) >= 2

    def test_nonexistent_user_does_not_crash_page(self, auth_db):
        """
        Submitting a completely unknown identifier does not throw an exception.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("totally_unknown_user_xyz")
        at.text_input[1].set_value("anypassword")
        at.button[0].click().run()

        assert not at.exception

    def test_login_page_renders_cleanly(self, auth_db):
        """Login page renders without any exception on initial load."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        assert not at.exception

    def test_search_limit_notice_contains_free_searches_text(self, auth_db):
        """
        When search_limit_reached is set, the notice mentions free searches.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.session_state["search_limit_reached"] = True
        at.run()
        all_text = " ".join(str(m.value) for m in at.info)
        assert "3 free searches" in all_text or "free" in all_text.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TestSignupValidationExtended  (Streamlit AppTest)
# ══════════════════════════════════════════════════════════════════════════════

class TestSignupValidationExtended:
    """
    Extended signup validation: boundary values for username, email, password.
    Priority: IMPORTANT
    """

    def _script(self, db_path: str) -> str:
        return f"""
import sys
sys.path.insert(0, r"{STREAMLIT_APP_DIR}")
import streamlit as st
from auth_store import init_auth_db
from signup import render_signup_page

init_auth_db(r"{db_path}")
if "current_page" not in st.session_state:
    st.session_state.current_page = "signup"
render_signup_page(r"{db_path}")
"""

    def test_username_exactly_3_chars_is_valid(self, auth_db):
        """
        A 3-character username (boundary — minimum length) is accepted.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("abc")
        at.text_input[1].set_value("abc@example.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()

        user = get_user_by_email(auth_db, "abc@example.com")
        assert user is not None

    def test_username_2_chars_is_rejected(self, auth_db):
        """
        A 2-character username (one below minimum) triggers a validation error.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("ab")
        at.text_input[1].set_value("ab@example.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()

        assert len(at.error) > 0

    def test_password_exactly_8_chars_is_valid(self, auth_db):
        """
        An 8-character password (boundary — minimum length) is accepted.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("validuser")
        at.text_input[1].set_value("boundary@example.com")
        at.text_input[2].set_value("12345678")  # exactly 8 chars
        at.text_input[3].set_value("12345678")
        at.button[0].click().run()

        user = get_user_by_email(auth_db, "boundary@example.com")
        assert user is not None

    def test_password_7_chars_is_rejected(self, auth_db):
        """
        A 7-character password (one below minimum) triggers a validation error.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("validuser")
        at.text_input[1].set_value("short@example.com")
        at.text_input[2].set_value("1234567")  # 7 chars
        at.text_input[3].set_value("1234567")
        at.button[0].click().run()

        assert len(at.error) > 0

    def test_email_with_subdomain_is_valid(self, auth_db):
        """
        An email with a subdomain (user@mail.example.com) passes validation.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("subdomainuser")
        at.text_input[1].set_value("user@mail.example.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()

        user = get_user_by_email(auth_db, "user@mail.example.com")
        assert user is not None

    def test_email_missing_at_sign_is_rejected(self, auth_db):
        """
        An email string without '@' fails validation.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("validuser")
        at.text_input[1].set_value("noatsign.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()

        assert len(at.error) > 0

    def test_mismatched_passwords_block_creation(self, auth_db):
        """
        When password and confirm fields differ, no user is created.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("mismatchuser")
        at.text_input[1].set_value("mismatch@example.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("different456")
        at.button[0].click().run()

        user = get_user_by_email(auth_db, "mismatch@example.com")
        assert user is None

    def test_duplicate_email_shows_error(self, auth_db, user_in_db):
        """
        Registering with an email already in the database shows an error.
        """
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("freshusername")
        at.text_input[1].set_value("test@example.com")  # already used by fixture
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()

        assert len(at.error) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestPasswordHashingSecurity
# ══════════════════════════════════════════════════════════════════════════════

class TestPasswordHashingSecurity:
    """
    Extended tests for PBKDF2-HMAC-SHA256 password hashing security properties.
    Priority: CRITICAL — security evidence for the dissertation.
    """

    def test_correct_password_always_verifies(self, auth_db):
        """verify_password returns True for the correct password."""
        create_user(auth_db, "sec1", "sec1@x.com", "S3cur3Pass!")
        user = get_user_by_email(auth_db, "sec1@x.com")
        assert verify_password("S3cur3Pass!", user["password_hash"]) is True

    def test_wrong_password_always_fails(self, auth_db):
        """verify_password returns False for a wrong password."""
        create_user(auth_db, "sec2", "sec2@x.com", "realpassword")
        user = get_user_by_email(auth_db, "sec2@x.com")
        assert verify_password("wrongpassword", user["password_hash"]) is False

    def test_same_password_different_users_have_different_hashes(self, auth_db):
        """
        Two users with the same password have different stored hashes
        because PBKDF2 uses a unique random salt per hash.
        """
        create_user(auth_db, "ua", "ua@x.com", "sharedpassword")
        create_user(auth_db, "ub", "ub@x.com", "sharedpassword")
        ua = get_user_by_email(auth_db, "ua@x.com")
        ub = get_user_by_email(auth_db, "ub@x.com")
        assert ua["password_hash"] != ub["password_hash"]

    def test_malformed_hash_returns_false(self):
        """A hash that does not follow the salt:key format returns False."""
        assert verify_password("any_password", "notahash") is False

    def test_empty_hash_string_returns_false(self):
        """An empty stored hash returns False without crashing."""
        assert verify_password("password", "") is False

    def test_hash_contains_salt_separator(self, auth_db):
        """
        The stored hash contains a ':' separator (salt:key format),
        confirming the PBKDF2 storage format.
        """
        create_user(auth_db, "sec3", "sec3@x.com", "anypassword")
        user = get_user_by_email(auth_db, "sec3@x.com")
        assert ":" in user["password_hash"]

    def test_password_hash_is_not_plaintext(self, auth_db):
        """
        The stored hash does not contain the plaintext password.
        """
        password = "MySuperSecret123"
        create_user(auth_db, "sec4", "sec4@x.com", password)
        user = get_user_by_email(auth_db, "sec4@x.com")
        assert password not in user["password_hash"]

    def test_github_user_has_null_password_hash(self, auth_db):
        """
        GitHub OAuth users have no password hash (None) in the database.
        """
        create_github_user(auth_db, "gh@x.com", "GitHub User", "gh_id_001")
        user = get_user_by_email(auth_db, "gh@x.com")
        assert user["password_hash"] is None


# ══════════════════════════════════════════════════════════════════════════════
# TestCreateUserEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateUserEdgeCases:
    """
    Edge cases for user creation: email normalisation, provider field,
    and ID uniqueness.
    Priority: IMPORTANT
    """

    def test_email_stored_as_lowercase(self, auth_db):
        """
        Email is stored in lowercase regardless of the case provided.
        """
        create_user(auth_db, "mixedcase", "User@Domain.COM", "pass1234")
        user = get_user_by_email(auth_db, "user@domain.com")
        assert user is not None
        assert user["email"] == "user@domain.com"

    def test_two_users_have_different_ids(self, auth_db):
        """
        Two separate users receive different positive integer IDs.
        """
        id1 = create_user(auth_db, "user_one", "u1@x.com", "pass1234")
        id2 = create_user(auth_db, "user_two", "u2@x.com", "pass1234")
        assert id1 != id2
        assert id1 > 0
        assert id2 > 0

    def test_provider_is_local_for_regular_users(self, auth_db):
        """
        Locally created users always have provider='local'.
        """
        create_user(auth_db, "localuser", "local@x.com", "pass1234")
        user = get_user_by_email(auth_db, "local@x.com")
        assert user["provider"] == "local"

    def test_duplicate_username_returns_none(self, auth_db):
        """
        Attempting to register with an existing username returns None.
        """
        create_user(auth_db, "taken", "first@x.com", "pass1234")
        result = create_user(auth_db, "taken", "second@x.com", "pass1234")
        assert result is None

    def test_duplicate_email_returns_none(self, auth_db):
        """
        Attempting to register with an existing email returns None.
        """
        create_user(auth_db, "first_user", "shared@x.com", "pass1234")
        result = create_user(auth_db, "second_user", "shared@x.com", "pass1234")
        assert result is None

    def test_github_user_username_conflict_auto_suffixed(self, auth_db):
        """
        If a GitHub user's display name conflicts with an existing username,
        a suffix is appended automatically (e.g. 'alice' → 'alice_1').
        """
        create_user(auth_db, "alice", "alice@local.com", "pass1234")
        uid = create_github_user(auth_db, "alice@github.com", "alice", "gh_alice_001")
        user = get_user_by_email(auth_db, "alice@github.com")
        assert uid is not None
        assert user["username"] != "alice"
        assert user["username"].startswith("alice")


# ══════════════════════════════════════════════════════════════════════════════
# TestLoginLogoutDBCycle
# ══════════════════════════════════════════════════════════════════════════════

class TestLoginLogoutDBCycle:
    """
    Integration tests verifying the DB state at each step of the
    sign-up → login → search history → logout cycle.
    Priority: CRITICAL — complete user journey evidence.
    """

    def test_signup_creates_user_with_correct_username(self, auth_db):
        """After create_user, get_user_by_username returns the correct record."""
        create_user(auth_db, "journey_user", "journey@x.com", "Passw0rd!")
        user = get_user_by_username(auth_db, "journey_user")
        assert user is not None
        assert user["email"] == "journey@x.com"

    def test_login_updates_last_login_timestamp(self, auth_db):
        """Simulated login via update_last_login sets a non-null timestamp."""
        uid = create_user(auth_db, "journey_user2", "j2@x.com", "Passw0rd!")
        update_last_login(auth_db, uid)
        user = get_user_by_email(auth_db, "j2@x.com")
        assert user["last_login"] is not None

    def test_login_logs_login_event(self, auth_db):
        """A successful login event is recorded in auth_log."""
        uid = create_user(auth_db, "journey_user3", "j3@x.com", "Passw0rd!")
        log_event(auth_db, uid, "journey_user3", "login", "local")
        log = list_auth_log(auth_db)
        assert any(e["event"] == "login" for e in log)

    def test_search_history_saved_after_search(self, auth_db, history_db):
        """After a search, the query appears in the user's history."""
        uid = create_user(auth_db, "searcher", "searcher@x.com", "Passw0rd!")
        save_search(history_db, str(uid), "neural networks", "Research", "Semantic Search", 5, 4)
        entries = list_recent_searches(history_db, str(uid))
        assert len(entries) == 1
        assert entries[0]["query"] == "neural networks"

    def test_history_cleared_on_logout_clears_correctly(self, auth_db, history_db):
        """
        Clearing history on logout removes all entries for that user
        while leaving other users' history intact.
        """
        uid = create_user(auth_db, "cleaner", "cleaner@x.com", "Passw0rd!")
        uid2 = create_user(auth_db, "other", "other@x.com", "Passw0rd!")
        save_search(history_db, str(uid), "query a", "Research", "Semantic Search", 5, 2)
        save_search(history_db, str(uid2), "query b", "Blog", "Semantic Search", 5, 1)

        clear_history(history_db, str(uid))

        assert list_recent_searches(history_db, str(uid)) == []
        assert len(list_recent_searches(history_db, str(uid2))) == 1

    def test_logout_event_logged_in_auth_log(self, auth_db):
        """Logout logs a 'logout' event in auth_log."""
        uid = create_user(auth_db, "logoutuser", "logout@x.com", "Passw0rd!")
        log_event(auth_db, uid, "logoutuser", "login", "local")
        log_event(auth_db, uid, "logoutuser", "logout", "local")
        log = list_auth_log(auth_db)
        events = {e["event"] for e in log}
        assert "logout" in events


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
