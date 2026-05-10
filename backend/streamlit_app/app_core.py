import os
import re
import sys
import time
import unicodedata
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from paths import AUTH_DB_PATH, CSV_PATH, EMBEDDINGS_PATH, HISTORY_DB_PATH, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from search_engine import (
    load_resources,
    search,
    rag_answer,
    more_like_this,
)
from history_store import init_db, save_search, list_recent_searches, clear_history, save_rag_cache, get_rag_cache
from auth_store import init_auth_db, init_sessions_table, delete_session, log_event

csv_path = str(CSV_PATH)
embeddings_path = str(EMBEDDINGS_PATH)
encoding = "latin1"
history_db_path = str(HISTORY_DB_PATH)
auth_db_path = str(AUTH_DB_PATH)

# Anonymous users get this many free searches before login is required.
ANON_SEARCH_LIMIT = 3


@st.cache_resource(show_spinner=False)
def get_resources(csv_path: str, embeddings_path: str, encoding: str):
    return load_resources(csv_path, embeddings_path, encoding)


def _rag_worker(resources: Any, query: str, top_k: int, domain: str) -> dict[str, Any]:
    return rag_answer(resources, query, top_k=top_k, domain=domain)


def _get_admin_emails() -> list[str]:
    try:
        return list(st.secrets["admin"]["emails"])
    except Exception:
        return []


def _normalize_title(title: str) -> str:
    """Lowercase, strip accents, remove non-alphanumeric chars for dedup comparison."""
    t = unicodedata.normalize("NFKC", str(title)).lower()
    return re.sub(r"[^a-z0-9]", "", t)


class App:
    SIMILAR_CANDIDATE_BUDGET = 25
    HISTORY_LIMIT = 20

    def __init__(self) -> None:
        self.resources = get_resources(csv_path, embeddings_path, encoding)
        init_db(history_db_path)
        init_auth_db(auth_db_path)
        init_sessions_table(auth_db_path)
        self.executor = ThreadPoolExecutor(max_workers=2)

    # ---------- Session ----------

    def initialize_session_state(self, reset: bool = False) -> None:
        active_page = st.session_state.get("current_page", "landing")

        defaults = {
            "results": [],
            "answer": "",
            "rag_elapsed_seconds": None,
            "rag_error": "",
            "similar_results": [],
            "similar_candidates_total": 0,
            "similar_threshold_used": 0.70,
            "similarity_threshold": 0.70,
            "similar_cache": {},
            "selected_domain": "Research",
            "search_method": "Semantic Search",
            "top_k": 5,
            "query_input": "",
            "current_page": "landing",
            "history_selected_id": "",
            "history_applied_id": None,
            "search_performed": False,
            "history_search_pending": False,
            "rag_from_cache": False,
            # Auth
            "auth_user": None,
            "anon_search_count": 0,
            "search_limit_reached": False,
        }

        if reset:
            # Preserve auth across resets
            preserved_auth = st.session_state.get("auth_user")
            preserved_count = st.session_state.get("anon_search_count", 0)
            st.session_state.clear()
            defaults["auth_user"] = preserved_auth
            defaults["anon_search_count"] = preserved_count

        for key, value in defaults.items():
            if reset or key not in st.session_state:
                st.session_state[key] = value

        if reset:
            st.session_state.current_page = active_page

    # ---------- Auth helpers ----------

    def _current_user_id(self) -> str | None:
        user = st.session_state.get("auth_user")
        return str(user["id"]) if user else None

    def _is_logged_in(self) -> bool:
        return st.session_state.get("auth_user") is not None

    def _check_anon_limit(self) -> bool:
        """Returns True if the anonymous user has hit the search limit (should block)."""
        if self._is_logged_in():
            return False
        count = st.session_state.get("anon_search_count", 0)
        if count >= ANON_SEARCH_LIMIT:
            st.session_state.search_limit_reached = True
            st.session_state.current_page = "login"
            return True
        return False

    # ---------- Workers ----------

    def run_with_spinner(self, func: Callable, *args, **kwargs) -> Any:
        with st.spinner("Thinking..."):
            return func(*args, **kwargs)

    def run_with_progress(self, label: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
        container = st.empty()
        bar = container.progress(0, text=str(label))
        start = time.perf_counter()

        future = self.executor.submit(fn, *args, **kwargs)

        _STAGES_BY_SECONDS = [
            (0,  "Initializing…"),
            (3,  "Retrieving documents…"),
            (10, "Generating answer…"),
        ]
        _ESTIMATED_DURATION = 30.0  # seconds before bar fills; rewinds if exceeded
        _REWIND_TO = 60             # % the bar rewinds to after filling

        try:
            while not future.done():
                elapsed = time.perf_counter() - start

                if elapsed < _ESTIMATED_DURATION:
                    pct = min(int((elapsed / _ESTIMATED_DURATION) * 100), 99)
                    stage_label = next(
                        (s[1] for s in reversed(_STAGES_BY_SECONDS) if elapsed >= s[0]),
                        str(label),
                    )
                else:
                    # Bar filled before the answer arrived — rewind and refill slowly
                    refill_elapsed = (elapsed - _ESTIMATED_DURATION) % _ESTIMATED_DURATION
                    pct = _REWIND_TO + int((refill_elapsed / _ESTIMATED_DURATION) * (99 - _REWIND_TO))
                    stage_label = "Still working… this is taking longer than expected."

                bar.progress(pct, text=stage_label)
                time.sleep(1.0)

            result = future.result()
            bar.progress(100, text="Done")
            time.sleep(0.3)
            return result
        finally:
            container.empty()

    # ---------- Search actions ----------

    def execute_semantic_search(self, query: str, top_k: int, domain: str) -> None:
        results = self.run_with_progress("Searching...", search, self.resources, query, top_k, domain=domain)
        st.session_state.results = results
        st.session_state.similar_results = []
        st.session_state.similar_cache = {}
        st.session_state.rag_elapsed_seconds = None
        st.session_state.answer = ""
        st.session_state.search_performed = True

    def execute_rag_search(self, query: str, top_k: int, domain: str, user_id: str | None = None) -> None:
        st.session_state.rag_error = ""
        st.session_state.rag_elapsed_seconds = None
        st.session_state.answer = ""
        st.session_state.results = []
        st.session_state.similar_results = []
        st.session_state.similar_cache = {}
        st.session_state.rag_from_cache = False

        started_at = time.perf_counter()
        result = self.run_with_progress(
            "RAG Answering...", _rag_worker, self.resources, query, min(int(top_k), 7), domain
        )
        if result is not None:
            elapsed = time.perf_counter() - started_at
            st.session_state.answer = result.get("answer", "")
            st.session_state.results = result.get("results", [])
            st.session_state.rag_elapsed_seconds = elapsed
            if user_id:
                save_rag_cache(
                    history_db_path,
                    user_id,
                    query,
                    domain,
                    min(int(top_k), 7),
                    result.get("answer", ""),
                    result.get("results", []),
                    elapsed,
                )
        st.session_state.search_performed = True

    def execute_more_like_this(self, index: int, domain: str) -> None:
        threshold = float(st.session_state.get("similarity_threshold", 0.70))
        all_result_indices = {item.get("index") for item in st.session_state.get("results", [])}
        cache_key = (index, str(domain).strip().lower(), tuple(sorted(i for i in all_result_indices if i is not None)), round(threshold, 2))
        cache = st.session_state.get("similar_cache", {})

        if cache_key in cache:
            st.session_state.similar_results = cache[cache_key]
            st.session_state.current_page = "similar"
            return

        results = self.run_with_spinner(
            more_like_this,
            self.resources,
            index,
            top_k=self.SIMILAR_CANDIDATE_BUDGET,
            domain=domain,
        )

        # Build a set of normalized titles to exclude: seed paper + current result set
        seen_titles: set[str] = set()
        if 0 <= index < len(self.resources.titles):
            seen_titles.add(_normalize_title(self.resources.titles[index]))
        for item in st.session_state.get("results", []):
            seen_titles.add(_normalize_title(item.get("title", "")))

        filtered_results = [
            item for item in results
            if item.get("index") not in all_result_indices
            and float(item.get("score", 0.0)) > threshold
            and _normalize_title(item.get("title", "")) not in seen_titles
        ]

        cache[cache_key] = filtered_results
        st.session_state.similar_cache = cache
        st.session_state.similar_results = filtered_results
        st.session_state.similar_candidates_total = len(results)
        st.session_state.similar_threshold_used = threshold
        st.session_state.current_page = "similar"

    # ---------- Navigation ----------

    def go_to_search(self) -> None:
        st.session_state.current_page = "search"

    def go_to_landing(self) -> None:
        self.initialize_session_state(reset=True)
        st.session_state.current_page = "landing"

    # ---------- Core search flow ----------

    def perform_search(self) -> None:
        if self._check_anon_limit():
            return

        query_text = st.session_state.get("query_input", "")
        selected_domain_value = st.session_state.get("selected_domain", "Research")
        search_method_value = st.session_state.get("search_method", "Semantic Search")
        top_k_value = st.session_state.get("top_k", 5)

        if not query_text:
            st.session_state.results = []
            st.session_state.answer = ""
            st.session_state.rag_elapsed_seconds = None
            return

        user_id = self._current_user_id()

        if search_method_value == "Semantic Search":
            self.execute_semantic_search(query_text, top_k_value, selected_domain_value)
        elif search_method_value == "RAG Answer":
            cached = get_rag_cache(history_db_path, user_id, query_text, selected_domain_value, min(int(top_k_value), 7)) if user_id else None
            if cached:
                st.session_state.rag_error = ""
                st.session_state.answer = cached["answer"]
                st.session_state.results = cached["results"]
                st.session_state.rag_elapsed_seconds = cached["elapsed_seconds"]
                st.session_state.similar_results = []
                st.session_state.similar_cache = {}
                st.session_state.rag_from_cache = True
                st.session_state.search_performed = True
            else:
                self.execute_rag_search(query_text, top_k_value, selected_domain_value, user_id=user_id)

        # Increment anonymous counter or save to history
        if user_id:
            save_search(
                history_db_path,
                user_id,
                query_text,
                selected_domain_value,
                search_method_value,
                top_k_value,
                len(st.session_state.get("results", [])),
            )
        else:
            st.session_state.anon_search_count = st.session_state.get("anon_search_count", 0) + 1

    def start_search_flow(self) -> None:
        self.perform_search()
        if st.session_state.get("query_input", "") and st.session_state.get("current_page") != "login":
            st.session_state.current_page = "search"

    # ---------- UI state / events ----------

    def get_ui_state(self) -> dict[str, Any]:
        user_id = self._current_user_id()
        history_entries = list_recent_searches(history_db_path, user_id, limit=self.HISTORY_LIMIT) if user_id else []
        history_payload = []
        for entry in history_entries:
            query_text = str(entry.get("query", ""))
            history_payload.append(
                {
                    "id": str(entry.get("id", "")),
                    "query": query_text,
                    "query_preview": query_text if len(query_text) <= 40 else query_text[:40] + "...",
                    "domain": entry.get("domain", "Research"),
                    "method": entry.get("method", "Semantic Search"),
                    "top_k": int(entry.get("top_k", 5)),
                }
            )

        return {
            "current_page": st.session_state.get("current_page", "landing"),
            "query_input": st.session_state.get("query_input", ""),
            "selected_domain": st.session_state.get("selected_domain", "Research"),
            "search_method": st.session_state.get("search_method", "Semantic Search"),
            "top_k": int(st.session_state.get("top_k", 5)),
            "answer": st.session_state.get("answer", ""),
            "rag_elapsed_seconds": st.session_state.get("rag_elapsed_seconds"),
            "results": st.session_state.get("results", []),
            "similar_results": st.session_state.get("similar_results", []),
            "history": history_payload,
        }

    def handle_ui_event(self, event: dict[str, Any]) -> bool:
        action = str(event.get("action", "")).strip().lower()
        if not action:
            return False

        if action == "go_landing":
            self.go_to_landing()
            return True

        if action == "go_search":
            self.go_to_search()
            return True

        if action == "back_to_search":
            self.go_to_search()
            return True

        if action == "reset_search":
            self.initialize_session_state(reset=True)
            return True

        if action == "clear_history":
            user_id = self._current_user_id()
            if user_id:
                clear_history(history_db_path, user_id)
            st.session_state.history_selected_id = ""
            st.session_state.history_applied_id = None
            return True

        if action == "apply_history":
            history_id = str(event.get("history_id", "")).strip()
            if not history_id:
                return False
            user_id = self._current_user_id()
            if not user_id:
                return False
            entries = list_recent_searches(history_db_path, user_id, limit=self.HISTORY_LIMIT)
            for entry in entries:
                if str(entry.get("id", "")) == history_id:
                    st.session_state.query_input = entry["query"]
                    st.session_state.selected_domain = entry["domain"]
                    st.session_state.search_method = entry["method"]
                    st.session_state.top_k = int(entry["top_k"])
                    st.session_state.history_applied_id = history_id
                    st.session_state.current_page = "search"
                    self.perform_search()
                    return True
            return False

        if action == "start_search":
            st.session_state.query_input = str(event.get("query", ""))
            st.session_state.selected_domain = str(event.get("domain", "Research"))
            st.session_state.search_method = str(event.get("method", "Semantic Search"))
            st.session_state.top_k = max(1, min(20, int(event.get("top_k", 5))))
            self.start_search_flow()
            return True

        if action == "find_similar":
            index_raw = event.get("index")
            if index_raw is None or str(index_raw) == "":
                return False
            self.execute_more_like_this(
                int(index_raw),
                str(event.get("domain", st.session_state.get("selected_domain", "Research"))),
            )
            return True

        return False

    # ---------- Sidebar ----------

    def render_history_sidebar(self) -> None:
        st.sidebar.title("SemantikML")

        auth_user = st.session_state.get("auth_user")

        if auth_user:
            # Logged-in user header
            st.sidebar.markdown(
                f'<p class="type-caption" style="margin:0 0 0.4rem;">Signed in as '
                f'<strong>{auth_user["username"]}</strong></p>',
                unsafe_allow_html=True,
            )
            if auth_user.get("is_admin"):
                if st.sidebar.button("Admin Panel", use_container_width=True, type="secondary"):
                    st.session_state.current_page = "admin"
                    st.rerun()

            if st.sidebar.button("Log Out", use_container_width=True, type="secondary"):
                _do_logout(auth_user)

            st.sidebar.divider()
            st.sidebar.subheader("Search History")

            if st.sidebar.button("Clear history", use_container_width=True):
                clear_history(history_db_path, str(auth_user["id"]))
                st.session_state.history_selected_id = ""
                st.session_state.history_applied_id = None

            user_id = str(auth_user["id"])
            entries = list_recent_searches(history_db_path, user_id, limit=self.HISTORY_LIMIT)
            if not entries:
                st.sidebar.caption("No saved searches yet.")
                return

            options = [""] + [str(entry["id"]) for entry in entries]
            entries_by_id = {str(entry["id"]): entry for entry in entries}

            def _label_for(history_id: str) -> str:
                if history_id == "":
                    return "Select a previous search"
                entry = entries_by_id[history_id]
                q = entry["query"]
                return q if len(q) <= 40 else q[:40] + "..."

            selected_id = st.sidebar.selectbox(
                "Previous searches",
                options=options,
                key="history_selected_id",
                format_func=_label_for,
            )

            if selected_id and st.session_state.get("history_applied_id") != selected_id:
                entry = entries_by_id[selected_id]
                st.session_state.query_input = entry["query"]
                st.session_state.selected_domain = entry["domain"]
                st.session_state.search_method = entry["method"]
                st.session_state.top_k = int(entry["top_k"])
                st.session_state.history_applied_id = selected_id
                st.session_state.current_page = "search"
                st.session_state.history_search_pending = True
                st.rerun()

        else:
            # Anonymous user
            anon_count = st.session_state.get("anon_search_count", 0)
            remaining = max(ANON_SEARCH_LIMIT - anon_count, 0)
            st.sidebar.caption(f"Free searches remaining: **{remaining} / {ANON_SEARCH_LIMIT}**")
            st.sidebar.divider()
            if st.sidebar.button("Log In", use_container_width=True):
                st.session_state.current_page = "login"
                st.rerun()
            if st.sidebar.button("Sign Up", use_container_width=True, type="secondary"):
                st.session_state.current_page = "signup"
                st.rerun()


# ---------- Logout helper (module-level so it can be used without an App instance) ----------

def _do_logout(auth_user: dict) -> None:
    log_event(auth_db_path, auth_user["id"], auth_user["username"], "logout", auth_user.get("provider", "local"))
    provider = auth_user.get("provider", "local")

    # Clear persistent session token
    token = st.session_state.get("_session_token")
    if token:
        delete_session(auth_db_path, token)
    cm = st.session_state.get("_cm")
    if cm is not None:
        try:
            cm.delete("semantikml_session")
        except Exception:
            pass

    st.session_state.auth_user = None
    st.session_state["_session_token"] = None
    st.session_state.current_page = "landing"
    if provider == "github":
        st.logout()
    else:
        st.rerun()
