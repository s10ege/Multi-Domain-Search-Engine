# test_streamlit_app.py
# Tests for the Streamlit auth system and UI pages.
#
# Coverage:
#   - auth_store   : password hashing, user CRUD, session building, logging
#   - history_store: list_all_searches (admin query)
#   - Streamlit UI : login and signup pages via AppTest

import sys
import tempfile
from pathlib import Path

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
    link_github,
    list_all_users,
    list_auth_log,
    log_event,
    update_last_login,
    verify_password,
)
from history_store import init_db, list_all_searches, save_search


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# TestPasswordHashing
# ══════════════════════════════════════════════════════════════════════════════

class TestPasswordHashing:
    """Password hashing and verification."""

    def test_correct_password_verifies(self, auth_db):
        """
        verify_password returns True for the correct password.
        Priority: CRITICAL
        """
        create_user(auth_db, "u1", "u1@x.com", "secret123")
        user = get_user_by_email(auth_db, "u1@x.com")
        assert verify_password("secret123", user["password_hash"]) is True

    def test_wrong_password_rejected(self, auth_db):
        """
        verify_password returns False for an incorrect password.
        Priority: CRITICAL
        """
        create_user(auth_db, "u2", "u2@x.com", "secret123")
        user = get_user_by_email(auth_db, "u2@x.com")
        assert verify_password("wrongpassword", user["password_hash"]) is False

    def test_two_identical_passwords_have_different_hashes(self, auth_db):
        """
        Salt ensures identical passwords produce different hashes.
        Priority: IMPORTANT
        """
        create_user(auth_db, "ua", "ua@x.com", "samepass")
        create_user(auth_db, "ub", "ub@x.com", "samepass")
        ua = get_user_by_email(auth_db, "ua@x.com")
        ub = get_user_by_email(auth_db, "ub@x.com")
        assert ua["password_hash"] != ub["password_hash"]

    def test_invalid_hash_format_returns_false(self):
        """
        verify_password returns False for a malformed stored hash.
        Priority: IMPORTANT
        """
        assert verify_password("password", "notahashstring") is False

    def test_empty_password_can_be_hashed_and_verified(self, auth_db):
        """
        Empty string is a valid (if weak) password value.
        Priority: NICE-TO-HAVE
        """
        create_user(auth_db, "ue", "ue@x.com", "")
        user = get_user_by_email(auth_db, "ue@x.com")
        assert verify_password("", user["password_hash"]) is True
        assert verify_password("notempty", user["password_hash"]) is False


# ══════════════════════════════════════════════════════════════════════════════
# TestInitAuthDB
# ══════════════════════════════════════════════════════════════════════════════

class TestInitAuthDB:
    """DB initialisation."""

    def test_creates_users_table(self, tmp_path):
        """
        init_auth_db creates the users table.
        Priority: CRITICAL
        """
        import sqlite3
        db = str(tmp_path / "a.db")
        init_auth_db(db)
        with sqlite3.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "users" in tables

    def test_creates_auth_log_table(self, tmp_path):
        """
        init_auth_db creates the auth_log table.
        Priority: CRITICAL
        """
        import sqlite3
        db = str(tmp_path / "b.db")
        init_auth_db(db)
        with sqlite3.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "auth_log" in tables

    def test_idempotent_when_called_twice(self, tmp_path):
        """
        Calling init_auth_db twice does not raise an error.
        Priority: IMPORTANT
        """
        db = str(tmp_path / "c.db")
        init_auth_db(db)
        init_auth_db(db)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# TestCreateUser
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateUser:
    """Local user creation."""

    def test_returns_integer_id(self, auth_db):
        """
        create_user returns a positive integer id on success.
        Priority: CRITICAL
        """
        uid = create_user(auth_db, "alice", "alice@x.com", "pass1234")
        assert isinstance(uid, int) and uid > 0

    def test_duplicate_username_returns_none(self, auth_db):
        """
        create_user returns None when username already exists.
        Priority: CRITICAL
        """
        create_user(auth_db, "bob", "bob@x.com", "pass1234")
        result = create_user(auth_db, "bob", "other@x.com", "pass1234")
        assert result is None

    def test_duplicate_email_returns_none(self, auth_db):
        """
        create_user returns None when email already exists.
        Priority: CRITICAL
        """
        create_user(auth_db, "carol", "carol@x.com", "pass1234")
        result = create_user(auth_db, "carol2", "carol@x.com", "pass1234")
        assert result is None

    def test_email_stored_lowercase(self, auth_db):
        """
        Email is stored in lowercase regardless of input case.
        Priority: IMPORTANT
        """
        create_user(auth_db, "dave", "DAVE@X.COM", "pass1234")
        user = get_user_by_email(auth_db, "dave@x.com")
        assert user is not None
        assert user["email"] == "dave@x.com"

    def test_provider_set_to_local(self, auth_db):
        """
        Local users have provider='local'.
        Priority: IMPORTANT
        """
        create_user(auth_db, "eve", "eve@x.com", "pass1234")
        user = get_user_by_email(auth_db, "eve@x.com")
        assert user["provider"] == "local"


# ══════════════════════════════════════════════════════════════════════════════
# TestGetUser
# ══════════════════════════════════════════════════════════════════════════════

class TestGetUser:
    """User lookup functions."""

    def test_get_by_email_found(self, auth_db, user_in_db):
        """
        get_user_by_email returns the user dict when found.
        Priority: CRITICAL
        """
        user = get_user_by_email(auth_db, "test@example.com")
        assert user is not None
        assert user["username"] == "testuser"

    def test_get_by_email_not_found(self, auth_db):
        """
        get_user_by_email returns None for unknown email.
        Priority: CRITICAL
        """
        assert get_user_by_email(auth_db, "nobody@x.com") is None

    def test_get_by_email_case_insensitive(self, auth_db, user_in_db):
        """
        get_user_by_email is case-insensitive.
        Priority: IMPORTANT
        """
        user = get_user_by_email(auth_db, "TEST@EXAMPLE.COM")
        assert user is not None

    def test_get_by_username_found(self, auth_db, user_in_db):
        """
        get_user_by_username returns the user dict when found.
        Priority: CRITICAL
        """
        user = get_user_by_username(auth_db, "testuser")
        assert user is not None
        assert user["email"] == "test@example.com"

    def test_get_by_username_not_found(self, auth_db):
        """
        get_user_by_username returns None for unknown username.
        Priority: CRITICAL
        """
        assert get_user_by_username(auth_db, "ghost") is None

    def test_get_by_username_case_insensitive(self, auth_db, user_in_db):
        """
        get_user_by_username is case-insensitive.
        Priority: IMPORTANT
        """
        user = get_user_by_username(auth_db, "TESTUSER")
        assert user is not None


# ══════════════════════════════════════════════════════════════════════════════
# TestGitHubUser
# ══════════════════════════════════════════════════════════════════════════════

class TestGitHubUser:
    """GitHub-provider user management."""

    def test_create_github_user_success(self, auth_db):
        """
        create_github_user returns a new user id.
        Priority: CRITICAL
        """
        uid = create_github_user(auth_db, "gh@x.com", "GH User", "gh_sub_001")
        assert isinstance(uid, int) and uid > 0

    def test_github_user_has_correct_provider(self, auth_db):
        """
        GitHub users have provider='github'.
        Priority: CRITICAL
        """
        create_github_user(auth_db, "gh2@x.com", "GH2", "gh_sub_002")
        user = get_user_by_email(auth_db, "gh2@x.com")
        assert user["provider"] == "github"
        assert user["password_hash"] is None

    def test_github_user_username_conflict_resolved(self, auth_db):
        """
        create_github_user auto-suffixes username if it already exists.
        Priority: IMPORTANT
        """
        create_user(auth_db, "ghuser", "local@x.com", "pass")
        uid = create_github_user(auth_db, "gh3@x.com", "ghuser", "gh_sub_003")
        user = get_user_by_email(auth_db, "gh3@x.com")
        assert uid is not None
        assert user["username"] != "ghuser"         # was suffixed
        assert user["username"].startswith("ghuser")

    def test_github_duplicate_email_returns_none(self, auth_db):
        """
        create_github_user returns None when email already exists.
        Priority: CRITICAL
        """
        create_github_user(auth_db, "dup@x.com", "User", "sub_001")
        result = create_github_user(auth_db, "dup@x.com", "User2", "sub_002")
        assert result is None

    def test_link_github_updates_github_id(self, auth_db, user_in_db):
        """
        link_github sets the github_id on an existing local user.
        Priority: CRITICAL
        """
        link_github(auth_db, user_in_db["id"], "gh_sub_999")
        user = get_user_by_email(auth_db, "test@example.com")
        assert user["github_id"] == "gh_sub_999"


# ══════════════════════════════════════════════════════════════════════════════
# TestUpdateLastLogin
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateLastLogin:

    def test_sets_last_login_timestamp(self, auth_db, user_in_db):
        """
        update_last_login stores a non-null timestamp.
        Priority: IMPORTANT
        """
        update_last_login(auth_db, user_in_db["id"])
        user = get_user_by_email(auth_db, "test@example.com")
        assert user["last_login"] is not None

    def test_last_login_initially_null(self, auth_db, user_in_db):
        """
        last_login is NULL before the first login update.
        Priority: IMPORTANT
        """
        user = get_user_by_email(auth_db, "test@example.com")
        assert user["last_login"] is None


# ══════════════════════════════════════════════════════════════════════════════
# TestLogEvent
# ══════════════════════════════════════════════════════════════════════════════

class TestLogEvent:

    def test_log_event_inserts_row(self, auth_db, user_in_db):
        """
        log_event inserts one row into auth_log.
        Priority: CRITICAL
        """
        log_event(auth_db, user_in_db["id"], "testuser", "login", "local")
        entries = list_auth_log(auth_db)
        assert len(entries) == 1
        assert entries[0]["event"] == "login"
        assert entries[0]["provider"] == "local"

    def test_log_event_allows_null_user_id(self, auth_db):
        """
        log_event accepts None user_id for anonymous events (e.g. failed login).
        Priority: IMPORTANT
        """
        log_event(auth_db, None, "unknown", "failed_login", "local")
        entries = list_auth_log(auth_db)
        assert entries[0]["user_id"] is None

    def test_log_event_multiple_entries(self, auth_db, user_in_db):
        """
        Multiple events are all stored in order.
        Priority: IMPORTANT
        """
        log_event(auth_db, user_in_db["id"], "testuser", "login", "local")
        log_event(auth_db, user_in_db["id"], "testuser", "logout", "local")
        entries = list_auth_log(auth_db)
        assert len(entries) == 2
        events = {e["event"] for e in entries}
        assert events == {"login", "logout"}


# ══════════════════════════════════════════════════════════════════════════════
# TestBuildSessionUser
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSessionUser:

    def _make_user(self, email="user@x.com", provider="local"):
        return {"id": 1, "username": "user", "email": email, "provider": provider}

    def test_builds_correct_keys(self):
        """
        build_session_user returns dict with expected keys.
        Priority: CRITICAL
        """
        result = build_session_user(self._make_user(), admin_emails=[])
        assert set(result.keys()) == {"id", "username", "email", "provider", "is_admin"}

    def test_non_admin_email(self):
        """
        is_admin is False when email is not in admin_emails.
        Priority: CRITICAL
        """
        result = build_session_user(self._make_user(), admin_emails=["admin@x.com"])
        assert result["is_admin"] is False

    def test_admin_email_match(self):
        """
        is_admin is True when email matches admin_emails.
        Priority: CRITICAL
        """
        result = build_session_user(
            self._make_user(email="admin@x.com"), admin_emails=["admin@x.com"]
        )
        assert result["is_admin"] is True

    def test_admin_check_is_case_insensitive(self):
        """
        Admin email comparison is case-insensitive.
        Priority: IMPORTANT
        """
        result = build_session_user(
            self._make_user(email="Admin@X.COM"), admin_emails=["admin@x.com"]
        )
        assert result["is_admin"] is True

    def test_empty_admin_list(self):
        """
        is_admin is always False when admin_emails is empty.
        Priority: IMPORTANT
        """
        result = build_session_user(self._make_user(), admin_emails=[])
        assert result["is_admin"] is False


# ══════════════════════════════════════════════════════════════════════════════
# TestListAllUsers
# ══════════════════════════════════════════════════════════════════════════════

class TestListAllUsers:

    def test_empty_when_no_users(self, auth_db):
        """
        list_all_users returns empty list when DB is fresh.
        Priority: IMPORTANT
        """
        assert list_all_users(auth_db) == []

    def test_returns_all_users(self, auth_db):
        """
        list_all_users returns one entry per registered user.
        Priority: CRITICAL
        """
        create_user(auth_db, "u1", "u1@x.com", "pass")
        create_user(auth_db, "u2", "u2@x.com", "pass")
        users = list_all_users(auth_db)
        assert len(users) == 2

    def test_does_not_return_password_hash(self, auth_db):
        """
        list_all_users does not expose password_hash.
        Priority: CRITICAL
        """
        create_user(auth_db, "u3", "u3@x.com", "pass")
        users = list_all_users(auth_db)
        assert "password_hash" not in users[0]


# ══════════════════════════════════════════════════════════════════════════════
# TestListAllSearches
# ══════════════════════════════════════════════════════════════════════════════

class TestListAllSearches:

    def test_empty_when_no_searches(self, history_db):
        """
        list_all_searches returns empty list on fresh DB.
        Priority: IMPORTANT
        """
        assert list_all_searches(history_db) == []

    def test_returns_searches_from_all_users(self, history_db):
        """
        list_all_searches includes searches from multiple user IDs.
        Priority: CRITICAL
        """
        save_search(history_db, "1", "neural networks", "Research", "Semantic Search", 5, 3)
        save_search(history_db, "2", "transformers",    "Blog",     "RAG Answer",       3, 2)
        results = list_all_searches(history_db)
        user_ids = {r["user_id"] for r in results}
        assert "1" in user_ids
        assert "2" in user_ids

    def test_respects_limit(self, history_db):
        """
        list_all_searches respects the limit parameter.
        Priority: IMPORTANT
        """
        for i in range(10):
            save_search(history_db, f"{i}", f"query {i}", "Research", "Semantic Search", 5, i)
        results = list_all_searches(history_db, limit=4)
        assert len(results) <= 4

    def test_ordered_by_most_recent(self, history_db):
        """
        list_all_searches returns newest entries first.
        Priority: IMPORTANT
        """
        save_search(history_db, "1", "first query",  "Research", "Semantic Search", 5, 1)
        save_search(history_db, "1", "second query", "Research", "Semantic Search", 5, 2)
        results = list_all_searches(history_db)
        assert results[0]["query"] == "second query"


# ══════════════════════════════════════════════════════════════════════════════
# TestLoginPageUI  (Streamlit AppTest)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoginPageUI:
    """Login page rendering and form behaviour via Streamlit AppTest."""

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

    def test_renders_without_exception(self, auth_db):
        """
        Login page runs without raising an exception.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        assert not at.exception

    def test_shows_identifier_and_password_inputs(self, auth_db):
        """
        Login page renders username/email and password text inputs.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        assert len(at.text_input) >= 2

    def test_empty_submission_shows_error(self, auth_db):
        """
        Submitting blank credentials shows an error message.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        # Leave fields blank and submit
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_wrong_credentials_show_error(self, auth_db, user_in_db):
        """
        Submitting wrong password shows an invalid-credentials error.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("testuser")
        at.text_input[1].set_value("wrongpassword")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_unknown_user_shows_error(self, auth_db):
        """
        Submitting credentials for a non-existent user shows an error.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("nobody")
        at.text_input[1].set_value("somepassword")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_correct_credentials_set_session_user(self, auth_db, user_in_db):
        """
        Successful login stores auth_user in session state.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("testuser")
        at.text_input[1].set_value("Passw0rd!")
        at.button[0].click().run()
        assert "auth_user" in at.session_state
        assert at.session_state["auth_user"]["username"] == "testuser"

    def test_search_limit_notice_shown_when_flag_set(self, auth_db):
        """
        Info message appears when search_limit_reached is True.
        Priority: IMPORTANT
        """
        from streamlit.testing.v1 import AppTest

        script = self._script(auth_db)
        at = AppTest.from_string(script)
        at.run()
        at.session_state["search_limit_reached"] = True
        at.run()
        assert any("3 free searches" in m.value for m in at.info)

    def test_signup_button_navigates_to_signup(self, auth_db):
        """
        Clicking 'Sign up' sets current_page to 'signup'.
        Priority: IMPORTANT
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        # Last button is "Don't have an account? Sign up"
        at.button[-1].click().run()
        assert at.session_state["current_page"] == "signup"


# ══════════════════════════════════════════════════════════════════════════════
# TestSignupPageUI  (Streamlit AppTest)
# ══════════════════════════════════════════════════════════════════════════════

class TestSignupPageUI:
    """Signup page rendering and form validation via Streamlit AppTest."""

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

    def test_renders_without_exception(self, auth_db):
        """
        Signup page runs without raising an exception.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        assert not at.exception

    def test_shows_four_input_fields(self, auth_db):
        """
        Signup page renders username, email, password, and confirm-password inputs.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        assert len(at.text_input) == 4

    def test_short_username_shows_error(self, auth_db):
        """
        Username shorter than 3 characters triggers a validation error.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("ab")          # username too short
        at.text_input[1].set_value("x@x.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_invalid_email_shows_error(self, auth_db):
        """
        Email without '@' triggers a validation error.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("validuser")
        at.text_input[1].set_value("notanemail")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_short_password_shows_error(self, auth_db):
        """
        Password shorter than 8 characters triggers a validation error.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("validuser")
        at.text_input[1].set_value("valid@x.com")
        at.text_input[2].set_value("short")
        at.text_input[3].set_value("short")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_mismatched_passwords_show_error(self, auth_db):
        """
        Non-matching password and confirm-password triggers a validation error.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("validuser")
        at.text_input[1].set_value("valid@x.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("different123")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_valid_signup_creates_user(self, auth_db):
        """
        Valid signup creates the user in the database.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("newuser")
        at.text_input[1].set_value("new@x.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()
        user = get_user_by_email(auth_db, "new@x.com")
        assert user is not None
        assert user["username"] == "newuser"

    def test_valid_signup_redirects_to_login(self, auth_db):
        """
        Successful signup sets current_page to 'login'.
        Priority: CRITICAL
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("anotheruser")
        at.text_input[1].set_value("another@x.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()
        assert at.session_state["current_page"] == "login"

    def test_duplicate_username_shows_error(self, auth_db, user_in_db):
        """
        Signing up with an existing username shows an error.
        Priority: IMPORTANT
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.text_input[0].set_value("testuser")      # already exists
        at.text_input[1].set_value("fresh@x.com")
        at.text_input[2].set_value("password123")
        at.text_input[3].set_value("password123")
        at.button[0].click().run()
        assert len(at.error) > 0

    def test_login_button_navigates_to_login(self, auth_db):
        """
        Clicking 'Log in' sets current_page to 'login'.
        Priority: IMPORTANT
        """
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_string(self._script(auth_db))
        at.run()
        at.button[-1].click().run()
        assert at.session_state["current_page"] == "login"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
