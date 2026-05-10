from pathlib import Path

import streamlit as st
from extra_streamlit_components import CookieManager
from app_core import App, auth_db_path, history_db_path
from search import render_search_page
from similar import render_similar_page
from login import render_login_page
from signup import render_signup_page
from admin import render_admin_page
from auth_store import (
    build_session_user,
    create_github_user,
    get_session_user,
    get_user_by_email,
    link_github,
    log_event,
    update_last_login,
)
from paths import LOGO_PATH

# NOTE: GitHub OAuth is not supported by st.login() because GitHub OAuth Apps
# use OAuth 2.0 only — they do not implement the OIDC protocol that Streamlit
# requires. The GitHub callback handler below is kept for future use if a
# compatible OIDC provider (e.g. Google) is configured under [auth.<provider>].

st.set_page_config(
    page_title="SemantikML",
    page_icon=LOGO_PATH,
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# To contact the developer: 2166507@brunel.ac.uk",
    },
)

# ---------- Helpers ----------

def _get_admin_emails() -> list[str]:
    try:
        return list(st.secrets["admin"]["emails"])
    except Exception:
        return []


def _css_mtime_ns() -> int:
    css_path = Path(__file__).with_name("styles.css")
    if not css_path.exists():
        return -1
    return css_path.stat().st_mtime_ns


@st.fragment(run_every="1s")
def watch_css_changes() -> None:
    current_mtime = _css_mtime_ns()
    previous_mtime = st.session_state.get("_css_mtime_ns")
    if previous_mtime is None:
        st.session_state["_css_mtime_ns"] = current_mtime
        return
    if current_mtime != previous_mtime:
        st.session_state["_css_mtime_ns"] = current_mtime
        st.rerun()


def load_css() -> None:
    css_path = Path(__file__).with_name("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ---------- Cookie-based session restore ----------

def _restore_auth_from_cookie(cm: CookieManager, admin_emails: list[str]) -> None:
    """If a valid session cookie exists, restore auth_user from the DB."""
    try:
        token = cm.get("semantikml_session")
    except Exception:
        return
    if not token:
        return
    user = get_session_user(auth_db_path, token)
    if user:
        st.session_state.auth_user = build_session_user(user, admin_emails)
        st.session_state["_session_token"] = token


# ---------- GitHub OAuth callback ----------

def _handle_github_callback(admin_emails: list[str]) -> None:
    """
    Runs on every rerun. If Streamlit has an active OAuth session but our
    app session doesn't know about it yet, look up / create the user in our DB.
    """
    try:
        logged_in = st.user.is_logged_in
    except Exception:
        return  # OAuth not configured — skip silently

    if not logged_in or st.session_state.get("auth_user"):
        return  # Nothing to do

    try:
        email = st.user.email or ""
        display_name = getattr(st.user, "name", "") or ""
        github_sub = str(getattr(st.user, "sub", "") or "")
    except Exception:
        return

    if not email:
        return

    user = get_user_by_email(auth_db_path, email)

    if user is None:
        # New GitHub user — create account
        create_github_user(auth_db_path, email, display_name, github_sub)
        user = get_user_by_email(auth_db_path, email)
        if user:
            log_event(auth_db_path, user["id"], user["username"], "register", "github")
    elif user.get("github_id") is None:
        # Existing local account with matching email — link GitHub
        link_github(auth_db_path, user["id"], github_sub)
        log_event(auth_db_path, user["id"], user["username"], "github_link", "github")

    if user:
        update_last_login(auth_db_path, user["id"])
        log_event(auth_db_path, user["id"], user["username"], "login", "github")
        st.session_state.auth_user = build_session_user(user, admin_emails)
        if st.session_state.get("current_page") in ("login", "signup"):
            st.session_state.current_page = "landing"


# ---------- Landing page ----------

def render_landing_page(app: App) -> None:
    app.render_history_sidebar()

    st.markdown(
        """
            <section class="hero">
                <div class="hero__eyebrow">SemantikML</div>
                <h1 class="hero__title">Multi Domain ML Search Engine</h1>
                <p class="hero__subtitle">A txtAI based search engine.</p>
                <p class="hero__subtitle">Search ML papers like a pro!</p>
            </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="search_form"):
        with st.form(key="initial_search_page_form"):
            st.radio("Choose a domain: ", ["Research", "Blog", "Documentation"], key="selected_domain")
            st.radio("Select search method:", ["Semantic Search", "RAG Answer"], key="search_method")
            st.slider("Select Top-K results:", min_value=1, max_value=20, key="top_k")
            st.text_area("Enter your search query:", key="query_input", height=80)
            start_col, spacer_col, reset_col = st.columns([1.5, 3.6, 1.5])
            with start_col:
                st.form_submit_button(
                    "Start Search",
                    on_click=app.start_search_flow,
                    use_container_width=True,
                )
            with spacer_col:
                st.write("")
            with reset_col:
                st.form_submit_button(
                    "Reset Search",
                    on_click=app.initialize_session_state,
                    kwargs={"reset": True},
                    type="secondary",
                    use_container_width=True,
                )


# ---------- Main ----------

def main() -> None:
    watch_css_changes()
    load_css()

    admin_emails = _get_admin_emails()

    # Create the cookie manager once per session and share it via session state
    cm = CookieManager(key="semantikml_cm")
    st.session_state["_cm"] = cm

    # Restore auth from persistent cookie before anything else
    if not st.session_state.get("auth_user"):
        _restore_auth_from_cookie(cm, admin_emails)

    _handle_github_callback(admin_emails)

    app = App()
    app.initialize_session_state()

    if st.query_params.get("go") == "landing":
        app.go_to_landing()
        st.query_params.clear()

    page = st.session_state.get("current_page", "landing")
    auth_user = st.session_state.get("auth_user")

    if page == "login":
        render_login_page(auth_db_path, admin_emails)

    elif page == "signup":
        render_signup_page(auth_db_path)

    elif page == "admin":
        if auth_user and auth_user.get("is_admin"):
            render_admin_page(auth_db_path, history_db_path, on_back=app.go_to_landing)
        else:
            st.error("Access denied.")
            st.session_state.current_page = "landing"
            st.rerun()

    elif page == "search":
        render_search_page(app)

    elif page == "similar":
        render_similar_page(on_back=app.go_to_search)

    else:
        render_landing_page(app)


if __name__ == "__main__":
    main()
