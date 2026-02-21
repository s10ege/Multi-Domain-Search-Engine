# login.py
# Renders the login page: username/password form + GitHub OAuth button.

from pathlib import Path

import streamlit as st

from auth_store import (
    build_session_user,
    get_user_by_email,
    get_user_by_username,
    log_event,
    update_last_login,
    verify_password,
)

LOGO_PATH = Path(__file__).resolve().parent.parent / "images" / "logo.png"


def render_login_page(auth_db_path: str, admin_emails: list[str]) -> None:
    st.logo(image=str(LOGO_PATH), icon_image=str(LOGO_PATH), size="large")

    st.markdown(
        """
        <section class="hero" style="text-align:center; margin-bottom:1.5rem;">
            <div class="hero__eyebrow">SemantikML</div>
            <h1 class="hero__title" style="font-size:clamp(1.5rem,1.2rem+1vw,2rem);">Welcome Back</h1>
            <p class="hero__subtitle">Sign in to unlock your personal search history</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Redirect notice when search limit is hit
    if st.session_state.get("search_limit_reached"):
        st.info("You've used your 3 free searches. Log in to keep searching.")
        st.session_state.search_limit_reached = False

    with st.container(key="login_card"):
        with st.form(key="login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)

    # Sign-up navigation
    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        if st.button(
            "Don't have an account? Sign up",
            use_container_width=True,
            type="secondary",
            key="goto_signup_btn",
        ):
            st.session_state.current_page = "signup"
            st.rerun()

    if submitted:
        _handle_login(auth_db_path, admin_emails, identifier, password)


# ---------- Helpers ----------

def _handle_login(
    auth_db_path: str, admin_emails: list[str], identifier: str, password: str
) -> None:
    if not identifier or not password:
        st.error("Please fill in all fields.")
        return

    user = get_user_by_username(auth_db_path, identifier)
    if user is None:
        user = get_user_by_email(auth_db_path, identifier)

    if user is None or not user.get("password_hash"):
        st.error("Invalid username or password.")
        log_event(auth_db_path, None, identifier, "failed_login", "local")
        return

    if not verify_password(password, user["password_hash"]):
        st.error("Invalid username or password.")
        log_event(auth_db_path, user["id"], user["username"], "failed_login", "local")
        return

    update_last_login(auth_db_path, user["id"])
    log_event(auth_db_path, user["id"], user["username"], "login", "local")
    st.session_state.auth_user = build_session_user(user, admin_emails)
    st.session_state.anon_search_count = 0
    st.session_state.current_page = "landing"
    st.rerun()
