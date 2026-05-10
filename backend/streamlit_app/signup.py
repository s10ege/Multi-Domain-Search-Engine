# signup.py
# Renders the self-registration page.

import base64
import re
import streamlit as st

from auth_store import create_user, get_user_by_email, get_user_by_username, log_event
from paths import LOGO_PATH

_MIN_USERNAME_LEN = 3
_MIN_PASSWORD_LEN = 8
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def _home_logo_btn_html() -> str:
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return (
        '<a href="?go=landing" target="_self" class="home-logo-btn">'
        f'<img src="data:image/png;base64,{logo_b64}" alt="" height="26" style="display:block;">'
        'SemantikML'
        '</a>'
    )


def render_signup_page(auth_db_path: str) -> None:
    st.markdown(_home_logo_btn_html(), unsafe_allow_html=True)

    st.markdown(
        """
        <section class="hero" style="text-align:center; margin-bottom:1.5rem;">
            <div class="hero__eyebrow">SemantikML</div>
            <h1 class="hero__title" style="font-size:clamp(1.5rem,1.2rem+1vw,2rem);">Create Account</h1>
            <p class="hero__subtitle">Join to unlock your personal search history</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="signup_card"):
        with st.form(key="signup_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        if st.button(
            "Already have an account? Log in",
            use_container_width=True,
            type="secondary",
            key="goto_login_btn",
        ):
            st.session_state.current_page = "login"
            st.rerun()

    if submitted:
        _handle_signup(auth_db_path, username, email, password, confirm)


# ---------- Helpers ----------

def _handle_signup(
    auth_db_path: str, username: str, email: str, password: str, confirm: str
) -> None:
    errors = _validate(username, email, password, confirm)
    if errors:
        for msg in errors:
            st.error(msg)
        return

    user_id = create_user(auth_db_path, username.strip(), email.strip(), password)
    if user_id is None:
        # Identify which field conflicts for a helpful message
        if get_user_by_username(auth_db_path, username.strip()):
            st.error("That username is already taken.")
        elif get_user_by_email(auth_db_path, email.strip()):
            st.error("That email is already registered. Try logging in.")
        else:
            st.error("Registration failed. Please try again.")
        return

    log_event(auth_db_path, user_id, username.strip(), "register", "local")
    st.success("Account created! Redirecting to login...")
    st.session_state.current_page = "login"
    st.rerun()


def _validate(username: str, email: str, password: str, confirm: str) -> list[str]:
    errors = []
    if not username or len(username.strip()) < _MIN_USERNAME_LEN:
        errors.append(f"Username must be at least {_MIN_USERNAME_LEN} characters.")
    if not email or not _EMAIL_RE.match(email.strip()):
        errors.append("Please enter a valid email address.")
    if not password or len(password) < _MIN_PASSWORD_LEN:
        errors.append(f"Password must be at least {_MIN_PASSWORD_LEN} characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors
