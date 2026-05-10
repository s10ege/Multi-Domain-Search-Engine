# admin.py
# Admin panel: users, search activity, auth log.

import base64
import pandas as pd
import streamlit as st

from auth_store import list_all_users, list_auth_log
from history_store import list_all_searches
from paths import LOGO_PATH


def _home_logo_btn_html() -> str:
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return (
        '<a href="?go=landing" target="_self" class="home-logo-btn">'
        f'<img src="data:image/png;base64,{logo_b64}" alt="" height="26" style="display:block;">'
        'SemantikML'
        '</a>'
    )


def render_admin_page(auth_db_path: str, search_history_db_path: str, on_back=None) -> None:
    st.markdown(_home_logo_btn_html(), unsafe_allow_html=True)

    st.markdown('<h1 class="page-title type-h1">Admin Panel</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="type-short">System-wide activity overview.</p>',
        unsafe_allow_html=True,
    )

    if on_back is not None:
        st.button("Back", on_click=on_back, type="secondary")

    tab_users, tab_searches, tab_log = st.tabs(["Users", "Search Activity", "Auth Log"])

    with tab_users:
        users = list_all_users(auth_db_path)
        if users:
            df = pd.DataFrame(users)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(users)} registered user(s).")
        else:
            st.info("No users registered yet.")

    with tab_searches:
        searches = list_all_searches(search_history_db_path)
        if searches:
            df = pd.DataFrame(searches)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(searches)} search record(s) (most recent 500).")
        else:
            st.info("No searches recorded yet.")

    with tab_log:
        log = list_auth_log(auth_db_path)
        if log:
            df = pd.DataFrame(log)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(log)} auth event(s) (most recent 200).")
        else:
            st.info("No auth events yet.")
