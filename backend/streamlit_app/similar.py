import base64
import streamlit as st
from html import escape

from paths import LOGO_PATH


def _home_logo_btn_html() -> str:
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return (
        '<a href="?go=landing" target="_self" class="home-logo-btn">'
        f'<img src="data:image/png;base64,{logo_b64}" alt="" height="26" style="display:block;">'
        'SemantikML'
        '</a>'
    )


def render_similar_page(on_back=None) -> None:
    st.markdown(_home_logo_btn_html(), unsafe_allow_html=True)
    st.markdown('<h1 class="page-title type-h1">Similar Documents</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="type-short">Documents ranked by semantic similarity above the quality threshold.</p>',
        unsafe_allow_html=True,
    )

    if on_back is not None:
        st.button("Back to Search", on_click=on_back)

    results = st.session_state.get("similar_results", [])
    threshold = st.session_state.get("similar_threshold_used", 0.70)
    if not results:
        st.markdown(
            f'<p class="type-short">No similar documents found above the {threshold:.2f} similarity threshold after excluding the current search results.</p>',
            unsafe_allow_html=True,
        )
        return

    total = st.session_state.get("similar_candidates_total", 0)
    if total > len(results):
        st.caption(
            f"Showing {len(results)} of {total} candidates — "
            f"only results above the {threshold:.2f} similarity threshold are displayed."
        )

    for i, result in enumerate(results):
        title = escape(str(result.get("title", "Untitled")))
        domain = escape(str(result.get("domain", "Unknown")))
        source = escape(str(result.get("source", "Unknown")))
        score = float(result.get("score", 0.0))
        raw_text = str(result.get("text", ""))
        preview = raw_text[:850] + "..." if len(raw_text) > 850 else raw_text
        preview = escape(preview)

        st.markdown(
            f"""
            <article class="result-card">
                <p class="result-card__meta type-meta">Result {i + 1}</p>
                <h3 class="result-card__title type-h3">{title}</h3>
                <div class="result-card__chips">
                    <span class="chip type-caption">Domain: {domain}</span>
                    <span class="chip type-caption">Source: {source}</span>
                    <span class="chip type-caption">Score: {score:.4f}</span>
                </div>
                <p class="result-card__snippet type-short">{preview}</p>
            </article>
            """,
            unsafe_allow_html=True,
        )