import streamlit as st
from html import escape


def render_similar_page(on_back=None) -> None:
    st.markdown('<h1 class="page-title type-h1">Similar Documents</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="type-short">Documents ranked by semantic similarity above the quality threshold.</p>',
        unsafe_allow_html=True,
    )

    if on_back is not None:
        st.button("Back to Search", on_click=on_back)

    results = st.session_state.get("similar_results", [])
    if not results:
        st.markdown(
            '<p class="type-short">No similar documents above 0.70 similarity after excluding the current search results.</p>',
            unsafe_allow_html=True,
        )
        return

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