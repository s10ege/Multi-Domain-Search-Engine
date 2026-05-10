import base64
import streamlit as st
from html import escape

from app_core import App
from paths import LOGO_PATH

def _home_logo_btn_html() -> str:
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return (
        '<a href="?go=landing" target="_self" class="home-logo-btn">'
        f'<img src="data:image/png;base64,{logo_b64}" alt="" height="26" style="display:block;">'
        'SemantikML'
        '</a>'
    )


def render_search_page(app_core: App) -> None:
	app_core.render_history_sidebar()

	if st.session_state.get("history_search_pending"):
		st.session_state.history_search_pending = False
		app_core.perform_search()

	st.markdown(_home_logo_btn_html(), unsafe_allow_html=True)
	st.markdown('<h1 class="page-title type-h1">Search</h1>', unsafe_allow_html=True)
	st.markdown(
		'<p class="type-short">Find relevant papers and explore similar content from each result.</p>',
		unsafe_allow_html=True,
	)

	with st.container(key="search_form"):
		with st.form(key="search_page_form_secondary"):
			st.text_area("Search query", key="query_input", height=80)
			with st.container(key="search_form_actions"):
				start_col, spacer_col, reset_col = st.columns([1.5, 3.6, 1.5])
				with start_col:
					st.form_submit_button(
						"Start Search",
						on_click=app_core.perform_search,
						use_container_width=True,
					)
				with spacer_col:
					st.write("")
				with reset_col:
					st.form_submit_button(
						"Reset Search",
						on_click=app_core.initialize_session_state,
						kwargs={"reset": True},
						type="secondary",
						use_container_width=True,
				)


 
	with st.container(key="parameters"):
		with st.expander("Search Parameters", expanded=False):
			st.radio("Choose a domain: ", ["Research", "Blog", "Documentation"], key="selected_domain")
			st.radio("Select search method:", ["Semantic Search", "RAG Answer"], key="search_method")
			st.slider("Select Top-K results:", min_value=1, max_value=20, key="top_k")
			st.slider(
				"Similarity threshold (Find Similar):",
				min_value=0.0,
				max_value=1.0,
				step=0.05,
				value=0.70,
				key="similarity_threshold",
				help="Only 'Find Similar' results above this score are shown. Lower = more results, higher = stricter.",
			)

	if st.session_state.answer:
		rag_elapsed_seconds = st.session_state.get("rag_elapsed_seconds")
		rag_from_cache = st.session_state.get("rag_from_cache", False)
		if rag_from_cache:
			st.markdown(
				'<p class="type-caption">&#9889; Loaded from cache &mdash; answer is instant</p>',
				unsafe_allow_html=True,
			)
		elif rag_elapsed_seconds is not None:
			total_seconds = max(int(round(float(rag_elapsed_seconds))), 0)
			minutes, seconds = divmod(total_seconds, 60)
			st.markdown(
				f'<p class="type-caption">Thought time: {minutes}m {seconds}s</p>',
				unsafe_allow_html=True,
			)

		st.markdown('<h2 class="section-title type-h2">Answer</h2>', unsafe_allow_html=True)
		st.markdown(
			f'<div class="answer-box"><p class="type-body">{escape(str(st.session_state.answer))}</p></div>',
			unsafe_allow_html=True,
		)
	


	if st.session_state.results:
		st.markdown('<h2 class="section-title type-h2">Related Papers</h2>', unsafe_allow_html=True)
		for i, result in enumerate(st.session_state.results):
			title = escape(str(result.get("title", "Untitled")))
			domain = escape(str(result.get("domain", "Unknown")))
			source = escape(str(result.get("source", "Unknown")))
			score = float(result.get("score", 0.0))
			raw_text = str(result.get("text", ""))
			preview = raw_text[:1200] + "..." if len(raw_text) > 1200 else raw_text
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

			st.button(
				"Find Similar",
				key=f"similar_{result['index']}",
				on_click=app_core.execute_more_like_this,
				args=(result["index"], st.session_state.get("selected_domain", "Research")),
			)
	elif st.session_state.get("search_performed") and st.session_state.get("query_input"):
		st.info("No results found for your query. Try different keywords or a broader domain.")
