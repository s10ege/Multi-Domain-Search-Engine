"""Capture three poster-ready SemantikML screenshots via Playwright.

Run from the project root with the venv activated:

    .venv/Scripts/python.exe scripts/poster_screenshots.py

The script launches Streamlit on localhost:8501, drives a headed Chromium
browser through Playwright (so you can watch it work), submits one anonymous
semantic search to land on the search page, and then DOM-mutates the page
between shots to stage three distinct views: Semantic Search results, RAG
Answer, and More Like This.

Outputs (high-res, 2x DPR):
    scripts/output/poster_semantic_search.png
    scripts/output/poster_rag_answer.png
    scripts/output/poster_more_like_this.png

Inspection copies resized to 1100px wide are also written next to them as
_inspect_*.png so the legibility check can be eyeballed at the print size.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterator

from PIL import Image
from playwright.sync_api import Page, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "backend" / "streamlit_app" / "main.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
VIEWPORT = {"width": 1600, "height": 1200}
DEVICE_SCALE_FACTOR = 2
INSPECT_WIDTH = 1100  # ≈ 95 mm at 300 dpi

# ---------------------------------------------------------------------------
# Streamlit subprocess management
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    """Stronger readiness check than just a TCP probe — Streamlit accepts the
    socket well before its main bundle is being served."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


def start_streamlit() -> subprocess.Popen:
    """Launch Streamlit and block until the port is accepting connections."""
    if _port_open("localhost", STREAMLIT_PORT):
        print(f"[poster] Streamlit already running on :{STREAMLIT_PORT} — reusing it.")
        return None  # type: ignore[return-value]

    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.runOnSave",
        "false",
    ]
    print(f"[poster] Launching Streamlit: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

    deadline = time.time() + 120
    while time.time() < deadline:
        if _http_ok(STREAMLIT_URL):
            print("[poster] Streamlit is up.")
            return proc
        time.sleep(0.7)
    proc.terminate()
    raise RuntimeError("Streamlit failed to start within 120 seconds.")


def stop_streamlit(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# JS / CSS injection
# ---------------------------------------------------------------------------

# Hides Streamlit chrome and bumps font sizes inside result cards so the cards
# remain legible after the screenshot is downscaled to ~1100px wide for print.
POSTER_CSS = """
header, footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stAppDeployButton"],
.stDeployButton,
#MainMenu,
[data-testid="stExpander"],
[data-testid="stForm"] [data-testid="stRadio"],
[data-testid="stForm"] [data-testid="stSlider"],
.st-key-parameters,
[data-testid="stSpinner"],
[data-testid="stProgress"] {
    display: none !important;
}

/* Ditch the small "Find Similar" Streamlit buttons that follow each card. */
[data-testid="stMain"] [data-testid="stButton"]:not(.poster-keep) {
    display: none !important;
}

/* Tighten the page margins so cropped screenshots fill the frame. */
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.2rem !important;
    padding-bottom: 1.2rem !important;
    max-width: 1400px !important;
}

/* Keep the dark navy palette but enlarge typography for print legibility. */
.page-title { font-size: 40px !important; line-height: 1.1 !important; }
.section-title { font-size: 32px !important; line-height: 1.15 !important; margin-top: 1.4rem !important; }

.result-card { padding: 1.4rem 1.5rem !important; margin-bottom: 1.0rem !important; }
.result-card__meta { font-size: 16px !important; }
.result-card__title { font-size: 26px !important; line-height: 1.2 !important; font-weight: 700 !important; }
.result-card__snippet {
    font-size: 19px !important;
    line-height: 1.55 !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 2 !important;
    -webkit-box-orient: vertical !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.result-card--compact { padding: 1.0rem 1.5rem !important; }
.result-card--compact .result-card__snippet {
    -webkit-line-clamp: 1 !important;
}
.chip { font-size: 15px !important; padding: 0.30rem 0.7rem !important; }

.answer-box { padding: 1.5rem !important; }
.answer-box .type-body, .answer-box p { font-size: 22px !important; line-height: 1.55 !important; }

.poster-banner {
    border: 1px solid rgba(140, 171, 255, 0.45);
    border-radius: 14px;
    background: rgba(20, 30, 51, 0.78);
    padding: 0.9rem 1.2rem;
    margin: 0.4rem 0 1.0rem;
    color: rgba(235, 241, 255, 0.96);
    font-size: 22px;
    line-height: 1.35;
    box-shadow: 0 8px 24px rgba(0,0,0,0.22);
}
.poster-banner__label {
    color: rgba(157, 182, 255, 0.92);
    font-size: 16px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}
.poster-banner__title { font-weight: 700; }

.cite {
    color: #ffb558;
    font-style: italic;
    font-weight: 600;
}

/* The home-logo button at the top of the search/similar pages. */
.home-logo-btn { font-size: 18px !important; }
"""


def inject_poster_css(page: Page) -> None:
    page.add_style_tag(content=POSTER_CSS)


# ---------------------------------------------------------------------------
# DOM staging — fictional content per the spec
# ---------------------------------------------------------------------------

SEMANTIC_RESULTS_HTML = """
<h2 class="section-title type-h2">Related Papers</h2>

<article class="result-card">
    <p class="result-card__meta type-meta">Result 1</p>
    <h3 class="result-card__title type-h3">
        Attention Is All You Need: Scaling Transformer Architectures
    </h3>
    <div class="result-card__chips">
        <span class="chip type-caption">Domain: research</span>
        <span class="chip type-caption">Source: arXiv</span>
        <span class="chip type-caption">Score: 0.94</span>
    </div>
    <p class="result-card__snippet type-short">
        This work introduces the Transformer, a model architecture relying
        entirely on self-attention mechanisms to draw global dependencies
        between input and output, dispensing with recurrence and convolutions
        and achieving state-of-the-art results on translation benchmarks.
    </p>
</article>

<article class="result-card result-card--compact">
    <p class="result-card__meta type-meta">Result 2</p>
    <h3 class="result-card__title type-h3">
        A Survey of Attention Variants in Modern Neural Networks
    </h3>
    <div class="result-card__chips">
        <span class="chip type-caption">Domain: blog</span>
        <span class="chip type-caption">Source: Towards Data Science</span>
        <span class="chip type-caption">Score: 0.88</span>
    </div>
    <p class="result-card__snippet type-short">
        A practical tour of additive, multiplicative, scaled-dot-product, and sparse attention variants and when each one pays off.
    </p>
</article>
"""

RAG_ANSWER_HTML = """
<h2 class="section-title type-h2">Answer</h2>
<div class="answer-box">
    <p class="type-body">
        Transfer learning is a technique where a model trained on one task is
        reused as the starting point for a model on a related task
        <span class="cite">(Goodfellow et al., Deep Learning Book)</span>.
        It is most useful when the target dataset is small, since pretrained
        features capture general patterns that would otherwise require large
        amounts of labelled data. The technique underpins most modern vision
        and language systems, where large pretrained backbones are fine-tuned
        on downstream tasks
        <span class="cite">(Howard &amp; Ruder, ULMFiT)</span>.
    </p>
</div>

<h2 class="section-title type-h2">Source</h2>
<article class="result-card">
    <p class="result-card__meta type-meta">Most cited</p>
    <h3 class="result-card__title type-h3">
        Universal Language Model Fine-tuning for Text Classification
    </h3>
    <div class="result-card__chips">
        <span class="chip type-caption">Domain: research</span>
        <span class="chip type-caption">Source: arXiv</span>
    </div>
</article>
"""

SIMILAR_HTML = """
<h1 class="page-title type-h1">Similar Documents</h1>
<p class="type-short">Documents ranked by semantic similarity</p>

<div class="poster-banner">
    <span class="poster-banner__label">Similar to</span>
    <span class="poster-banner__title">
        Attention Is All You Need: Scaling Transformer Architectures
    </span>
</div>

<article class="result-card">
    <p class="result-card__meta type-meta">Result 1</p>
    <h3 class="result-card__title type-h3">
        BERT: Pre-training Deep Bidirectional Transformers
    </h3>
    <div class="result-card__chips">
        <span class="chip type-caption">Domain: research</span>
        <span class="chip type-caption">Source: arXiv</span>
        <span class="chip type-caption">Score: 0.91</span>
    </div>
    <p class="result-card__snippet type-short">
        Pre-trains deep bidirectional representations from unlabelled text by
        jointly conditioning on left and right context across all layers,
        producing a single backbone that fine-tunes to many downstream tasks.
    </p>
</article>

<article class="result-card">
    <p class="result-card__meta type-meta">Result 2</p>
    <h3 class="result-card__title type-h3">
        Vision Transformers: An Image Is Worth 16×16 Words
    </h3>
    <div class="result-card__chips">
        <span class="chip type-caption">Domain: research</span>
        <span class="chip type-caption">Source: arXiv</span>
        <span class="chip type-caption">Score: 0.87</span>
    </div>
    <p class="result-card__snippet type-short">
        Splits an image into fixed-size patches, embeds each, and feeds the
        sequence to a standard Transformer — matching CNN accuracy at scale
        without any image-specific inductive biases.
    </p>
</article>
"""


# JS that wipes existing Related-Papers / Answer / Find-Similar nodes and
# inserts the staged HTML into the main content area.  We target the inner
# block container so Streamlit's outer chrome stays put.
STAGE_JS = r"""
(args) => {
    const { html, hideSearchForm } = args;

    const main = document.querySelector('[data-testid="stMain"]')
              || document.querySelector('section.main')
              || document.body;

    // Remove any prior poster injection.
    main.querySelectorAll('.poster-injection').forEach(n => n.remove());

    // Reset display state on elements we may have toggled in a previous shot.
    main.querySelectorAll('[data-poster-hidden="1"]').forEach(n => {
        n.style.display = '';
        n.removeAttribute('data-poster-hidden');
    });

    const hide = (el) => {
        if (!el) return;
        el.style.display = 'none';
        el.setAttribute('data-poster-hidden', '1');
    };

    // Hide every existing Streamlit-rendered article.result-card / answer-box
    // / matching headings so they don't compete with our staged content.
    main.querySelectorAll(
        'article.result-card, .answer-box, .section-title'
    ).forEach(hide);

    // Hide the "Search" / "Similar Documents" page title and the muted
    // "Find relevant papers..." subtitle that follows it.  Our injected
    // content provides its own headings.
    const pageTitle = main.querySelector('h1.page-title');
    hide(pageTitle);
    main.querySelectorAll('p.type-short').forEach((p) => {
        if (!p.closest('.result-card') && !p.closest('.poster-injection')) {
            hide(p);
        }
    });

    // Optionally hide the search form (used for the RAG and Similar shots).
    const searchFormBlock = main.querySelector('.st-key-search_form');
    if (hideSearchForm) hide(searchFormBlock);

    // Determine where to splice in the staged HTML.  When the search form is
    // visible we want results below it; otherwise place the injection at the
    // top of the main content so the wordmark + injected heading lead.
    let anchor = null;
    if (!hideSearchForm && searchFormBlock) {
        anchor = searchFormBlock;
    } else {
        const logoLink = main.querySelector('a.home-logo-btn');
        anchor = logoLink ? logoLink.closest('[data-testid="stMarkdown"]') : null;
    }

    const wrap = document.createElement('div');
    wrap.className = 'poster-injection';
    wrap.innerHTML = html;

    if (anchor && anchor.parentElement) {
        anchor.parentElement.insertBefore(wrap, anchor.nextSibling);
    } else {
        main.appendChild(wrap);
    }

    // Make sure no scrollbars steal pixels.
    document.documentElement.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
}
"""


def stage_view(page: Page, *, html: str, hide_search_form: bool) -> None:
    page.evaluate(STAGE_JS, {"html": html, "hideSearchForm": hide_search_form})
    # Let layout settle.
    page.wait_for_timeout(400)


# ---------------------------------------------------------------------------
# Navigation: get the browser onto the search page once
# ---------------------------------------------------------------------------

def _debug_screenshot(page: Page, label: str) -> None:
    try:
        path = OUTPUT_DIR / f"_debug_{label}.png"
        page.screenshot(path=str(path), full_page=False)
        print(f"[poster] DEBUG screenshot: {path}")
    except Exception as exc:
        print(f"[poster] DEBUG screenshot failed: {exc}")


def navigate_to_search_page(page: Page, query: str) -> None:
    """Submit one anonymous search from the landing page so we land on the
    search results view.  Subsequent screenshots reuse this page state and
    only swap the content via DOM injection — keeping us under the 3-search
    anonymous limit."""
    page.goto(STREAMLIT_URL, wait_until="domcontentloaded", timeout=120000)

    # Wait for Streamlit's app shell to mount, then for our landing form's
    # textarea.  First-load embedding model + cookie manager warmup is slow.
    try:
        page.wait_for_selector(
            '[data-testid="stApp"]', timeout=120000, state="attached"
        )
        page.wait_for_selector("textarea", timeout=120000)
    except Exception:
        _debug_screenshot(page, "landing_timeout")
        raise

    # Type query into the landing-page textarea (there's only one on landing).
    textarea = page.locator("textarea").first
    textarea.click()
    textarea.fill(query)

    # Click "Start Search".
    page.get_by_role("button", name="Start Search").first.click()

    # The search page renders either Related Papers (results) or an info
    # banner (no results).  Either way, wait for the page-title h1 to flip
    # to "Search".
    try:
        page.wait_for_function(
            """() => {
                const h1 = document.querySelector('h1.page-title');
                return h1 && h1.textContent.trim() === 'Search';
            }""",
            timeout=180000,
        )
    except Exception:
        _debug_screenshot(page, "search_page_timeout")
        raise
    # Belt and braces: wait for any of the markers that indicate the search
    # page has finished its initial render.
    page.wait_for_timeout(1500)


# ---------------------------------------------------------------------------
# Capture pipeline
# ---------------------------------------------------------------------------

SHOTS: list[tuple[str, str, bool]] = [
    # filename, staged HTML, hide the search form?
    ("poster_semantic_search.png", SEMANTIC_RESULTS_HTML, False),
    ("poster_rag_answer.png", RAG_ANSWER_HTML, True),
    ("poster_more_like_this.png", SIMILAR_HTML, True),
]


def capture_all(page: Page) -> Iterator[Path]:
    inject_poster_css(page)

    for filename, html, hide_search_form in SHOTS:
        stage_view(page, html=html, hide_search_form=hide_search_form)
        # Re-inject CSS after every stage in case Streamlit's reactive layer
        # has stripped style tags (it doesn't usually, but cheap insurance).
        inject_poster_css(page)
        page.wait_for_timeout(400)
        out_path = OUTPUT_DIR / filename
        page.screenshot(path=str(out_path), full_page=False)
        print(f"[poster] wrote {out_path}")
        yield out_path


def write_inspection_copy(src: Path) -> Path:
    """Resize to 1100px wide for the print-legibility check."""
    inspect_path = src.with_name(f"_inspect_{src.name}")
    with Image.open(src) as img:
        ratio = INSPECT_WIDTH / img.width
        target = (INSPECT_WIDTH, int(round(img.height * ratio)))
        img.resize(target, Image.LANCZOS).save(inspect_path)
    print(f"[poster] inspect copy: {inspect_path}  ({target[0]}x{target[1]})")
    return inspect_path


def main() -> int:
    streamlit_proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, slow_mo=120)
            context = browser.new_context(
                viewport=VIEWPORT,
                device_scale_factor=DEVICE_SCALE_FACTOR,
            )
            page = context.new_page()

            navigate_to_search_page(
                page, query="attention mechanisms in transformers"
            )

            shots = list(capture_all(page))
            for shot in shots:
                write_inspection_copy(shot)

            browser.close()
    finally:
        stop_streamlit(streamlit_proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
