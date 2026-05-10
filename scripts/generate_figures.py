"""
Generate all evaluation figures for the SemantikML dissertation.

Run from the project root with the project venv:
    .venv\\Scripts\\python scripts\\generate_figures.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

EVAL_JSON = PROJECT_ROOT / "evaluation" / "eval_results.json"
RAG_EVAL_JSON = PROJECT_ROOT / "evaluation" / "rag_eval_results.json"
REPORT_HTML = PROJECT_ROOT / "backend" / "test" / "report.html"
STREAMLIT_ENTRY = PROJECT_ROOT / "backend" / "streamlit_app" / "main.py"

generated: list[str] = []
skipped: list[tuple[str, str]] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(name)
    print(f"  [OK] {name}")


def skip(name: str, reason: str) -> None:
    skipped.append((name, reason))
    print(f"  [SKIP] {name}: {reason}")


# ── Figure 3: Per-domain retrieval metrics chart ──────────────────────────────

def make_domain_comparison() -> None:
    print("\n[1/5] Domain comparison chart...")
    try:
        data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        by_domain = data["filtered"]["by_domain"]

        domains = ["research", "blog", "documentation"]
        metrics = ["P@5", "MRR", "nDCG@10"]
        keys = ["mean_p_at_5", "mean_mrr", "mean_ndcg_at_10"]

        values = {
            d: [by_domain[d][k] for k in keys]
            for d in domains
        }

        x = np.arange(len(metrics))
        width = 0.25
        colours = ["#2166ac", "#f4a582", "#4dac26"]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for i, (domain, colour) in enumerate(zip(domains, colours)):
            bars = ax.bar(
                x + (i - 1) * width,
                values[domain],
                width,
                label=domain.capitalize(),
                color=colour,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            "Per-Domain Retrieval Metrics (Filtered Mode)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
        ax.xaxis.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            fontsize=10,
        )

        save(fig, "fig_domain_comparison.png")
    except Exception as exc:
        skip("fig_domain_comparison.png", str(exc))


# ── Figure 4: Filtered vs unfiltered comparison ───────────────────────────────

def make_filtered_vs_unfiltered() -> None:
    print("\n[2/5] Filtered vs unfiltered chart...")
    try:
        data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        f_agg = data["filtered"]["aggregate"]
        u_agg = data["unfiltered"]["aggregate"]

        metrics = ["P@5", "P@10", "R@10", "MRR", "nDCG@10"]
        keys = ["mean_p_at_5", "mean_p_at_10", "mean_r_at_10", "mean_mrr", "mean_ndcg_at_10"]

        f_vals = [f_agg[k] for k in keys]
        u_vals = [u_agg[k] for k in keys]

        x = np.arange(len(metrics))
        width = 0.35
        colours = {"filtered": "#2166ac", "unfiltered": "#d6604d"}

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        bars_f = ax.bar(x - width / 2, f_vals, width, label="Filtered",
                        color=colours["filtered"], edgecolor="white", linewidth=0.5)
        bars_u = ax.bar(x + width / 2, u_vals, width, label="Unfiltered",
                        color=colours["unfiltered"], edgecolor="white", linewidth=0.5)

        for bars in (bars_f, bars_u):
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.008,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.10)
        ax.set_title(
            "Filtered vs Unfiltered Aggregate Retrieval Metrics",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc")
        ax.xaxis.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)

        save(fig, "fig_filtered_vs_unfiltered.png")
    except Exception as exc:
        skip("fig_filtered_vs_unfiltered.png", str(exc))


# ── Figure 5: Per-query table ─────────────────────────────────────────────────

def make_query_level_table() -> None:
    print("\n[3/5] Per-query table...")
    try:
        data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        rows = data["filtered"]["per_query"]

        col_labels = ["Query ID", "Domain", "P@10", "MRR", "nDCG@10", "Latency (s)"]
        table_data = [
            [
                r["id"],
                r["domain"].capitalize(),
                f"{r['p_at_10']:.3f}",
                f"{r['mrr']:.3f}",
                f"{r['ndcg_at_10']:.3f}",
                f"{r['latency_s']:.3f}",
            ]
            for r in rows
        ]

        n_rows = len(table_data)
        fig_h = max(6, 0.35 * n_rows + 1.5)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        ax.axis("off")
        fig.patch.set_facecolor("white")

        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))

        # Header style
        for col_idx in range(len(col_labels)):
            cell = tbl[0, col_idx]
            cell.set_facecolor("#2166ac")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#ffffff")
            cell.set_height(0.045)

        # Alternating row colours by domain
        domain_colours = {
            "Research": "#dce9f5",
            "Blog": "#fde8d8",
            "Documentation": "#dff0d8",
        }
        alt_base = {"Research": "#edf4fb", "Blog": "#fef4ec", "Documentation": "#edf7e8"}

        for row_idx, row in enumerate(table_data):
            domain = row[1]
            colour = domain_colours.get(domain, "#f5f5f5") if row_idx % 2 == 0 else alt_base.get(domain, "#fafafa")
            for col_idx in range(len(col_labels)):
                cell = tbl[row_idx + 1, col_idx]
                cell.set_facecolor(colour)
                cell.set_edgecolor("#dddddd")
                cell.set_height(0.038)

        ax.set_title(
            "Per-Query Retrieval Results — Filtered Mode (25 Queries)",
            fontsize=12,
            fontweight="bold",
            pad=10,
            y=1.0,
        )

        # Legend
        patches = [
            mpatches.Patch(color="#dce9f5", label="Research"),
            mpatches.Patch(color="#fde8d8", label="Blog"),
            mpatches.Patch(color="#dff0d8", label="Documentation"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8,
                  frameon=True, framealpha=0.8, ncol=3)

        save(fig, "fig_query_level_table.png")
    except Exception as exc:
        skip("fig_query_level_table.png", str(exc))


# ── Figures 1–2: pytest-html screenshots ─────────────────────────────────────

def make_pytest_screenshots() -> None:
    print("\n[4/5] pytest-html screenshots (Playwright)...")
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]
    except ImportError:
        skip("fig_test_summary.png", "playwright not installed — run: pip install playwright && python -m playwright install chromium")
        skip("fig_test_list.png", "playwright not installed")
        return

    if not REPORT_HTML.exists():
        skip("fig_test_summary.png", f"report.html not found at {REPORT_HTML}")
        skip("fig_test_list.png", "report.html not found")
        return

    report_url = "file:///" + REPORT_HTML.as_posix().lstrip("/")
    # Windows path fix: file:///C:/...
    if sys.platform == "win32":
        report_url = "file:///" + str(REPORT_HTML).replace("\\", "/")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(report_url, wait_until="networkidle")
            page.wait_for_timeout(1500)

            # ── Summary bar ──────────────────────────────────────────────────
            # pytest-html wraps summary in a <p class="run-count"> or similar.
            # Try to find the summary area and screenshot it.
            try:
                # Expand "passed" filter to show tests if collapsed
                passed_toggle = page.query_selector('input[data-test-result="passed"]')
                if passed_toggle:
                    passed_toggle.check()
                    page.wait_for_timeout(500)
            except Exception:
                pass

            # Screenshot the top 200px as the summary bar
            summary_path = str(FIGURES_DIR / "fig_test_summary.png")
            page.screenshot(
                path=summary_path,
                clip={"x": 0, "y": 0, "width": 1280, "height": 220},
            )
            generated.append("fig_test_summary.png")
            print("  [OK] fig_test_summary.png")

            # ── Test list ────────────────────────────────────────────────────
            # Scroll down a bit to get past the summary into the test table
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(300)

            list_path = str(FIGURES_DIR / "fig_test_list.png")
            page.screenshot(
                path=list_path,
                clip={"x": 0, "y": 150, "width": 1280, "height": 800},
            )
            generated.append("fig_test_list.png")
            print("  [OK] fig_test_list.png")

            browser.close()
    except Exception as exc:
        skip("fig_test_summary.png", str(exc))
        skip("fig_test_list.png", str(exc))


# ── Figure 6: Streamlit screenshots ──────────────────────────────────────────

def _wait_for_streamlit(url: str = "http://localhost:8501", timeout: int = 60) -> bool:
    """Poll until Streamlit HTTP endpoint responds."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def make_streamlit_screenshots() -> None:
    print("\n[5/5] Streamlit UI screenshots (Playwright)...")

    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]
    except ImportError:
        skip("fig_ui_login.png", "playwright not installed")
        skip("fig_ui_search.png", "playwright not installed")
        skip("fig_ui_rag.png", "playwright not installed")
        return

    if not STREAMLIT_ENTRY.exists():
        skip("fig_ui_login.png", f"Streamlit entry not found: {STREAMLIT_ENTRY}")
        skip("fig_ui_search.png", "Streamlit entry not found")
        skip("fig_ui_rag.png", "Streamlit entry not found")
        return

    # Start Streamlit
    python_exe = sys.executable
    env = os.environ.copy()
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

    proc = subprocess.Popen(
        [
            python_exe, "-m", "streamlit", "run",
            str(STREAMLIT_ENTRY),
            "--server.headless", "true",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    print("  Waiting for Streamlit to start (up to 90 s)...")
    if not _wait_for_streamlit(timeout=90):
        proc.terminate()
        skip("fig_ui_login.png", "Streamlit did not respond within 90 s")
        skip("fig_ui_search.png", "Streamlit did not respond within 90 s")
        skip("fig_ui_rag.png", "Streamlit did not respond within 90 s")
        return

    print("  Streamlit is up. Warming up (waiting for model load)...")

    # Warm up: open a plain HTTP request to trigger model loading, then
    # wait until the Streamlit app renders its first textarea element.
    try:
        with sync_playwright() as pw:
            wb = pw.chromium.launch(headless=True)
            wp = wb.new_page(viewport={"width": 1280, "height": 800})
            wp.goto("http://localhost:8501", wait_until="networkidle")
            # Wait up to 90 s for the first form textarea (signals model loaded)
            try:
                wp.wait_for_selector("textarea", timeout=90_000)
                print("  App fully loaded (textarea visible).")
            except Exception:
                print("  WARNING: textarea did not appear during warmup — screenshots may be partial.")
            wb.close()
    except Exception as exc:
        print(f"  WARNING: warmup failed ({exc})")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            _take_streamlit_shots(browser)
            browser.close()
    except Exception as exc:
        print(f"  [ERROR] Streamlit screenshots failed: {exc}")
        skip("fig_ui_login.png", str(exc))
        skip("fig_ui_search.png", str(exc))
        skip("fig_ui_rag.png", str(exc))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _take_streamlit_shots(browser) -> None:  # noqa: ANN001
    """Inner function: browser is already launched."""
    BASE_URL = "http://localhost:8501"
    VP = {"width": 1280, "height": 800}

    # ── fig_ui_login.png ─────────────────────────────────────────────────────
    # Strategy: open a fresh context, do 4 searches (limit=3 free), 4th redirects to login.
    ctx_login = browser.new_context(viewport=VP)
    page_login = ctx_login.new_page()
    try:
        page_login.goto(BASE_URL, wait_until="networkidle")
        page_login.wait_for_timeout(5000)

        # We need to exhaust the 3-search free limit, then trigger a 4th search.
        # Each search: fill query, click Start Search, wait, navigate back.
        dummy_queries = [
            "neural network training",
            "support vector machine kernel",
            "random forest ensemble",
        ]
        for i, q in enumerate(dummy_queries):
            _fill_and_search_landing(page_login, q)
            page_login.wait_for_timeout(4000)
            if i < len(dummy_queries) - 1:
                # Navigate back to landing for next search
                try:
                    home_btn = page_login.query_selector(".home-logo-btn")
                    if home_btn:
                        home_btn.click()
                    else:
                        page_login.goto(BASE_URL + "?go=landing", wait_until="networkidle")
                    page_login.wait_for_timeout(2000)
                except Exception:
                    page_login.goto(BASE_URL, wait_until="networkidle")
                    page_login.wait_for_timeout(2000)

        # Now on search page with anon_count=3; navigate back and trigger a 4th search
        try:
            home_btn = page_login.query_selector(".home-logo-btn")
            if home_btn:
                home_btn.click()
            else:
                page_login.goto(BASE_URL + "?go=landing", wait_until="networkidle")
            page_login.wait_for_timeout(2000)
        except Exception:
            pass

        _fill_and_search_landing(page_login, "deep learning optimization")
        page_login.wait_for_timeout(3000)

        # Should now be on login page
        page_login.screenshot(path=str(FIGURES_DIR / "fig_ui_login.png"))
        generated.append("fig_ui_login.png")
        print("  [OK] fig_ui_login.png")
    except Exception as exc:
        skip("fig_ui_login.png", str(exc))
    finally:
        ctx_login.close()

    # ── fig_ui_search.png ─────────────────────────────────────────────────────
    # Fresh context: first search shows results (within free limit)
    ctx_search = browser.new_context(viewport=VP)
    page_search = ctx_search.new_page()
    try:
        page_search.goto(BASE_URL, wait_until="networkidle")
        page_search.wait_for_timeout(5000)

        _fill_and_search_landing(page_search, "transformer attention mechanism")
        # Wait for results to load (semantic search is fast ~2s)
        page_search.wait_for_timeout(8000)
        page_search.screenshot(path=str(FIGURES_DIR / "fig_ui_search.png"))
        generated.append("fig_ui_search.png")
        print("  [OK] fig_ui_search.png")
    except Exception as exc:
        skip("fig_ui_search.png", str(exc))
    finally:
        ctx_search.close()

    # ── fig_ui_rag.png ────────────────────────────────────────────────────────
    # Fresh context: show landing page with RAG Answer configured (without submitting,
    # since RAG takes 300-500 s for anonymous users). We screenshot the configured form.
    ctx_rag = browser.new_context(viewport=VP)
    page_rag = ctx_rag.new_page()
    try:
        page_rag.goto(BASE_URL, wait_until="networkidle")
        page_rag.wait_for_timeout(5000)

        # Type query
        ta = page_rag.query_selector("textarea")
        if ta:
            ta.fill("What is batch normalisation?")

        # Select "RAG Answer" radio button
        try:
            # Streamlit renders radio as <label> elements; click the one containing "RAG"
            labels = page_rag.query_selector_all("label")
            for lbl in labels:
                if "RAG" in (lbl.inner_text() or ""):
                    lbl.click()
                    break
        except Exception:
            pass

        page_rag.wait_for_timeout(1000)

        # Expand parameters to show RAG Answer is selected
        try:
            expander = page_rag.query_selector('summary:has-text("Search Parameters")')
            if expander:
                expander.click()
                page_rag.wait_for_timeout(500)
        except Exception:
            pass

        page_rag.screenshot(path=str(FIGURES_DIR / "fig_ui_rag.png"))
        generated.append("fig_ui_rag.png")
        print("  [OK] fig_ui_rag.png  (NOTE: shows configured RAG form, not result — RAG latency is 300-500 s)")
    except Exception as exc:
        skip("fig_ui_rag.png", str(exc))
    finally:
        ctx_rag.close()


def _fill_and_search_landing(page, query: str) -> None:  # noqa: ANN001
    """Fill the landing page search form and click Start Search."""
    # The landing page has a text area with key="query_input"
    # Streamlit text areas are rendered as <textarea>
    textarea = page.wait_for_selector("textarea", timeout=60_000)
    textarea.fill(query)
    page.wait_for_timeout(300)

    # Click "Start Search" button
    # Streamlit form submit buttons are <button> elements
    start_btn = page.query_selector('button:has-text("Start Search")')
    if start_btn:
        start_btn.click()
    else:
        # Fallback: press Enter in the textarea
        textarea.press("Enter")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SemantikML — Dissertation Figure Generator")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

    make_domain_comparison()
    make_filtered_vs_unfiltered()
    make_query_level_table()
    make_pytest_screenshots()
    make_streamlit_screenshots()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Generated ({len(generated)}):")
    for name in generated:
        print(f"  figures/{name}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for name, reason in skipped:
            print(f"  figures/{name}  — {reason}")

    print("\nDone.")


if __name__ == "__main__":
    main()
