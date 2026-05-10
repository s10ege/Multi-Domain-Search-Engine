"""Build a compressed and specific PDF report for all evaluation artifacts.

Usage:
    python evaluation/generate_evaluation_pdf.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from fpdf import FPDF

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = EVAL_DIR / "EVALUATION_REPORT.pdf"


class ReportPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "SemantikML Evaluation Report", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, f"Generated: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_title(pdf: ReportPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def add_h2(pdf: ReportPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(245, 245, 245)
    pdf.cell(0, 8, text, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def add_body(pdf: ReportPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, text)


def add_bullet(pdf: ReportPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, f"- {text}")


def draw_table(
    pdf: ReportPDF,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float],
) -> None:
    row_h = 6
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 230, 230)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], row_h, h, border=1, fill=True)
    pdf.ln(row_h)

    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(230, 230, 230)
            for i, h in enumerate(headers):
                pdf.cell(widths[i], row_h, h, border=1, fill=True)
            pdf.ln(row_h)
            pdf.set_font("Helvetica", "", 9)

        for i, col in enumerate(row):
            pdf.cell(widths[i], row_h, str(col), border=1)
        pdf.ln(row_h)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def sec(value: float) -> str:
    return f"{value:.3f}s"


def ms(value: float) -> str:
    return f"{value * 1000:.0f}ms"


def main() -> None:
    eval_results = load_json(EVAL_DIR / "eval_results.json")
    rag_results = load_json(EVAL_DIR / "rag_eval_results.json")
    eval_queries = load_json(EVAL_DIR / "eval_queries.json")

    filtered = eval_results["filtered"]
    unfiltered = eval_results["unfiltered"]

    f_agg = filtered["aggregate"]
    u_agg = unfiltered["aggregate"]
    by_domain = filtered["by_domain"]

    rag_agg = rag_results["aggregate"]
    rag_by_domain = rag_results["by_domain"]

    total_docs = 9000
    domain_counts = {"research": 4000, "blog": 3000, "documentation": 2000}

    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    add_title(pdf, "Compressed Evaluation Report")
    add_body(
        pdf,
        "This PDF compresses all artifacts under evaluation/ into a single, specific summary: "
        "retrieval quality, domain-level behavior, filtered-vs-unfiltered impact, and RAG quality.",
    )
    pdf.ln(2)

    add_h2(pdf, "1) Evaluation Scope")
    add_bullet(pdf, f"Corpus size: {total_docs} docs")
    add_bullet(
        pdf,
        "Domain split: "
        + ", ".join(f"{k}={v}" for k, v in domain_counts.items()),
    )
    add_bullet(pdf, f"Retrieval evaluation queries: {len(eval_queries)}")
    add_bullet(pdf, f"RAG evaluation queries: {rag_agg['n_queries']}")
    add_bullet(pdf, "Retrieval metrics: P@5, P@10, R@5, R@10, MRR, nDCG@10, latency")
    add_bullet(pdf, "RAG metrics: Faithfulness (0-2), Relevance (0-2), latency")

    pdf.ln(2)
    add_h2(pdf, "2) Retrieval Aggregate (All Queries)")

    retrieval_rows = [
        [
            "Filtered",
            pct(f_agg["mean_p_at_5"]),
            pct(f_agg["mean_p_at_10"]),
            pct(f_agg["mean_r_at_10"]),
            f"{f_agg['mean_mrr']:.3f}",
            f"{f_agg['mean_ndcg_at_10']:.3f}",
            ms(f_agg["mean_latency_s"]),
        ],
        [
            "Unfiltered",
            pct(u_agg["mean_p_at_5"]),
            pct(u_agg["mean_p_at_10"]),
            pct(u_agg["mean_r_at_10"]),
            f"{u_agg['mean_mrr']:.3f}",
            f"{u_agg['mean_ndcg_at_10']:.3f}",
            ms(u_agg["mean_latency_s"]),
        ],
        [
            "Delta",
            f"{(f_agg['mean_p_at_5'] - u_agg['mean_p_at_5']) * 100:+.1f}pp",
            f"{(f_agg['mean_p_at_10'] - u_agg['mean_p_at_10']) * 100:+.1f}pp",
            f"{(f_agg['mean_r_at_10'] - u_agg['mean_r_at_10']) * 100:+.1f}pp",
            f"{(f_agg['mean_mrr'] - u_agg['mean_mrr']):+.3f}",
            f"{(f_agg['mean_ndcg_at_10'] - u_agg['mean_ndcg_at_10']):+.3f}",
            f"{(f_agg['mean_latency_s'] - u_agg['mean_latency_s']) * 1000:+.0f}ms",
        ],
    ]

    draw_table(
        pdf,
        ["Mode", "P@5", "P@10", "R@10", "MRR", "nDCG@10", "Latency"],
        retrieval_rows,
        [28, 24, 24, 24, 22, 28, 28],
    )

    add_bullet(
        pdf,
        "Domain filtering improves quality and speed at the same time: "
        "MRR +0.073, nDCG@10 +0.066, mean latency -487ms.",
    )

    pdf.ln(2)
    add_h2(pdf, "3) Retrieval by Domain (Filtered)")

    domain_rows = []
    query_counts = {"research": 10, "blog": 8, "documentation": 7}
    for domain in ("research", "blog", "documentation"):
        m = by_domain[domain]
        domain_rows.append(
            [
                f"{domain} (n={query_counts[domain]})",
                pct(m["mean_p_at_5"]),
                pct(m["mean_p_at_10"]),
                pct(m["mean_r_at_10"]),
                f"{m['mean_mrr']:.3f}",
                f"{m['mean_ndcg_at_10']:.3f}",
                ms(m["mean_latency_s"]),
            ]
        )

    draw_table(
        pdf,
        ["Domain", "P@5", "P@10", "R@10", "MRR", "nDCG@10", "Latency"],
        domain_rows,
        [45, 20, 20, 20, 18, 24, 24],
    )

    add_bullet(pdf, "Research is near-perfect at rank quality: MRR=1.000, nDCG@10=0.979.")
    add_bullet(pdf, "Documentation is also perfect on rank position: MRR=1.000, nDCG@10=1.000.")
    add_bullet(pdf, "Blog is the weak domain: MRR=0.500, nDCG@10=0.490.")

    pdf.ln(2)
    add_h2(pdf, "4) Query-Level Retrieval Failures (Filtered)")

    per_query = filtered["per_query"]
    worst_queries = sorted(
        per_query,
        key=lambda x: (x["p_at_10"], x["ndcg_at_10"], x["mrr"]),
    )[:8]

    worst_rows = [
        [
            q["id"],
            q["domain"],
            f"{q['p_at_10']:.2f}",
            f"{q['mrr']:.2f}",
            f"{q['ndcg_at_10']:.2f}",
            sec(q["latency_s"]),
        ]
        for q in worst_queries
    ]

    draw_table(
        pdf,
        ["ID", "Domain", "P@10", "MRR", "nDCG@10", "Latency"],
        worst_rows,
        [18, 32, 20, 20, 24, 24],
    )

    add_bullet(pdf, "Critical zeros are concentrated in blog queries: b04, b05, b06, b07.")
    add_bullet(pdf, "d04 (DataLoader) is partially covered and should be expanded in docs corpus.")

    pdf.ln(2)
    add_h2(pdf, "5) RAG Quality")

    rag_rows = [
        [
            "Overall",
            f"{rag_agg['mean_faithfulness']:.2f}/2.00",
            f"{rag_agg['mean_relevance']:.2f}/2.00",
            sec(rag_agg["mean_latency_s"]),
        ]
    ]

    for domain in ("research", "blog", "documentation"):
        m = rag_by_domain[domain]
        rag_rows.append(
            [
                domain,
                f"{m['mean_faithfulness']:.2f}/2.00",
                f"{m['mean_relevance']:.2f}/2.00",
                "-",
            ]
        )

    draw_table(
        pdf,
        ["Scope", "Faithfulness", "Relevance", "Latency"],
        rag_rows,
        [45, 35, 35, 30],
    )

    add_bullet(pdf, "RAG answers are consistently on-topic (overall relevance 2.00/2.00).")
    add_bullet(pdf, "Faithfulness drops in blog content (1.00/2.00), indicating weak grounding.")
    add_bullet(pdf, "Mean generation latency is high at 336.087s/query on CPU.")

    pdf.ln(2)
    add_h2(pdf, "6) Low-Faithfulness RAG Cases")

    low_faith = sorted(
        rag_results["per_query"],
        key=lambda x: (x["faithfulness"], x["relevance"], x["latency_s"]),
    )[:3]

    low_rows = [
        [
            q["id"],
            q["domain"],
            str(q["faithfulness"]),
            str(q["relevance"]),
            sec(q["latency_s"]),
        ]
        for q in low_faith
    ]

    draw_table(
        pdf,
        ["ID", "Domain", "Faith", "Rel", "Latency"],
        low_rows,
        [24, 32, 20, 20, 24],
    )

    for case in low_faith:
        add_bullet(
            pdf,
            f"{case['id']}: faithfulness={case['faithfulness']}/2, "
            f"reason={case['scoring_reasoning']}",
        )

    pdf.ln(2)
    add_h2(pdf, "7) Root Causes and Specific Actions")

    actions = [
        "Blog corpus quality: remove duplicates and low-information templates before indexing.",
        "Coverage gaps: add missing DataLoader and autograd-centric docs to documentation domain.",
        "Retrieval policy: keep domain filtering ON by default; it improves both quality and speed.",
        "RAG runtime: move generation to GPU or smaller model for interactive latency targets.",
        "Monitoring: track per-domain MRR and faithfulness weekly to catch regressions early.",
    ]
    for action in actions:
        add_bullet(pdf, action)

    pdf.ln(2)
    add_h2(pdf, "8) Artifact Inventory")
    inventory = [
        "evaluation/eval_queries.json - 25 labeled retrieval queries",
        "evaluation/raw_query_results.json - raw top-10 hits per query",
        "evaluation/eval_results.json - filtered/unfiltered retrieval metrics",
        "evaluation/rag_eval_results.json - RAG scores and per-query outputs",
        "evaluation/eval_search.py - retrieval evaluation pipeline",
        "evaluation/eval_rag.py - RAG evaluation pipeline",
        "evaluation/EVALUATION_SUMMARY.md - narrative summary",
    ]
    for item in inventory:
        add_bullet(pdf, item)

    pdf.output(str(OUTPUT_PDF))
    print(f"Wrote: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
