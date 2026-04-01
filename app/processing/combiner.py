"""
Combiner — merges text summary + figure descriptions into a single
enriched Markdown table row per paper.
"""

from __future__ import annotations

from typing import List

from app.logging_cfg import log
from app.schemas import (
    CombinedSummary,
    FigureDescription,
    PaperMeta,
    PaperStatus,
    TextSummary,
)


def _build_figure_block(figures: List[FigureDescription]) -> str:
    """Format figure descriptions as compact bullet list."""
    if not figures:
        return "_No figures extracted._"
    lines = []
    for fig in figures:
        desc = fig.llm_description or "N/A"
        rel = fig.relevance or ""
        line = f"• **Fig {fig.figure_index + 1}**: {desc}"
        if rel:
            line += f" — _{rel}_"
        lines.append(line)
    return "\n".join(lines)


def build_markdown_row(
    paper: PaperMeta,
    text_summary: TextSummary,
    figures: List[FigureDescription],
) -> str:
    """
    Build a single Markdown table row (actually a mini-card for Telegram readability).
    Telegram supports a subset of Markdown, so we use a card format rather than
    a true HTML table.
    """
    fig_block = _build_figure_block(figures)
    confidence_emoji = "🟢" if text_summary.confidence >= 80 else "🟡" if text_summary.confidence >= 50 else "🔴"

    card = (
        f"📄 **{paper.title}**\n"
        f"🏷 _{paper.domain}_ | {confidence_emoji} Confidence: {text_summary.confidence}%\n"
        f"👥 {', '.join(paper.authors[:3])}\n"
        f"🔗 {paper.url}\n\n"
        f"**What:** {text_summary.what}\n"
        f"**How:** {text_summary.how}\n"
        f"**To Whom:** {text_summary.to_whom}\n\n"
        f"**Figures:**\n{fig_block}\n"
    )
    return card


def combine(
    paper: PaperMeta,
    text_summary: TextSummary,
    figures: List[FigureDescription],
) -> CombinedSummary:
    """Create the final CombinedSummary for one paper."""
    md_row = build_markdown_row(paper, text_summary, figures)
    log.info("combined_summary_built", paper_id=paper.paper_id)
    return CombinedSummary(
        paper=paper,
        text_summary=text_summary,
        figures=figures,
        markdown_row=md_row,
        status=PaperStatus.READY,
    )


def build_daily_digest(summaries: List[CombinedSummary]) -> str:
    """Combine all paper cards into a daily digest message."""
    if not summaries:
        return "📭 No new papers today."

    header = (
        f"🔬 **Daily Research Digest**\n"
        f"_{len(summaries)} papers across {len(set(s.paper.domain for s in summaries))} domains_\n"
        f"{'─' * 30}\n\n"
    )
    cards = "\n\n---\n\n".join(s.markdown_row for s in summaries)
    return header + cards
