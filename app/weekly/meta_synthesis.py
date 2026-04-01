"""
Weekly Meta-Synthesis — runs every Sunday (configurable).

Flow:
  1. Pull all 👍-approved papers from the past 7 days.
  2. Optionally retrieve related older papers from the vector store (RAG).
  3. Build a hierarchical prompt with domain-grouped summaries.
  4. Run the LLM to produce a cross-domain macro-trend report.
  5. Send the report to Telegram.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import List

from app.config import settings
from app.llm_client import chat
from app.logging_cfg import log
from app.processing.prompts import WEEKLY_SYNTHESIS_SYSTEM, WEEKLY_SYNTHESIS_USER
from app.schemas import CombinedSummary, WeeklySynthesis
from app.storage.paper_db import get_approved_papers_since, row_to_combined
from app.storage.vector_store import query_similar


def _group_by_domain(summaries: List[CombinedSummary]) -> str:
    """Format approved papers grouped by domain for the synthesis prompt."""
    domains: dict[str, list[str]] = {}
    for s in summaries:
        domain = s.paper.domain or "Other"
        if domain not in domains:
            domains[domain] = []
        entry = (
            f"  - **{s.paper.title}**\n"
            f"    What: {s.text_summary.what}\n"
            f"    How: {s.text_summary.how}\n"
            f"    To Whom: {s.text_summary.to_whom}"
        )
        domains[domain].append(entry)

    sections = []
    for domain, entries in sorted(domains.items()):
        sections.append(f"\n### {domain}\n" + "\n".join(entries))

    return "\n".join(sections)


def _enrich_with_rag(summaries: List[CombinedSummary]) -> str:
    """Use vector store to find related older papers for context."""
    if not summaries:
        return ""

    # Build a composite query from this week's themes
    themes = " ".join(s.text_summary.what[:100] for s in summaries[:5])
    related = query_similar(themes, n_results=5)

    if not related:
        return ""

    lines = ["\n### Related Earlier Papers (from vector store)"]
    for r in related:
        meta = r.get("metadata", {})
        lines.append(
            f"  - **{meta.get('title', 'Unknown')}** ({meta.get('domain', '')}): "
            f"{meta.get('what', 'N/A')}"
        )

    return "\n".join(lines)


def generate_synthesis(summaries: List[CombinedSummary]) -> str:
    """Run the LLM to produce the cross-domain report."""
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=7)

    paper_text = _group_by_domain(summaries)
    rag_context = _enrich_with_rag(summaries)

    full_summaries = paper_text
    if rag_context:
        full_summaries += "\n" + rag_context

    user_prompt = WEEKLY_SYNTHESIS_USER.format(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        paper_summaries=full_summaries,
    )

    try:
        return chat(
            system_prompt=WEEKLY_SYNTHESIS_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.4,
            max_tokens=2048,
        )
    except Exception as exc:
        log.error("weekly_synthesis_llm_failed", error=str(exc))
        return f"⚠️ Synthesis generation failed: {exc}"


def run_weekly_synthesis() -> None:
    """Full weekly synthesis pipeline."""
    log.info("weekly_synthesis_start")

    # Get approved papers from the last 7 days
    since = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    rows = get_approved_papers_since(since)

    if not rows:
        log.info("no_approved_papers_for_synthesis")
        _send_telegram("📭 *Weekly Synthesis*: No approved papers this week.")
        return

    summaries = [row_to_combined(row) for row in rows]
    log.info("weekly_synthesis_papers", count=len(summaries))

    report = generate_synthesis(summaries)

    # Build the full message
    header = (
        f"📊 *Weekly Cross-Domain Meta-Synthesis*\n"
        f"_{since} to {dt.date.today().isoformat()}_\n"
        f"_{len(summaries)} approved papers across "
        f"{len(set(s.paper.domain for s in summaries))} domains_\n"
        f"{'─' * 30}\n\n"
    )
    full_message = header + report

    _send_telegram(full_message)
    log.info("weekly_synthesis_complete")


def _send_telegram(message: str) -> None:
    """Send a message to the configured Telegram chat (sync wrapper)."""
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.warning("telegram_not_configured_for_synthesis")
        return

    import httpx

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"

    # Split long messages (Telegram limit: 4096 chars)
    chunks = []
    while len(message) > 4000:
        split_at = message[:4000].rfind("\n")
        if split_at == -1:
            split_at = 4000
        chunks.append(message[:split_at])
        message = message[split_at:]
    chunks.append(message)

    for chunk in chunks:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(url, json={
                    "chat_id": int(settings.telegram_chat_id),
                    "text": chunk,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                })
                resp.raise_for_status()
        except Exception as exc:
            log.error("telegram_send_failed", error=str(exc))
            # Try without markdown
            try:
                with httpx.Client(timeout=30) as client:
                    client.post(url, json={
                        "chat_id": int(settings.telegram_chat_id),
                        "text": chunk,
                        "disable_web_page_preview": True,
                    })
            except Exception:
                pass
