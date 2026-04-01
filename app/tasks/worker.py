"""
RQ Worker Tasks — the processing backbone.

Three queues:
  • text_pipeline    — extract text + run LLM summary
  • figure_pipeline  — extract figures + run vision LLM
  • combine_pipeline — merge results + mark as ready

Each task is idempotent and stores results in the paper DB.
"""

from __future__ import annotations

import json
from pathlib import Path

from redis import Redis
from rq import Queue

from app.config import settings
from app.logging_cfg import log
from app.processing.text_pipeline import analyse_text
from app.processing.figure_pipeline import analyse_all_figures
from app.processing.combiner import combine
from app.schemas import PaperMeta, PaperStatus, TextSummary, FigureDescription
from app.storage.paper_db import (
    get_paper,
    save_text_summary,
    save_figures,
    save_combined,
    update_status,
)

redis_conn = Redis.from_url(settings.redis_url)
text_q = Queue("text_pipeline", connection=redis_conn)
figure_q = Queue("figure_pipeline", connection=redis_conn)
combine_q = Queue("combine_pipeline", connection=redis_conn)


# ─────────────────────────────────────────────────────────────────
#  Task 1: Text Pipeline
# ─────────────────────────────────────────────────────────────────
def task_text_pipeline(paper_id: str, pdf_path: str) -> dict:
    """
    RQ task: extract text from PDF → run structured LLM analysis.
    Stores TextSummary in DB, then enqueues combine step.
    """
    log.info("task_text_start", paper_id=paper_id)
    update_status(paper_id, PaperStatus.TEXT_PROCESSING)

    row = get_paper(paper_id)
    if not row:
        log.error("paper_not_found", paper_id=paper_id)
        return {"error": "Paper not found"}

    paper = PaperMeta(
        paper_id=row["paper_id"],
        title=row["title"],
        authors=json.loads(row.get("authors", "[]")),
        abstract=row.get("abstract", ""),
        url=row.get("url", ""),
        pdf_url=row.get("pdf_url", ""),
        source=row.get("source", "arxiv"),
        domain=row.get("domain", ""),
    )

    try:
        summary = analyse_text(paper, Path(pdf_path))
        save_text_summary(paper_id, summary)
        log.info("task_text_complete", paper_id=paper_id, confidence=summary.confidence)

        # Check if figure pipeline is also done → enqueue combine
        _maybe_enqueue_combine(paper_id)

        return summary.model_dump()
    except Exception as exc:
        log.error("task_text_failed", paper_id=paper_id, error=str(exc))
        update_status(paper_id, PaperStatus.ERROR)
        raise


# ─────────────────────────────────────────────────────────────────
#  Task 2: Figure Pipeline
# ─────────────────────────────────────────────────────────────────
def task_figure_pipeline(paper_id: str, pdf_path: str) -> dict:
    """
    RQ task: extract figures from PDF → analyse with vision LLM.
    Stores FigureDescriptions in DB, then enqueues combine step.
    """
    log.info("task_figure_start", paper_id=paper_id)
    update_status(paper_id, PaperStatus.FIGURE_PROCESSING)

    row = get_paper(paper_id)
    if not row:
        log.error("paper_not_found", paper_id=paper_id)
        return {"error": "Paper not found"}

    paper = PaperMeta(
        paper_id=row["paper_id"],
        title=row["title"],
        authors=json.loads(row.get("authors", "[]")),
        domain=row.get("domain", ""),
    )

    try:
        figures = analyse_all_figures(paper, Path(pdf_path))
        save_figures(paper_id, figures)
        log.info("task_figure_complete", paper_id=paper_id, count=len(figures))

        _maybe_enqueue_combine(paper_id)

        return {"figures": len(figures)}
    except Exception as exc:
        log.error("task_figure_failed", paper_id=paper_id, error=str(exc))
        update_status(paper_id, PaperStatus.ERROR)
        raise


# ─────────────────────────────────────────────────────────────────
#  Task 3: Combine Pipeline
# ─────────────────────────────────────────────────────────────────
def task_combine(paper_id: str) -> dict:
    """
    RQ task: merge text summary + figure descriptions into final card.
    """
    log.info("task_combine_start", paper_id=paper_id)
    update_status(paper_id, PaperStatus.COMBINING)

    row = get_paper(paper_id)
    if not row:
        return {"error": "Paper not found"}

    paper = PaperMeta(
        paper_id=row["paper_id"],
        title=row["title"],
        authors=json.loads(row.get("authors", "[]")),
        abstract=row.get("abstract", ""),
        url=row.get("url", ""),
        pdf_url=row.get("pdf_url", ""),
        source=row.get("source", "arxiv"),
        domain=row.get("domain", ""),
        published=row.get("published"),
    )

    try:
        ts = TextSummary.model_validate_json(row.get("text_summary", "{}"))
    except Exception:
        ts = TextSummary(what=paper.title, how="N/A", to_whom="N/A", domain=paper.domain)

    try:
        figs_raw = json.loads(row.get("figures", "[]"))
        figs = [FigureDescription(**f) for f in figs_raw]
    except Exception:
        figs = []

    combined = combine(paper, ts, figs)
    save_combined(paper_id, combined)
    log.info("task_combine_complete", paper_id=paper_id)

    return {"status": "ready", "paper_id": paper_id}


# ─────────────────────────────────────────────────────────────────
#  Helper: enqueue combine only when both pipelines are done
# ─────────────────────────────────────────────────────────────────
def _maybe_enqueue_combine(paper_id: str) -> None:
    """
    Check if both text and figure results are stored.
    If yes, enqueue the combine task.
    """
    row = get_paper(paper_id)
    if not row:
        return

    has_text = row.get("text_summary", "{}") != "{}"
    has_figures = row.get("figures", "[]") != "[]"

    # Also check if one pipeline stored an empty but valid result
    try:
        ts = json.loads(row.get("text_summary", "{}"))
        has_text = bool(ts.get("what"))
    except Exception:
        has_text = False

    # Figures can legitimately be empty (paper has no figures)
    # So we check if figure_processing status was reached
    status = row.get("status", "")
    figures_attempted = status in (
        PaperStatus.FIGURE_PROCESSING.value,
        PaperStatus.TEXT_PROCESSING.value,
        PaperStatus.COMBINING.value,
    )

    if has_text and figures_attempted:
        combine_q.enqueue(
            task_combine,
            paper_id,
            job_timeout="10m",
            retry=None,
        )
        log.info("combine_enqueued", paper_id=paper_id)


# ─────────────────────────────────────────────────────────────────
#  Orchestrator: enqueue both pipelines for a paper
# ─────────────────────────────────────────────────────────────────
def enqueue_paper_processing(paper_id: str, pdf_path: str) -> None:
    """Enqueue both text and figure pipelines in parallel."""
    text_q.enqueue(
        task_text_pipeline,
        paper_id,
        pdf_path,
        job_timeout="15m",
        retry=None,
    )
    figure_q.enqueue(
        task_figure_pipeline,
        paper_id,
        pdf_path,
        job_timeout="20m",
        retry=None,
    )
    log.info("processing_enqueued", paper_id=paper_id)
