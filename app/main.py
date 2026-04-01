"""
FastAPI Application — lightweight API for health checks, status,
and manual triggers.

Most work happens via RQ tasks and the scheduler. This API provides:
  • /health — liveness probe
  • /status — pipeline statistics
  • /trigger/ingest — manual daily ingestion
  • /trigger/digest — push ready papers to Telegram
  • /trigger/synthesis — run weekly synthesis now
  • /papers — list papers with optional status filter
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_cfg import setup_logging, log
from app.storage.paper_db import (
    get_ready_papers,
    get_all_approved_papers,
    get_queued_papers,
    get_paper,
    init_db,
)

setup_logging()
init_db()

app = FastAPI(
    title="Research Summarization Pipeline",
    version="1.0.0",
    description="Autonomous research paper ingestion, analysis, and delivery system.",
)


@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "research-pipeline"}


@app.get("/status")
async def status():
    """Pipeline statistics."""
    queued = get_queued_papers()
    ready = get_ready_papers()
    approved = get_all_approved_papers()
    return {
        "queued": len(queued),
        "ready": len(ready),
        "approved": len(approved),
        "domains": settings.domain_list,
        "models": {
            "text": settings.ollama_text_model,
            "vision": settings.ollama_vision_model,
            "embed": settings.ollama_embed_model,
        },
    }


@app.post("/trigger/ingest")
async def trigger_ingest():
    """Manually trigger daily ingestion."""
    from app.tasks.ingest_job import run_daily_ingestion

    try:
        count = run_daily_ingestion()
        return {"status": "ok", "papers_enqueued": count}
    except Exception as exc:
        log.error("manual_ingest_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/trigger/digest")
async def trigger_digest():
    """Push ready papers to Telegram (via direct API call)."""
    rows = get_ready_papers()
    if not rows:
        return {"status": "ok", "message": "No papers ready"}

    # This is a simplified sync push — the bot handles the full flow
    return {"status": "ok", "ready_papers": len(rows), "message": "Use /digest in Telegram"}


@app.post("/trigger/synthesis")
async def trigger_synthesis():
    """Run weekly synthesis immediately."""
    from app.weekly.meta_synthesis import run_weekly_synthesis

    try:
        run_weekly_synthesis()
        return {"status": "ok", "message": "Synthesis complete"}
    except Exception as exc:
        log.error("manual_synthesis_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/papers")
async def list_papers(status: Optional[str] = Query(None)):
    """List papers, optionally filtered by status."""
    from app.storage.paper_db import _get_conn

    conn = _get_conn()
    cur = conn.cursor()

    if status:
        cur.execute("SELECT paper_id, title, domain, status, updated_at FROM papers WHERE status = ? ORDER BY updated_at DESC LIMIT 50", (status,))
    else:
        cur.execute("SELECT paper_id, title, domain, status, updated_at FROM papers ORDER BY updated_at DESC LIMIT 50")

    rows = cur.fetchall()
    return [
        {
            "paper_id": r["paper_id"],
            "title": r["title"],
            "domain": r["domain"],
            "status": r["status"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


@app.get("/papers/{paper_id}")
async def get_paper_detail(paper_id: str):
    """Get full detail for one paper."""
    row = get_paper(paper_id)
    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Parse JSON fields for clean output
    result = dict(row)
    for field in ("authors", "figures"):
        if field in result and isinstance(result[field], str):
            try:
                result[field] = json.loads(result[field])
            except Exception:
                pass
    if "text_summary" in result and isinstance(result["text_summary"], str):
        try:
            result["text_summary"] = json.loads(result["text_summary"])
        except Exception:
            pass

    return result
