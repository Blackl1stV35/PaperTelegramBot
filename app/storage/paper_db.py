"""
Paper DB — lightweight SQLite-backed store for paper state management.

Tracks every paper from ingestion → processing → delivery → approval/discard.
No ORM — raw SQL keeps the dependency footprint tiny.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from app.config import settings
from app.logging_cfg import log
from app.schemas import CombinedSummary, PaperMeta, PaperStatus, TextSummary, FigureDescription

DB_PATH = settings.db_dir / "papers.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def _cursor() -> Generator[sqlite3.Cursor, None, None]:
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Create tables if they don't exist."""
    with _cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id    TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                authors     TEXT DEFAULT '[]',
                abstract    TEXT DEFAULT '',
                url         TEXT DEFAULT '',
                pdf_url     TEXT DEFAULT '',
                pdf_path    TEXT DEFAULT '',
                source      TEXT DEFAULT 'arxiv',
                domain      TEXT DEFAULT '',
                published   TEXT,
                fetched_at  TEXT,
                status      TEXT DEFAULT 'queued',
                text_summary TEXT DEFAULT '{}',
                figures     TEXT DEFAULT '[]',
                markdown_row TEXT DEFAULT '',
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_papers_domain ON papers(domain)
        """)
    log.info("paper_db_initialized", path=str(DB_PATH))


def paper_exists(paper_id: str) -> bool:
    with _cursor() as cur:
        cur.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,))
        return cur.fetchone() is not None


def insert_paper(paper: PaperMeta, pdf_path: str = "") -> None:
    """Insert a new paper (skip if already exists)."""
    if paper_exists(paper.paper_id):
        log.debug("paper_already_in_db", paper_id=paper.paper_id)
        return
    with _cursor() as cur:
        cur.execute(
            """INSERT INTO papers
               (paper_id, title, authors, abstract, url, pdf_url, pdf_path,
                source, domain, published, fetched_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                paper.paper_id,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                paper.url,
                paper.pdf_url,
                pdf_path,
                paper.source.value,
                paper.domain,
                paper.published,
                paper.fetched_at,
                PaperStatus.QUEUED.value,
            ),
        )
    log.info("paper_inserted", paper_id=paper.paper_id)


def update_status(paper_id: str, status: PaperStatus) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE papers SET status = ?, updated_at = datetime('now') WHERE paper_id = ?",
            (status.value, paper_id),
        )


def save_text_summary(paper_id: str, summary: TextSummary) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE papers SET text_summary = ?, updated_at = datetime('now') WHERE paper_id = ?",
            (summary.model_dump_json(), paper_id),
        )


def save_figures(paper_id: str, figures: List[FigureDescription]) -> None:
    data = json.dumps([f.model_dump() for f in figures])
    with _cursor() as cur:
        cur.execute(
            "UPDATE papers SET figures = ?, updated_at = datetime('now') WHERE paper_id = ?",
            (data, paper_id),
        )


def save_combined(paper_id: str, combined: CombinedSummary) -> None:
    with _cursor() as cur:
        cur.execute(
            """UPDATE papers
               SET text_summary = ?, figures = ?, markdown_row = ?,
                   status = ?, updated_at = datetime('now')
               WHERE paper_id = ?""",
            (
                combined.text_summary.model_dump_json(),
                json.dumps([f.model_dump() for f in combined.figures]),
                combined.markdown_row,
                PaperStatus.READY.value,
                paper_id,
            ),
        )


def get_ready_papers() -> List[dict]:
    """Get all papers with status='ready' for delivery."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM papers WHERE status = ?", (PaperStatus.READY.value,))
        return [dict(row) for row in cur.fetchall()]


def get_approved_papers_since(since_date: str) -> List[dict]:
    """Get approved papers since a date string (ISO format)."""
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM papers WHERE status = ? AND updated_at >= ?",
            (PaperStatus.APPROVED.value, since_date),
        )
        return [dict(row) for row in cur.fetchall()]


def get_all_approved_papers() -> List[dict]:
    """Get all approved papers."""
    with _cursor() as cur:
        cur.execute("SELECT * FROM papers WHERE status = ?", (PaperStatus.APPROVED.value,))
        return [dict(row) for row in cur.fetchall()]


def get_paper(paper_id: str) -> Optional[dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_queued_papers() -> List[dict]:
    with _cursor() as cur:
        cur.execute("SELECT * FROM papers WHERE status = ?", (PaperStatus.QUEUED.value,))
        return [dict(row) for row in cur.fetchall()]


def row_to_combined(row: dict) -> CombinedSummary:
    """Reconstruct a CombinedSummary from a DB row."""
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
        fetched_at=row.get("fetched_at", ""),
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

    return CombinedSummary(
        paper=paper,
        text_summary=ts,
        figures=figs,
        markdown_row=row.get("markdown_row", ""),
        status=PaperStatus(row.get("status", "ready")),
    )


# Auto-init on import
init_db()
