"""
Daily Ingestion Job — the entry point for the daily pipeline.

Flow:
  1. Scrape ArXiv + Semantic Scholar for all 9 domains.
  2. Deduplicate against existing DB entries.
  3. Download PDFs (throttled).
  4. Enqueue text + figure processing for each paper.
  5. Cap at MAX_PAPERS_PER_DAY to stay within free-tier resources.
"""

from __future__ import annotations

from app.config import settings
from app.logging_cfg import log
from app.ingestion.arxiv_scraper import scrape_arxiv
from app.ingestion.semantic_scholar import scrape_semantic_scholar
from app.ingestion.downloader import download_pdfs
from app.schemas import PaperMeta
from app.storage.paper_db import insert_paper, paper_exists
from app.tasks.worker import enqueue_paper_processing


def run_daily_ingestion() -> int:
    """
    Execute the full daily ingestion pipeline.
    Returns the number of papers enqueued for processing.
    """
    log.info("daily_ingestion_start", max_papers=settings.max_papers_per_day)

    # ── Step 1: Scrape ──
    # Split budget: ~1 paper per domain from ArXiv, fill remainder from S2
    per_domain_arxiv = max(1, settings.max_papers_per_day // len(settings.domain_list))
    arxiv_papers = scrape_arxiv(per_domain=per_domain_arxiv)
    s2_papers = scrape_semantic_scholar(per_domain=1)

    all_papers = arxiv_papers + s2_papers
    log.info("scrape_results", arxiv=len(arxiv_papers), s2=len(s2_papers))

    # ── Step 2: Deduplicate ──
    new_papers: list[PaperMeta] = []
    seen_titles: set[str] = set()

    for paper in all_papers:
        if len(new_papers) >= settings.max_papers_per_day:
            break
        # Skip if already in DB
        if paper_exists(paper.paper_id):
            continue
        # Skip duplicate titles (across sources)
        title_key = paper.title.lower().strip()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        new_papers.append(paper)

    log.info("after_dedup", new_papers=len(new_papers))

    if not new_papers:
        log.info("no_new_papers_today")
        return 0

    # ── Step 3: Download PDFs ──
    downloaded = download_pdfs(new_papers)
    log.info("pdfs_downloaded", count=len(downloaded))

    # ── Step 4: Insert into DB + enqueue processing ──
    enqueued = 0
    for paper, pdf_path in downloaded:
        insert_paper(paper, pdf_path=str(pdf_path))
        enqueue_paper_processing(paper.paper_id, str(pdf_path))
        enqueued += 1

    log.info("daily_ingestion_complete", enqueued=enqueued)
    return enqueued
