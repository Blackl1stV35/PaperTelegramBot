"""
PDF downloader — fetches full PDFs for scraped papers.

Features:
  • Throttled downloads (ArXiv requires ≥3 s between requests).
  • Retries with exponential back-off.
  • Deduplication: skips if PDF already exists on disk.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_cfg import log
from app.schemas import PaperMeta


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=3, max=60))
def _download_one(url: str, dest: Path) -> bool:
    """Download a single PDF. Returns True on success."""
    headers = {"User-Agent": "ResearchPipeline/1.0 (mailto:research@example.com)"}
    with httpx.Client(timeout=settings.pdf_download_timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        if len(resp.content) < 5000:
            log.warning("pdf_too_small", url=url, size=len(resp.content))
            return False
        dest.write_bytes(resp.content)
    return True


def download_pdfs(papers: List[PaperMeta]) -> List[Tuple[PaperMeta, Path]]:
    """
    Download PDFs for a list of papers.
    Returns list of (paper, local_path) tuples for successful downloads.
    """
    downloaded: List[Tuple[PaperMeta, Path]] = []

    for paper in papers:
        if not paper.pdf_url:
            log.warning("no_pdf_url", paper_id=paper.paper_id)
            continue

        # Sanitise filename
        safe_name = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in paper.paper_id
        )
        dest = settings.pdf_dir / f"{safe_name}.pdf"

        if dest.exists() and dest.stat().st_size > 5000:
            log.info("pdf_already_exists", paper_id=paper.paper_id)
            downloaded.append((paper, dest))
            continue

        log.info("downloading_pdf", paper_id=paper.paper_id, url=paper.pdf_url[:80])
        try:
            success = _download_one(paper.pdf_url, dest)
            if success:
                downloaded.append((paper, dest))
                log.info("pdf_downloaded", paper_id=paper.paper_id, size=dest.stat().st_size)
            else:
                log.warning("pdf_download_empty", paper_id=paper.paper_id)
        except Exception as exc:
            log.error("pdf_download_failed", paper_id=paper.paper_id, error=str(exc))

        # Throttle — especially important for ArXiv
        time.sleep(settings.arxiv_throttle_seconds)

    log.info("pdf_download_batch_complete", downloaded=len(downloaded), total=len(papers))
    return downloaded
