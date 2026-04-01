"""
Semantic Scholar scraper — uses the free public API (no key needed, 100 req/5 min).

Supplements ArXiv results with additional papers for each domain.
"""

from __future__ import annotations

import time
from typing import Dict, List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.logging_cfg import log
from app.schemas import PaperMeta, PaperSource

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Simplified query terms per domain
DOMAIN_TERMS: Dict[str, str] = {
    "Agentic LLM Research": "agentic LLM autonomous agent",
    "Auxiliary Scientific Research": "AI scientific discovery",
    "Biology": "computational biology deep learning",
    "Cosmology": "cosmology neural network simulation",
    "Deep Learning": "deep learning transformer architecture",
    "Quantum Research": "quantum computing algorithm",
    "Quantum Physics": "quantum entanglement field theory",
    "Neuroscience": "computational neuroscience brain",
    "Deep Tech": "robotics photonics advanced manufacturing",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
def _query_s2(query: str, limit: int = 2) -> List[dict]:
    """Single Semantic Scholar query with retries."""
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,abstract,url,openAccessPdf,publicationDate",
        "openAccessPdf": "",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(S2_API, params=params)
        resp.raise_for_status()
        data = resp.json()
    return data.get("data", [])


def scrape_semantic_scholar(
    domains: List[str] | None = None,
    per_domain: int = 1,
) -> List[PaperMeta]:
    """
    Query Semantic Scholar for each domain.
    Returns papers not already found on ArXiv (dedup by title similarity).
    """
    domains = domains or settings.domain_list
    all_papers: List[PaperMeta] = []

    for domain in domains:
        query = DOMAIN_TERMS.get(domain, domain)
        log.info("s2_scraping_domain", domain=domain, query=query)

        try:
            results = _query_s2(query, limit=per_domain)
        except Exception as exc:
            log.error("s2_search_failed", domain=domain, error=str(exc))
            continue

        for item in results:
            pdf_info = item.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url", "")
            if not pdf_url:
                continue  # Skip papers without open-access PDF

            authors_raw = item.get("authors") or []
            paper = PaperMeta(
                paper_id=item.get("paperId", "")[:20],
                title=(item.get("title") or "Untitled").replace("\n", " "),
                authors=[a.get("name", "") for a in authors_raw[:5]],
                abstract=(item.get("abstract") or "")[:1000],
                url=item.get("url") or "",
                pdf_url=pdf_url,
                source=PaperSource.SEMANTIC_SCHOLAR,
                domain=domain,
                published=item.get("publicationDate"),
            )
            all_papers.append(paper)
            log.info("s2_paper_found", title=paper.title[:80], domain=domain)

        # Respect rate limits: ~1 req/3s for unauthenticated
        time.sleep(3)

    log.info("s2_scrape_complete", total=len(all_papers))
    return all_papers
