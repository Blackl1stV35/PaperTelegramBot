"""
ArXiv scraper — queries the ArXiv API for each configured domain.

Respects ArXiv rate-limits with configurable throttle.
Returns a list of PaperMeta objects.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import arxiv

from app.config import settings
from app.logging_cfg import log
from app.schemas import PaperMeta, PaperSource

# Map human-readable domain names → ArXiv search queries.
# These are broad but reasonably targeted.
DOMAIN_QUERIES: Dict[str, str] = {
    "Agentic LLM Research": 'ti:"agentic" OR ti:"LLM agent" OR ti:"tool-use" AND cat:cs.AI',
    "Auxiliary Scientific Research": 'ti:"AI for science" OR ti:"scientific discovery" AND cat:cs.AI',
    "Biology": 'cat:q-bio AND (ti:"machine learning" OR ti:"deep learning" OR ti:"foundation model")',
    "Cosmology": 'cat:astro-ph.CO AND (ti:"neural" OR ti:"deep learning" OR ti:"simulation")',
    "Deep Learning": 'cat:cs.LG AND (ti:"transformer" OR ti:"attention" OR ti:"architecture")',
    "Quantum Research": 'cat:quant-ph AND (ti:"quantum computing" OR ti:"quantum algorithm")',
    "Quantum Physics": 'cat:quant-ph AND (ti:"entanglement" OR ti:"quantum field")',
    "Neuroscience": 'cat:q-bio.NC AND (ti:"neural" OR ti:"brain" OR ti:"cognitive")',
    "Deep Tech": 'ti:"robotics" OR ti:"photonics" OR ti:"metamaterial" AND cat:cs.RO',
}


def _search_domain(domain: str, max_results: int = 3) -> List[PaperMeta]:
    """Run a single ArXiv search for one domain. Blocking call."""
    query = DOMAIN_QUERIES.get(domain, f'ti:"{domain}"')
    client = arxiv.Client(page_size=max_results, delay_seconds=settings.arxiv_throttle_seconds)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[PaperMeta] = []
    try:
        for result in client.results(search):
            paper = PaperMeta(
                paper_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " "),
                authors=[a.name for a in result.authors[:5]],
                abstract=result.summary[:1000],
                url=result.entry_id,
                pdf_url=result.pdf_url or "",
                source=PaperSource.ARXIV,
                domain=domain,
                published=str(result.published.date()) if result.published else None,
            )
            papers.append(paper)
            log.info("arxiv_paper_found", title=paper.title[:80], domain=domain)
    except Exception as exc:
        log.error("arxiv_search_failed", domain=domain, error=str(exc))

    return papers


def scrape_arxiv(domains: List[str] | None = None, per_domain: int = 2) -> List[PaperMeta]:
    """
    Scrape ArXiv for all configured domains.
    Returns at most `per_domain * len(domains)` papers.
    """
    domains = domains or settings.domain_list
    all_papers: List[PaperMeta] = []

    for domain in domains:
        log.info("arxiv_scraping_domain", domain=domain)
        results = _search_domain(domain, max_results=per_domain)
        all_papers.extend(results)
        # Extra throttle between domain queries
        time.sleep(settings.arxiv_throttle_seconds)

    log.info("arxiv_scrape_complete", total=len(all_papers))
    return all_papers
