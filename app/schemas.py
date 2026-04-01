"""
Pydantic schemas — single source of truth for all data shapes.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PaperSource(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class PaperStatus(str, Enum):
    QUEUED = "queued"
    TEXT_PROCESSING = "text_processing"
    FIGURE_PROCESSING = "figure_processing"
    COMBINING = "combining"
    READY = "ready"
    DELIVERED = "delivered"
    APPROVED = "approved"
    DISCARDED = "discarded"
    ERROR = "error"


class PaperMeta(BaseModel):
    """Metadata scraped from ArXiv / Semantic Scholar."""

    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    source: PaperSource = PaperSource.ARXIV
    domain: str = ""
    published: Optional[str] = None
    fetched_at: str = Field(
        default_factory=lambda: dt.datetime.utcnow().isoformat()
    )


class TextSummary(BaseModel):
    """Structured extraction from the text pipeline."""

    what: str = Field(description="What the paper contributes")
    how: str = Field(description="Methodology / approach")
    to_whom: str = Field(description="Who benefits from this research")
    domain: str = ""
    confidence: int = Field(default=70, ge=0, le=100)


class FigureDescription(BaseModel):
    """A single extracted figure with its LLM-generated description."""

    figure_index: int
    image_path: str = ""
    original_caption: str = ""
    llm_description: str = ""
    relevance: str = ""


class CombinedSummary(BaseModel):
    """Merged text + figure summary — one per paper."""

    paper: PaperMeta
    text_summary: TextSummary
    figures: List[FigureDescription] = Field(default_factory=list)
    markdown_row: str = ""
    status: PaperStatus = PaperStatus.READY


class ApprovalAction(BaseModel):
    paper_id: str
    action: str  # "approve" or "discard"
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.utcnow().isoformat()
    )


class WeeklySynthesis(BaseModel):
    """Output of the weekly cross-domain meta-synthesis."""

    week_start: str
    week_end: str
    domain_count: int
    paper_count: int
    macro_trends: str
    cross_domain_links: str
    report_markdown: str
