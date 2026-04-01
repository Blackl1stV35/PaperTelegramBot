"""
Google Sheets storage — saves approved papers to a shared spreadsheet.

Setup:
  1. Create a Google Cloud project (free tier).
  2. Enable Google Sheets API.
  3. Create a Service Account and download the JSON credentials.
  4. Share the target spreadsheet with the service account email.
  5. Set GOOGLE_SHEETS_ENABLED=true and point GOOGLE_SHEETS_CREDENTIALS_FILE
     to the JSON file in your .env.
"""

from __future__ import annotations

import json
from typing import List

from app.config import settings
from app.logging_cfg import log
from app.schemas import CombinedSummary


def _get_client():
    """Lazy import and auth — only called when sheets are enabled."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(
        settings.google_sheets_credentials_file, scopes=scopes
    )
    return gspread.authorize(creds)


def _ensure_header(worksheet) -> None:
    """Add header row if sheet is empty."""
    if worksheet.row_count == 0 or not worksheet.cell(1, 1).value:
        headers = [
            "Paper ID", "Title", "Domain", "Authors", "Published",
            "What", "How", "To Whom", "Confidence",
            "Figure Count", "Figure Insights", "URL", "Approved At",
        ]
        worksheet.append_row(headers)


def save_to_sheets(summary: CombinedSummary) -> bool:
    """Append a single approved paper to Google Sheets."""
    if not settings.google_sheets_enabled:
        log.debug("google_sheets_disabled")
        return False

    try:
        gc = _get_client()
        sh = gc.open(settings.google_sheets_spreadsheet_name)
        worksheet = sh.sheet1
        _ensure_header(worksheet)

        fig_insights = " | ".join(
            f"Fig{f.figure_index + 1}: {f.llm_description[:80]}"
            for f in summary.figures
        )

        row = [
            summary.paper.paper_id,
            summary.paper.title,
            summary.paper.domain,
            ", ".join(summary.paper.authors[:3]),
            summary.paper.published or "",
            summary.text_summary.what,
            summary.text_summary.how,
            summary.text_summary.to_whom,
            summary.text_summary.confidence,
            len(summary.figures),
            fig_insights,
            summary.paper.url,
            summary.paper.fetched_at,
        ]
        worksheet.append_row(row, value_input_option="USER_ENTERED")
        log.info("saved_to_sheets", paper_id=summary.paper.paper_id)
        return True

    except Exception as exc:
        log.error("sheets_save_failed", paper_id=summary.paper.paper_id, error=str(exc))
        return False


def save_batch_to_sheets(summaries: List[CombinedSummary]) -> int:
    """Save multiple papers. Returns count of successful saves."""
    saved = 0
    for s in summaries:
        if save_to_sheets(s):
            saved += 1
    return saved
