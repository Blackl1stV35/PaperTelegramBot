"""
Figure Pipeline — extract figures from PDF, analyse with vision LLM.

REVISED: Uses app.llm_client.chat_with_vision() which routes to
Groq/Together/OpenRouter vision models via API, no local GPU needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from app.config import settings
from app.llm_client import chat_with_vision
from app.logging_cfg import log
from app.processing.prompts import (
    FIGURE_DESCRIPTION_FEWSHOT,
    FIGURE_DESCRIPTION_SYSTEM,
    FIGURE_DESCRIPTION_USER,
)
from app.schemas import FigureDescription, PaperMeta

MAX_FIGURES = 5
MIN_IMAGE_SIZE = 15_000  # bytes
FIGURE_DPI = 150


def extract_figures_from_pdf(pdf_path: Path, paper_id: str) -> List[Path]:
    """Extract figure images from a PDF (PyMuPDF, no external deps)."""
    figure_dir = settings.figure_dir / paper_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        log.error("figure_pdf_open_failed", path=str(pdf_path), error=str(exc))
        return saved

    # Strategy A: extract embedded images
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            if len(saved) >= MAX_FIGURES:
                break
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                raw = pix.tobytes("png")
                if len(raw) < MIN_IMAGE_SIZE:
                    continue
                out_path = figure_dir / f"fig_p{page_num}_i{img_idx}.png"
                out_path.write_bytes(raw)
                saved.append(out_path)
            except Exception:
                continue

    # Strategy B: rasterise figure-heavy pages
    raster_count = 0
    for page_num, page in enumerate(doc):
        if raster_count >= 2 or len(saved) >= MAX_FIGURES:
            break
        text_blocks = page.get_text("blocks")
        image_list = page.get_images(full=True)
        if len(image_list) >= 1 and len(text_blocks) < 8:
            try:
                pix = page.get_pixmap(dpi=FIGURE_DPI)
                out_path = figure_dir / f"page_{page_num}_raster.png"
                pix.save(str(out_path))
                saved.append(out_path)
                raster_count += 1
            except Exception:
                continue

    doc.close()
    log.info("figures_extracted", paper_id=paper_id, count=len(saved))
    return saved[:MAX_FIGURES]


def _parse_figure_json(raw: str) -> dict:
    """Parse JSON from vision model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"description": text[:200], "relevance": "Could not parse structured output."}


def analyse_figure(
    image_path: Path,
    paper: PaperMeta,
    figure_index: int,
) -> FigureDescription:
    """Analyse a single figure image with the vision LLM."""
    log.info("figure_analysis_start", paper_id=paper.paper_id, fig=figure_index)

    user_prompt = FIGURE_DESCRIPTION_FEWSHOT + "\n\n" + FIGURE_DESCRIPTION_USER.format(
        title=paper.title,
        domain=paper.domain,
    )

    try:
        raw_answer = chat_with_vision(
            system_prompt=FIGURE_DESCRIPTION_SYSTEM,
            user_prompt=user_prompt,
            image_path=image_path,
            temperature=0.1,
            max_tokens=256,
        )
        parsed = _parse_figure_json(raw_answer)

        return FigureDescription(
            figure_index=figure_index,
            image_path=str(image_path),
            original_caption="",
            llm_description=parsed.get("description", "No description generated."),
            relevance=parsed.get("relevance", ""),
        )

    except Exception as exc:
        log.error("figure_llm_failed", paper_id=paper.paper_id, fig=figure_index, error=str(exc))
        return FigureDescription(
            figure_index=figure_index,
            image_path=str(image_path),
            llm_description="Vision model analysis failed.",
            relevance="N/A",
        )


def analyse_all_figures(paper: PaperMeta, pdf_path: Path) -> List[FigureDescription]:
    """Full figure pipeline: extract → analyse each."""
    figure_paths = extract_figures_from_pdf(pdf_path, paper.paper_id)
    descriptions: List[FigureDescription] = []

    for idx, fig_path in enumerate(figure_paths):
        desc = analyse_figure(fig_path, paper, figure_index=idx)
        descriptions.append(desc)

    log.info("figure_pipeline_complete", paper_id=paper.paper_id, figures=len(descriptions))
    return descriptions
