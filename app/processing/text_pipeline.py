"""
Text Pipeline — extract text from PDF, then run structured LLM prompt.

REVISED: Uses app.llm_client (Groq/Together/Cerebras/OpenRouter/Ollama)
instead of direct Ollama calls. Works on 1 GB RAM VMs.

Groq-specific fixes:
  • Sends only Abstract + Introduction + Conclusion (~4-6k tokens) instead
    of the full paper, staying well within Groq free-tier TPM limits.
  • Requests response_format=json_object when supported.
  • Robust fallback parsing: JSON → brace extraction → regex field extraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.config import settings
from app.llm_client import chat, _get_provider
from app.logging_cfg import log
from app.processing.prompts import (
    TEXT_SUMMARY_FEWSHOT,
    TEXT_SUMMARY_SYSTEM,
    TEXT_SUMMARY_USER_TEMPLATE,
)
from app.schemas import PaperMeta, TextSummary

# Budget per section (chars). Total ≈ 16k chars → ~4k tokens.
# This keeps us safely under Groq's 6k TPM free-tier burst limit
# while sending the most information-dense parts of the paper.
ABSTRACT_BUDGET = 3_000
INTRO_BUDGET = 6_000
CONCLUSION_BUDGET = 5_000
FALLBACK_BUDGET = 16_000  # If section splitting fails, head+tail


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages_text).strip()
    except Exception as exc:
        log.error("text_extraction_failed", path=str(pdf_path), error=str(exc))
        return ""


def _extract_key_sections(full_text: str) -> str:
    """
    Pull out Abstract, Introduction, and Conclusion/Discussion sections.
    Falls back to head+tail truncation if section headers aren't found.

    This is critical for Groq free tier — sending 30 pages of dense text
    exceeds TPM limits and causes 400 errors. The key sections contain
    ~90% of the information needed for What/How/To Whom extraction.
    """
    text_lower = full_text.lower()
    sections: list[str] = []

    # ── Abstract ──
    abs_patterns = [
        (r'abstract\s*\n', r'\n\s*(?:1[\.\s]|introduction|keywords)'),
        (r'abstract[\.\s\-—]', r'\n\s*(?:1[\.\s]|introduction)'),
    ]
    abstract = _find_section(full_text, text_lower, abs_patterns, ABSTRACT_BUDGET)
    if abstract:
        sections.append(f"[ABSTRACT]\n{abstract}")

    # ── Introduction ──
    intro_patterns = [
        (r'(?:1[\.\s]+)?introduction\s*\n', r'\n\s*(?:2[\.\s]|related work|background|method|preliminar)'),
        (r'\n1[\.\s]+\w', r'\n2[\.\s]+\w'),
    ]
    intro = _find_section(full_text, text_lower, intro_patterns, INTRO_BUDGET)
    if intro:
        sections.append(f"[INTRODUCTION]\n{intro}")

    # ── Conclusion / Discussion ──
    concl_patterns = [
        (r'(?:\d[\.\s]+)?(?:conclusion|concluding|discussion|summary and)\s*\n?', r'\n\s*(?:references|bibliography|acknowledge|appendix)'),
        (r'(?:conclusion|discussion)\s', r'(?:references|bibliography)'),
    ]
    conclusion = _find_section(full_text, text_lower, concl_patterns, CONCLUSION_BUDGET)
    if conclusion:
        sections.append(f"[CONCLUSION]\n{conclusion}")

    # If we found at least 2 sections, use them
    if len(sections) >= 2:
        result = "\n\n".join(sections)
        log.info("section_extraction_success", sections=len(sections), chars=len(result))
        return result

    # Fallback: head + tail (first 60%, last 40%)
    log.info("section_extraction_fallback", reason="fewer_than_2_sections_found")
    return _head_tail_truncate(full_text, FALLBACK_BUDGET)


def _find_section(
    full_text: str,
    text_lower: str,
    patterns: list[tuple[str, str]],
    max_chars: int,
) -> str:
    """Try multiple regex patterns to find a section. Returns trimmed text or ''."""
    for start_pat, end_pat in patterns:
        start_match = re.search(start_pat, text_lower)
        if not start_match:
            continue
        start_idx = start_match.end()
        end_match = re.search(end_pat, text_lower[start_idx:])
        if end_match:
            end_idx = start_idx + end_match.start()
        else:
            end_idx = min(start_idx + max_chars, len(full_text))
        section = full_text[start_idx:end_idx].strip()
        if len(section) > 100:  # Sanity check: section should be meaningful
            return section[:max_chars]
    return ""


def _head_tail_truncate(text: str, max_chars: int) -> str:
    """Keep first 60% and last 40% of the budget."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n\n[...middle of paper truncated...]\n\n" + text[-tail:]


def _parse_json_response(raw: str) -> dict:
    """
    Robustly extract the summary JSON from LLM output.

    Three-stage fallback:
      1. Direct JSON parse (ideal case).
      2. Find first { ... } block and parse that.
      3. Regex extraction of individual fields (last resort).
    """
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        # Remove all fence markers
        text = re.sub(r'```(?:json)?', '', text).strip()

    # Stage 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 2: find outermost { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Stage 3: regex field extraction — handles cases where the model
    # wraps JSON in conversational text or produces malformed JSON
    log.warning("json_parse_falling_back_to_regex", raw_preview=text[:200])
    result = {}
    for field in ("what", "how", "to_whom", "domain"):
        # Match "field": "value" or "field": "value with \"escapes\""
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.){{10,}})"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[field] = match.group(1).replace('\\"', '"').replace('\\n', ' ').strip()

    # Confidence is an int
    conf_match = re.search(r'"confidence"\s*:\s*(\d+)', text)
    if conf_match:
        result["confidence"] = int(conf_match.group(1))

    if result:
        log.info("regex_extraction_recovered", fields=list(result.keys()))

    return result


def analyse_text(paper: PaperMeta, pdf_path: Path) -> TextSummary:
    """
    Full text pipeline: extract → smart section selection → LLM → parse.

    Sends only Abstract + Intro + Conclusion to avoid exceeding Groq TPM limits.
    """
    log.info("text_pipeline_start", paper_id=paper.paper_id)

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        log.warning("empty_text_extraction", paper_id=paper.paper_id)
        return TextSummary(
            what=f"[Extraction failed] {paper.title}",
            how="Text could not be extracted from PDF.",
            to_whom="Unknown",
            domain=paper.domain,
            confidence=0,
        )

    key_sections = _extract_key_sections(raw_text)

    user_prompt = TEXT_SUMMARY_FEWSHOT + "\n\n" + TEXT_SUMMARY_USER_TEMPLATE.format(
        domain=paper.domain,
        title=paper.title,
        text=key_sections,
    )

    try:
        raw_answer = chat(
            system_prompt=TEXT_SUMMARY_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=512,
        )
        parsed = _parse_json_response(raw_answer)

        # Build summary with safe defaults for any missing fields
        summary = TextSummary(
            what=parsed.get("what") or paper.title,
            how=parsed.get("how") or "See paper for methodology details.",
            to_whom=parsed.get("to_whom") or "Researchers in this field",
            domain=parsed.get("domain") or paper.domain,
            confidence=int(parsed.get("confidence", 50)),
        )
        log.info("text_pipeline_complete", paper_id=paper.paper_id, confidence=summary.confidence)
        return summary

    except Exception as exc:
        log.error("text_llm_failed", paper_id=paper.paper_id, error=str(exc))
        # Fallback: use abstract directly rather than empty error text
        abstract_preview = paper.abstract[:300] if paper.abstract else "See paper."
        return TextSummary(
            what=paper.title,
            how=f"LLM analysis failed. Abstract: {abstract_preview}",
            to_whom="Researchers",
            domain=paper.domain,
            confidence=10,
        )
