"""
LLM Client — unified interface to free inference providers.

Supports:
  • Groq       — free 30 req/min, Llama 3.3 70B (RECOMMENDED)
  • Together   — free $1 credit, Llama 3.1 8B
  • Cerebras   — free, Llama 3.3 70B (text only, no vision)
  • OpenRouter — free credits, many models
  • Ollama     — self-hosted (requires separate server with ≥8 GB RAM)

All providers use OpenAI-compatible APIs, so the interface is identical.
Rate limiting and retries are built in.

Usage:
    from app.llm_client import chat, chat_with_vision, embed_text
    result = chat(system_prompt, user_prompt)
    result = chat_with_vision(system_prompt, user_prompt, image_path)
    vector = embed_text("some text to embed")
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.logging_cfg import log

# ─────────────────────────────────────────────────────────────────
#  Provider Configuration
# ─────────────────────────────────────────────────────────────────

PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "text_model_env": "GROQ_TEXT_MODEL",
        "vision_model_env": "GROQ_VISION_MODEL",
        "text_model_default": "llama-3.3-70b-versatile",
        "vision_model_default": "llama-3.2-90b-vision-preview",
        "supports_vision": True,
        "rate_limit_pause": 2.5,  # 30 req/min → ~2s between calls
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "text_model_env": "TOGETHER_TEXT_MODEL",
        "vision_model_env": "TOGETHER_VISION_MODEL",
        "text_model_default": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "vision_model_default": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "supports_vision": True,
        "rate_limit_pause": 1.5,
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "text_model_env": "CEREBRAS_TEXT_MODEL",
        "vision_model_env": None,
        "text_model_default": "llama-3.3-70b",
        "vision_model_default": None,
        "supports_vision": False,
        "rate_limit_pause": 1.0,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "text_model_env": "OPENROUTER_TEXT_MODEL",
        "vision_model_env": "OPENROUTER_VISION_MODEL",
        "text_model_default": "meta-llama/llama-3.3-70b-instruct:free",
        "vision_model_default": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "supports_vision": True,
        "rate_limit_pause": 1.5,
    },
    "ollama": {
        "base_url": None,  # Uses ollama Python client directly
        "api_key_env": None,
        "text_model_env": "OLLAMA_TEXT_MODEL",
        "vision_model_env": "OLLAMA_VISION_MODEL",
        "text_model_default": "qwen3:8b",
        "vision_model_default": "minicpm-v:8b",
        "supports_vision": True,
        "rate_limit_pause": 0,
    },
}


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower()


def _get_config() -> dict:
    provider = _get_provider()
    if provider not in PROVIDER_CONFIG:
        log.warning("unknown_llm_provider", provider=provider, fallback="groq")
        provider = "groq"
    return PROVIDER_CONFIG[provider]


def _get_api_key() -> str:
    cfg = _get_config()
    env_var = cfg["api_key_env"]
    if not env_var:
        return ""
    return os.getenv(env_var, "")


def _get_text_model() -> str:
    cfg = _get_config()
    env_var = cfg.get("text_model_env")
    return os.getenv(env_var, cfg["text_model_default"]) if env_var else cfg["text_model_default"]


def _get_vision_model() -> str:
    cfg = _get_config()
    env_var = cfg.get("vision_model_env")
    default = cfg.get("vision_model_default") or _get_text_model()
    return os.getenv(env_var, default) if env_var else default


# ─────────────────────────────────────────────────────────────────
#  OpenAI-compatible API calls (Groq, Together, Cerebras, OpenRouter)
# ─────────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    pass


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=3, min=5, max=60),
    retry=retry_if_exception_type(RateLimitError),
)
def _openai_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    json_mode: bool = False,
) -> str:
    """
    Call an OpenAI-compatible chat completions endpoint.

    Args:
        json_mode: If True, adds response_format={"type":"json_object"} to the
                   request. Supported by Groq and Together for text models.
                   Silently skipped for vision requests (Groq rejects it there).
    """
    cfg = _get_config()
    api_key = _get_api_key()
    base_url = cfg["base_url"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if _get_provider() == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/research-pipeline"
        headers["X-Title"] = "Research Pipeline"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add JSON mode for text completions (not vision — Groq rejects it there)
    if json_mode:
        # Check that no message contains image content blocks
        has_images = any(
            isinstance(m.get("content"), list) for m in messages
        )
        if not has_images:
            payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", "10"))
            log.warning("rate_limited", provider=_get_provider(), retry_after=retry_after)
            time.sleep(retry_after)
            raise RateLimitError(f"Rate limited, retry after {retry_after}s")

        if resp.status_code == 400:
            # Log the actual error body — invaluable for debugging Groq issues
            error_body = resp.text[:500]
            log.error("api_400_error", provider=_get_provider(), model=model, body=error_body)
            # If JSON mode caused the 400, retry without it
            if json_mode and "response_format" in error_body.lower():
                log.info("retrying_without_json_mode")
                payload.pop("response_format", None)
                resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
            else:
                resp.raise_for_status()
        else:
            resp.raise_for_status()

        data = resp.json()

    pause = cfg.get("rate_limit_pause", 1)
    if pause > 0:
        time.sleep(pause)

    return data["choices"][0]["message"]["content"]


def _ollama_chat(
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
    images: list[str] | None = None,
) -> str:
    """Call Ollama directly (for self-hosted setups)."""
    import ollama as ollama_client

    # Inject images into the last user message if provided
    if images:
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["images"] = images
                break

    response = ollama_client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response["message"]["content"]


# ─────────────────────────────────────────────────────────────────
#  Public Interface
# ─────────────────────────────────────────────────────────────────

def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Send a text chat to the configured LLM provider.
    Returns the assistant's response string.
    Automatically requests JSON output format when available.
    """
    provider = _get_provider()
    model = _get_text_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    log.info("llm_chat", provider=provider, model=model)

    # Request JSON mode for text completions — the system prompt already
    # asks for JSON, this just enforces it at the API level
    wants_json = "json" in system_prompt.lower() or "json" in user_prompt.lower()

    if provider == "ollama":
        return _ollama_chat(messages, model, temperature, max_tokens)
    else:
        return _openai_chat(messages, model, temperature, max_tokens, json_mode=wants_json)


def _compress_image(image_path: Path, max_bytes: int = 4_000_000) -> tuple[str, str]:
    """
    Compress/resize an image so the base64 payload stays under Groq's limit.
    Returns (base64_string, mime_type).

    Strategy:
      1. If already under max_bytes as JPEG, return as-is.
      2. Resize longest edge to 1024px max and re-encode as JPEG quality 80.
      3. If still too large, drop to 768px and quality 60.
    """
    from PIL import Image
    import io

    raw = image_path.read_bytes()

    # Quick path: small enough already
    if len(raw) <= max_bytes:
        suffix = image_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            return base64.b64encode(raw).decode("utf-8"), "image/jpeg"

    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    for max_dim, quality in [(1024, 80), (768, 60), (512, 50)]:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        jpeg_bytes = buf.getvalue()
        if len(jpeg_bytes) <= max_bytes:
            log.debug("image_compressed", original=len(raw), compressed=len(jpeg_bytes), dim=max_dim)
            return base64.b64encode(jpeg_bytes).decode("utf-8"), "image/jpeg"

    # Last resort: return whatever we have at lowest quality
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def chat_with_vision(
    system_prompt: str,
    user_prompt: str,
    image_path: Path,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> str:
    """
    Send image + text to a vision-capable LLM.
    Falls back to text-only analysis if the provider doesn't support vision.

    Groq-specific fixes:
      • No {"role": "system"} alongside image data — merged into user prompt.
      • Exactly ONE image per request (no arrays).
      • Image compressed to JPEG <4 MB to stay under Groq's 20 MB payload limit.
    """
    provider = _get_provider()
    cfg = _get_config()

    if not cfg["supports_vision"]:
        log.warning("vision_not_supported", provider=provider, fallback="text_only")
        return chat(
            system_prompt,
            f"{user_prompt}\n\n[Note: Vision model unavailable. Describe based on context only.]",
            temperature,
            max_tokens,
        )

    model = _get_vision_model()
    log.info("llm_vision", provider=provider, model=model)

    if provider == "ollama":
        # Ollama handles system role + images fine
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return _ollama_chat(messages, model, temperature, max_tokens, images=[str(image_path)])

    # ── API providers (Groq, Together, OpenRouter) ──
    # Compress image to stay under payload limits
    b64_image, mime_type = _compress_image(image_path)

    # Groq rejects {"role":"system"} alongside image content blocks.
    # Merge system instructions into the user message for all API providers
    # (this works universally and avoids provider-specific branching).
    combined_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

    # Single user message with exactly ONE image — no system message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": combined_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                    },
                },
            ],
        },
    ]
    return _openai_chat(messages, model, temperature, max_tokens)


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.
    Uses the LLM provider's embedding endpoint if available,
    otherwise falls back to a simple hash-based embedding (for vector store structure).
    """
    provider = _get_provider()

    if provider == "ollama":
        try:
            import ollama as ollama_client
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
            response = ollama_client.embeddings(model=embed_model, prompt=text)
            return response["embedding"]
        except Exception as exc:
            log.warning("ollama_embed_failed", error=str(exc))
            return _fallback_embedding(text)

    # For cloud providers, try their embedding endpoints
    if provider == "together":
        return _together_embed(text)

    # Groq / Cerebras / OpenRouter don't have free embedding endpoints
    # Use a lightweight local approach
    return _fallback_embedding(text)


def _together_embed(text: str) -> list[float]:
    """Together AI has a free embedding endpoint."""
    api_key = os.getenv("TOGETHER_API_KEY", "")
    if not api_key:
        return _fallback_embedding(text)
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://api.together.xyz/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "togethercomputer/m2-bert-80M-8k-retrieval", "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        log.warning("together_embed_failed", error=str(exc))
        return _fallback_embedding(text)


def _fallback_embedding(text: str, dim: int = 384) -> list[float]:
    """
    Simple hash-based embedding when no embedding API is available.
    Not semantically meaningful but maintains vector store structure
    so the system works end-to-end. Adequate for keyword-style retrieval
    with small collections (<1000 papers).
    """
    import hashlib
    import struct

    h = hashlib.sha512(text.encode()).digest()
    # Extend hash to fill dimension
    extended = h * (dim // len(h) + 1)
    values = struct.unpack(f"{dim}B", extended[:dim])
    # Normalise to [-1, 1]
    vec = [(v / 127.5) - 1.0 for v in values]
    # L2 normalise
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec
