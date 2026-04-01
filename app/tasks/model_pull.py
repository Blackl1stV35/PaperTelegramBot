"""
Model Pull — only relevant when LLM_PROVIDER=ollama.

For API-based providers (groq, together, cerebras, openrouter),
this is a no-op that just verifies API connectivity.
"""

from __future__ import annotations

import os
import sys

import httpx

from app.config import settings
from app.logging_cfg import log, setup_logging


def verify_provider() -> None:
    """Verify the configured LLM provider is reachable."""
    setup_logging()
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        _pull_ollama_models()
        return

    # For API providers, just test connectivity
    test_urls = {
        "groq": "https://api.groq.com/openai/v1/models",
        "together": "https://api.together.xyz/v1/models",
        "cerebras": "https://api.cerebras.ai/v1/models",
        "openrouter": "https://openrouter.ai/api/v1/models",
    }

    url = test_urls.get(provider)
    if not url:
        log.warning("unknown_provider_skip_verify", provider=provider)
        return

    log.info("verifying_provider", provider=provider)
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers={"Authorization": f"Bearer {_get_key(provider)}"})
            if resp.status_code in (200, 401, 403):
                log.info("provider_reachable", provider=provider, status=resp.status_code)
            else:
                log.warning("provider_unexpected_status", provider=provider, status=resp.status_code)
    except Exception as exc:
        log.error("provider_unreachable", provider=provider, error=str(exc))


def _get_key(provider: str) -> str:
    key_map = {
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    return os.getenv(key_map.get(provider, ""), "")


def _pull_ollama_models() -> None:
    """Pull Ollama models (only when LLM_PROVIDER=ollama)."""
    try:
        import ollama as ollama_client
    except ImportError:
        log.error("ollama_package_not_installed",
                  hint="Run: pip install ollama==0.4.7")
        return

    models = [
        settings.ollama_text_model,
        settings.ollama_vision_model,
        settings.ollama_embed_model,
    ]

    for model_name in models:
        log.info("pulling_model", model=model_name)
        try:
            ollama_client.pull(model_name)
            log.info("model_pulled", model=model_name)
        except Exception as exc:
            log.error("model_pull_failed", model=model_name, error=str(exc))


if __name__ == "__main__":
    verify_provider()
