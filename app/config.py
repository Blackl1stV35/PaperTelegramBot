"""
Centralised configuration — loaded once from .env / environment variables.
All downstream modules import `settings` from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service URLs ──
    redis_url: str = "redis://redis:6379/0"

    # ── LLM Provider ──
    llm_provider: str = "groq"  # groq | together | cerebras | openrouter | ollama

    # Ollama (only used if llm_provider=ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_text_model: str = "qwen3:8b"
    ollama_vision_model: str = "minicpm-v:8b"
    ollama_embed_model: str = "nomic-embed-text:latest"

    # Embedding provider: "api" (uses LLM provider) or "ollama"
    embedding_provider: str = "api"

    # ── Domains ──
    research_domains: str = (
        "Agentic LLM Research,Auxiliary Scientific Research,Biology,"
        "Cosmology,Deep Learning,Quantum Research,Quantum Physics,"
        "Neuroscience,Deep Tech"
    )

    @property
    def domain_list(self) -> List[str]:
        return [d.strip() for d in self.research_domains.split(",") if d.strip()]

    # ── Ingestion ──
    max_papers_per_day: int = 6
    arxiv_throttle_seconds: int = 5
    pdf_download_timeout: int = 120

    # ── Telegram ──
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Google Sheets ──
    google_sheets_enabled: bool = False
    google_sheets_credentials_file: str = "/app/data/google_credentials.json"
    google_sheets_spreadsheet_name: str = "Research Pipeline"

    # ── Notion ──
    notion_enabled: bool = False
    notion_token: str = ""
    notion_database_id: str = ""

    # ── Vector Store ──
    chroma_persist_dir: str = "/app/data/db/chroma"
    chroma_collection: str = "research_papers"

    # ── Scheduling ──
    daily_ingest_hour: int = 6
    daily_ingest_minute: int = 0
    weekly_synthesis_day: str = "sunday"
    weekly_synthesis_hour: int = 18

    # ── Paths ──
    data_dir: Path = Path("/app/data")

    @property
    def pdf_dir(self) -> Path:
        p = self.data_dir / "pdfs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figure_dir(self) -> Path:
        p = self.data_dir / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_dir(self) -> Path:
        p = self.data_dir / "db"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Logging ──
    log_level: str = "INFO"


settings = Settings()
