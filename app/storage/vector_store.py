"""
Vector Store — ChromaDB + API-based embeddings.

REVISED: Uses app.llm_client.embed_text() instead of local Ollama embeddings.
Works on 1 GB RAM VMs — ChromaDB uses DuckDB+Parquet for persistence.
"""

from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.llm_client import embed_text
from app.logging_cfg import log
from app.schemas import CombinedSummary


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection with persistent storage."""
    client = chromadb.Client(
        ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=settings.chroma_persist_dir,
            anonymized_telemetry=False,
        )
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _build_document(summary: CombinedSummary) -> str:
    """Build a single text document for embedding."""
    parts = [
        f"Title: {summary.paper.title}",
        f"Domain: {summary.paper.domain}",
        f"What: {summary.text_summary.what}",
        f"How: {summary.text_summary.how}",
        f"To Whom: {summary.text_summary.to_whom}",
    ]
    for fig in summary.figures[:3]:
        if fig.llm_description:
            parts.append(f"Figure: {fig.llm_description}")
    return "\n".join(parts)


def add_paper(summary: CombinedSummary) -> bool:
    """Embed and store a single paper in the vector store."""
    doc_text = _build_document(summary)
    embedding = embed_text(doc_text)

    if not embedding:
        log.warning("skipping_vector_store_no_embedding", paper_id=summary.paper.paper_id)
        return False

    try:
        collection = _get_collection()
        collection.upsert(
            ids=[summary.paper.paper_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "title": summary.paper.title,
                "domain": summary.paper.domain,
                "what": summary.text_summary.what[:200],
                "confidence": summary.text_summary.confidence,
                "url": summary.paper.url,
            }],
        )
        log.info("paper_added_to_vector_store", paper_id=summary.paper.paper_id)
        return True
    except Exception as exc:
        log.error("vector_store_add_failed", paper_id=summary.paper.paper_id, error=str(exc))
        return False


def query_similar(query: str, n_results: int = 10, domain: Optional[str] = None) -> List[dict]:
    """Retrieve papers similar to a query string."""
    embedding = embed_text(query)
    if not embedding:
        return []

    try:
        collection = _get_collection()
        where_filter = {"domain": domain} if domain else None
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        papers = []
        for i, doc_id in enumerate(results["ids"][0]):
            papers.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return papers
    except Exception as exc:
        log.error("vector_query_failed", error=str(exc))
        return []


def get_all_documents() -> List[dict]:
    """Retrieve all stored documents (for meta-synthesis)."""
    try:
        collection = _get_collection()
        results = collection.get(include=["documents", "metadatas"])
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            docs.append({
                "id": doc_id,
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return docs
    except Exception as exc:
        log.error("vector_get_all_failed", error=str(exc))
        return []
