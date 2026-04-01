###############################################################################
# Dockerfile — Research Summarization Pipeline
# Targets: GCP e2-micro / AWS t2.micro / Azure B1s (1 GB RAM, 30 GB disk)
# Image size: ~450 MB (no torch, no sentence-transformers)
###############################################################################
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Minimal system deps for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY scripts/ /app/scripts/

RUN mkdir -p /app/data/pdfs /app/data/figures /app/data/db

EXPOSE 8000
