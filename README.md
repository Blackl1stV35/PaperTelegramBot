# 🔬 Autonomous Research Summarization Pipeline

> **1-click `docker compose up` system** that scrapes ArXiv + Semantic Scholar across 9 scientific domains, extracts text & figures via separate pipelines, produces structured "What / How / To Whom" summaries with figure analysis, and delivers daily Telegram digests with 👍/👎 approval. Approved papers feed into Google Sheets + a ChromaDB vector store. Weekly cross-domain meta-synthesis runs automatically.

**Runs on any free-tier cloud VM** — GCP, AWS, Azure, or any 1 GB+ VPS.

---

## Architecture

```
┌─────────────┐  scrape   ┌──────────┐  download  ┌──────────┐
│  Scheduler   │──────────▶│  ArXiv   │───────────▶│  PDFs    │
│  (daily cron)│           │  + S2    │            │  on disk │
└──────┬──────┘           └──────────┘            └─────┬────┘
       │                                                │
       ▼                                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    RQ Task Queue (Redis)                      │
│  ┌─────────────────┐          ┌─────────────────────┐        │
│  │  Text Pipeline   │          │  Figure Pipeline     │        │
│  │  PyMuPDF → text  │          │  PyMuPDF → images    │        │
│  │  Groq API → JSON │          │  Groq Vision → JSON  │        │
│  └────────┬────────┘          └──────────┬──────────┘        │
│           └────────────┬─────────────────┘                   │
│                        ▼                                      │
│              ┌─────────────────┐                              │
│              │  Combine Task    │                              │
│              │  → Markdown Card │                              │
│              └────────┬────────┘                              │
└───────────────────────┼──────────────────────────────────────┘
                        ▼
              ┌─────────────────┐     👍 ──▶ Google Sheets
              │  Telegram Bot    │──────────▶ ChromaDB
              │  daily digest    │     👎 ──▶ Discard
              └────────┬────────┘
                       │ weekly
                       ▼
              ┌─────────────────┐
              │  Meta-Synthesis  │
              │  RAG + LLM API  │
              └─────────────────┘
```

## LLM Strategy: Free API Inference

Since no free-tier VM has enough RAM for local LLM inference, this pipeline
offloads all AI work to **free cloud inference APIs**:

| Provider | Free Tier | Best Model Available | Vision | Sign-up |
|----------|-----------|---------------------|--------|---------|
| **Groq** (recommended) | 30 req/min, unlimited | Llama 3.3 70B | ✅ Llama 3.2 90B Vision | [console.groq.com](https://console.groq.com) |
| **Together** | $1 free credit (~500 calls) | Llama 3.1 8B Turbo | ✅ Llama 3.2 11B Vision | [api.together.xyz](https://api.together.xyz) |
| **Cerebras** | Free tier, fast | Llama 3.3 70B | ❌ | [cloud.cerebras.ai](https://cloud.cerebras.ai) |
| **OpenRouter** | Free credits on signup | Many free models | ✅ | [openrouter.ai](https://openrouter.ai) |

> **Groq is the default** — no credit card required, fastest inference, and you get Llama 3.3 70B which is *better* than any model that fits in 24 GB locally.

---

## Cloud Provider Comparison

| Provider | Tier | Specs | Duration | Credit Card |
|----------|------|-------|----------|-------------|
| **GCP** | Always Free | e2-micro: 2 vCPU (shared), 1 GB RAM, 30 GB disk | Forever | Required |
| **AWS** | Free Tier | t2.micro: 1 vCPU, 1 GB RAM, 30 GB EBS | 12 months | Required |
| **Azure** | Free Tier | B1s: 1 vCPU, 1 GB RAM, 64 GB disk | 12 months | Required |
| Any cheap VPS | — | 1+ GB RAM | Varies | Varies |

All three work. GCP is the only true "always free" option.

---

## Quick Start

### Prerequisites

- Docker 24+ with `docker compose` v2
- A free LLM API key (Groq recommended)
- A Telegram bot token (from [@BotFather](https://t.me/BotFather))

### 1. Get Your Free API Keys

```bash
# Groq (2 minutes, no credit card):
# 1. Go to https://console.groq.com
# 2. Sign in with Google/GitHub
# 3. Create API Key → copy it

# Telegram Bot:
# 1. Message @BotFather on Telegram
# 2. /newbot → follow prompts → copy token
# 3. Send /start to your new bot
# 4. Get your chat ID:
curl "https://api.telegram.org/bot<TOKEN>/getUpdates" | python3 -m json.tool
# Look for "chat":{"id": YOUR_CHAT_ID}
```

### 2. Deploy on GCP Always Free

```bash
# Create e2-micro instance in GCP Console:
#   Compute Engine → Create Instance
#   Machine type: e2-micro (2 vCPU, 1 GB)
#   Boot disk: Ubuntu 24.04, 30 GB Standard persistent disk
#   Region: us-west1, us-central1, or us-east1 (free tier eligible)
#   Firewall: Allow HTTP + HTTPS

# SSH in and install Docker
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER && newgrp docker

# Clone and configure
git clone https://github.com/YOUR_USERNAME/research-pipeline.git
cd research-pipeline
cp .env.example .env
nano .env  # Set GROQ_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Launch (takes ~2 min to build)
docker compose up -d
```

### 3. Deploy on AWS Free Tier

```bash
# Launch t2.micro EC2 instance:
#   AMI: Ubuntu 24.04 LTS
#   Instance type: t2.micro
#   Storage: 30 GB gp3
#   Security group: open ports 22, 8000

# SSH in
ssh -i your-key.pem ubuntu@<ip>

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# Same clone + configure + launch as GCP above
```

### 4. Deploy on Azure Free Tier

```bash
# Create B1s VM:
#   Azure Portal → Virtual Machines → Create
#   Size: Standard_B1s (1 vCPU, 1 GB)
#   Image: Ubuntu 24.04
#   Disk: 64 GB Standard SSD

# SSH in, install Docker, same as above
```

### 5. Verify

```bash
docker compose ps            # All services healthy?
curl http://localhost:8000/health   # {"status":"ok"}
curl http://localhost:8000/status   # Pipeline stats

# Trigger first ingestion manually
curl -X POST http://localhost:8000/trigger/ingest

# Or from Telegram: send /ingest to your bot
```

---

## Memory Budget (1 GB RAM)

| Component | Memory |
|-----------|--------|
| OS + Docker overhead | ~300 MB |
| Redis (capped) | 64 MB |
| API + Telegram bot + scheduler | 250 MB |
| RQ Worker | 250 MB |
| RQ Dashboard | 80 MB |
| **Total** | **~950 MB** |

PDF processing is the only memory spike — PyMuPDF processes one paper at a time
and peaks at ~100 MB per paper. The worker memory limit handles this.

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| `api` | 8000 | FastAPI + Telegram bot + scheduler |
| `worker` | — | RQ worker (text, figure, combine queues) |
| `redis` | 6379 | Task queue (localhost only) |
| `rq-dashboard` | 9181 | Task monitoring UI (localhost only) |

---

## Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/digest` | Send all ready papers now |
| `/status` | Show queue statistics |
| `/ingest` | Trigger manual scraping |

Each paper card has inline buttons: **👍 Track this Trend** / **👎 Discard**

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/status` | Pipeline statistics |
| `POST` | `/trigger/ingest` | Manual ingestion |
| `POST` | `/trigger/synthesis` | Run weekly synthesis |
| `GET` | `/papers?status=approved` | List papers |
| `GET` | `/papers/{id}` | Paper detail |

---

## Switching LLM Providers

Just change one line in `.env`:

```bash
# Option A: Groq (recommended — fastest, 70B model, free forever)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...

# Option B: Together (good vision support)
LLM_PROVIDER=together
TOGETHER_API_KEY=...

# Option C: Cerebras (very fast, no vision)
LLM_PROVIDER=cerebras
CEREBRAS_API_KEY=...

# Option D: OpenRouter (many free models)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...

# Option E: Self-hosted Ollama (needs separate 8+ GB server)
LLM_PROVIDER=ollama
OLLAMA_HOST=http://your-big-server:11434
# Also: pip install ollama inside containers
```

Then `docker compose restart api worker`.

---

## Project Structure

```
research-pipeline/
├── docker-compose.yml          # 4 services (no Ollama container)
├── Dockerfile                  # Slim image (~450 MB)
├── .env.example                # All config with provider options
├── requirements.txt            # Lightweight deps (no torch)
├── scripts/
│   └── manage.sh               # CLI helper
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Pydantic settings
│   ├── schemas.py              # Data models
│   ├── llm_client.py           # ★ Unified LLM client (Groq/Together/etc)
│   ├── logging_cfg.py          # Structured logging
│   ├── ingestion/
│   │   ├── arxiv_scraper.py    # ArXiv API client
│   │   ├── semantic_scholar.py # S2 API client
│   │   └── downloader.py       # PDF downloader
│   ├── processing/
│   │   ├── prompts.py          # Few-shot prompt templates
│   │   ├── text_pipeline.py    # Text extraction + LLM API
│   │   ├── figure_pipeline.py  # Figure extraction + Vision API
│   │   └── combiner.py         # Merge into Markdown cards
│   ├── tasks/
│   │   ├── worker.py           # RQ task definitions
│   │   ├── scheduler.py        # Cron scheduler
│   │   ├── ingest_job.py       # Daily ingestion orchestrator
│   │   └── model_pull.py       # Provider connectivity check
│   ├── delivery/
│   │   └── telegram_bot.py     # Telegram bot + approval flow
│   ├── storage/
│   │   ├── paper_db.py         # SQLite paper state DB
│   │   ├── google_sheets.py    # gspread integration
│   │   └── vector_store.py     # ChromaDB + API embeddings
│   └── weekly/
│       └── meta_synthesis.py   # Cross-domain RAG synthesis
└── data/
    ├── pdfs/                   # Downloaded papers
    ├── figures/                # Extracted figure images
    └── db/                     # SQLite + ChromaDB
```

---

## Troubleshooting

**Rate limited by Groq?**
The pipeline has automatic retry with exponential backoff. If you're hitting limits often, reduce `MAX_PAPERS_PER_DAY` to 4 or switch to Together AI.

**Out of memory on e2-micro?**
```bash
# Add 1 GB swap (free, uses disk)
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Docker build fails on 30 GB disk?**
```bash
# Clean up Docker cache
docker system prune -af
```

**Vision model not available on Cerebras?**
Cerebras doesn't support vision — switch to Groq or Together for figure analysis:
```bash
LLM_PROVIDER=groq  # Has vision support
```

---

## License

MIT
