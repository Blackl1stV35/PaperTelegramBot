#!/usr/bin/env bash
###############################################################################
# scripts/manage.sh — Helper commands for operating the pipeline
###############################################################################
set -euo pipefail

CMD="${1:-help}"

case "$CMD" in
  ingest)
    echo "🔄 Triggering manual ingestion..."
    curl -s -X POST http://localhost:8000/trigger/ingest | python3 -m json.tool
    ;;
  digest)
    echo "📬 Triggering digest delivery..."
    curl -s -X POST http://localhost:8000/trigger/digest | python3 -m json.tool
    ;;
  synthesis)
    echo "📊 Triggering weekly synthesis..."
    curl -s -X POST http://localhost:8000/trigger/synthesis | python3 -m json.tool
    ;;
  status)
    echo "📈 Pipeline status:"
    curl -s http://localhost:8000/status | python3 -m json.tool
    ;;
  papers)
    echo "📄 Recent papers:"
    curl -s "http://localhost:8000/papers?status=${2:-}" | python3 -m json.tool
    ;;
  logs)
    docker compose logs -f "${2:-api}"
    ;;
  rq)
    echo "🔗 RQ Dashboard: http://localhost:9181"
    xdg-open "http://localhost:9181" 2>/dev/null || open "http://localhost:9181" 2>/dev/null || echo "Open http://localhost:9181 in your browser"
    ;;
  pull-models)
    echo "🧠 Pulling Ollama models..."
    docker compose exec api python -m app.tasks.model_pull
    ;;
  reset-db)
    echo "⚠️  Resetting paper database..."
    docker compose exec api rm -f /app/data/db/papers.db
    echo "Done. Restart the api container."
    ;;
  help|*)
    echo "Usage: ./scripts/manage.sh <command>"
    echo ""
    echo "Commands:"
    echo "  ingest       Trigger manual ingestion"
    echo "  digest       Send ready papers to Telegram"
    echo "  synthesis    Run weekly cross-domain synthesis"
    echo "  status       Show pipeline statistics"
    echo "  papers [s]   List papers (optional status filter)"
    echo "  logs [svc]   Follow logs for a service"
    echo "  rq           Open RQ Dashboard"
    echo "  pull-models  Pull Ollama models"
    echo "  reset-db     Reset the paper database"
    ;;
esac
