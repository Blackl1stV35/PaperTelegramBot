"""
Telegram Bot — sends daily digests and handles 👍/👎 approval.

Features:
  • /start — welcome message.
  • /digest — manually trigger sending today's ready papers.
  • /status — show queue counts.
  • /ingest — manually trigger daily ingestion (admin).
  • Inline keyboards on each paper card: 👍 Track / 👎 Discard.
  • On 👍: saves to Google Sheets + vector store.

Uses python-telegram-bot v21+ (async, Application API).
"""

from __future__ import annotations

import asyncio
import json
from typing import List

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from app.config import settings
from app.logging_cfg import log, setup_logging
from app.schemas import CombinedSummary, PaperStatus
from app.storage.paper_db import (
    get_ready_papers,
    update_status,
    row_to_combined,
    get_paper,
)
from app.storage.google_sheets import save_to_sheets
from app.storage.vector_store import add_paper


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────
def _approval_keyboard(paper_id: str) -> InlineKeyboardMarkup:
    """Build inline keyboard for a paper card."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍 Track this Trend", callback_data=f"approve:{paper_id}"),
            InlineKeyboardButton("👎 Discard", callback_data=f"discard:{paper_id}"),
        ]
    ])


def _truncate_message(text: str, max_len: int = 4000) -> str:
    """Telegram messages have a 4096 char limit."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n_(truncated)_"


# ─────────────────────────────────────────────────────────────────
#  Command Handlers
# ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message."""
    await update.message.reply_text(
        "🔬 *Research Summarization Pipeline*\n\n"
        "I deliver daily research digests from 9 scientific domains.\n\n"
        "Commands:\n"
        "/digest — Send today's papers\n"
        "/status — Queue statistics\n"
        "/ingest — Trigger manual ingestion\n",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show pipeline status."""
    ready = get_ready_papers()
    from app.storage.paper_db import get_all_approved_papers, get_queued_papers
    queued = get_queued_papers()
    approved = get_all_approved_papers()

    await update.message.reply_text(
        f"📊 *Pipeline Status*\n\n"
        f"📥 Queued: {len(queued)}\n"
        f"✅ Ready for review: {len(ready)}\n"
        f"👍 Approved total: {len(approved)}\n",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually send all ready papers."""
    rows = get_ready_papers()
    if not rows:
        await update.message.reply_text("📭 No papers ready for review right now.")
        return

    await update.message.reply_text(
        f"🔬 *Research Digest* — {len(rows)} papers\n{'─' * 30}",
        parse_mode=ParseMode.MARKDOWN,
    )

    for row in rows:
        combined = row_to_combined(row)
        msg_text = _truncate_message(combined.markdown_row)
        try:
            await update.message.reply_text(
                msg_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=_approval_keyboard(combined.paper.paper_id),
                disable_web_page_preview=True,
            )
        except Exception as exc:
            # Fallback: send without markdown if parsing fails
            log.warning("markdown_send_failed", error=str(exc))
            await update.message.reply_text(
                msg_text,
                reply_markup=_approval_keyboard(combined.paper.paper_id),
                disable_web_page_preview=True,
            )

        update_status(combined.paper.paper_id, PaperStatus.DELIVERED)


async def cmd_ingest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger daily ingestion."""
    await update.message.reply_text("⏳ Starting ingestion... this may take a while.")

    # Run in executor to avoid blocking the bot
    from app.tasks.ingest_job import run_daily_ingestion
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, run_daily_ingestion)

    await update.message.reply_text(f"✅ Ingestion complete. {count} papers enqueued for processing.")


# ─────────────────────────────────────────────────────────────────
#  Callback: 👍 / 👎
# ─────────────────────────────────────────────────────────────────
async def handle_approval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()

    data = query.data  # "approve:paper_id" or "discard:paper_id"
    action, paper_id = data.split(":", 1)

    if action == "approve":
        update_status(paper_id, PaperStatus.APPROVED)

        # Save to Google Sheets + Vector Store (in executor)
        row = get_paper(paper_id)
        if row:
            combined = row_to_combined(row)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_to_sheets, combined)
            await loop.run_in_executor(None, add_paper, combined)

        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            f"👍 *Tracked:* _{paper_id[:20]}_\nSaved to sheets + vector store.",
            parse_mode=ParseMode.MARKDOWN,
        )
        log.info("paper_approved", paper_id=paper_id)

    elif action == "discard":
        update_status(paper_id, PaperStatus.DISCARDED)
        await query.edit_message_reply_markup(reply_markup=None)
        await query.message.reply_text(
            f"👎 Discarded: _{paper_id[:20]}_",
            parse_mode=ParseMode.MARKDOWN,
        )
        log.info("paper_discarded", paper_id=paper_id)


# ─────────────────────────────────────────────────────────────────
#  Proactive Delivery (called by scheduler)
# ─────────────────────────────────────────────────────────────────
async def send_daily_digest_async(app: Application) -> None:
    """Send ready papers to the configured chat ID."""
    if not settings.telegram_chat_id:
        log.warning("no_telegram_chat_id_configured")
        return

    chat_id = int(settings.telegram_chat_id)
    rows = get_ready_papers()

    if not rows:
        await app.bot.send_message(chat_id, "📭 No new papers today.")
        return

    await app.bot.send_message(
        chat_id,
        f"🔬 *Daily Research Digest* — {len(rows)} papers\n{'─' * 30}",
        parse_mode=ParseMode.MARKDOWN,
    )

    for row in rows:
        combined = row_to_combined(row)
        msg_text = _truncate_message(combined.markdown_row)
        try:
            await app.bot.send_message(
                chat_id,
                msg_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=_approval_keyboard(combined.paper.paper_id),
                disable_web_page_preview=True,
            )
        except Exception:
            await app.bot.send_message(
                chat_id,
                msg_text,
                reply_markup=_approval_keyboard(combined.paper.paper_id),
                disable_web_page_preview=True,
            )
        update_status(combined.paper.paper_id, PaperStatus.DELIVERED)


# ─────────────────────────────────────────────────────────────────
#  Bot Entry Point
# ─────────────────────────────────────────────────────────────────
def run_bot() -> None:
    """Start the Telegram bot (blocking)."""
    setup_logging()

    if not settings.telegram_bot_token or settings.telegram_bot_token == "your-telegram-bot-token-here":
        log.warning("telegram_bot_token_not_set — bot will not start")
        # Keep the process alive so docker doesn't restart
        import time
        while True:
            time.sleep(3600)

    log.info("telegram_bot_starting")

    app = Application.builder().token(settings.telegram_bot_token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("digest", cmd_digest))
    app.add_handler(CommandHandler("ingest", cmd_ingest))
    app.add_handler(CallbackQueryHandler(handle_approval))

    log.info("telegram_bot_running")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    run_bot()
