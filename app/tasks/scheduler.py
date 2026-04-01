"""
Scheduler — runs as a background process in the api container.

Two schedules:
  1. Daily ingestion at the configured hour.
  2. Weekly meta-synthesis on the configured day.

Uses the `schedule` library (no external cron daemon needed).
"""

from __future__ import annotations

import time
import schedule

from app.config import settings
from app.logging_cfg import log, setup_logging
from app.tasks.ingest_job import run_daily_ingestion
from app.weekly.meta_synthesis import run_weekly_synthesis


def _daily_job() -> None:
    log.info("scheduler_daily_triggered")
    try:
        run_daily_ingestion()
    except Exception as exc:
        log.error("scheduler_daily_failed", error=str(exc))


def _weekly_job() -> None:
    log.info("scheduler_weekly_triggered")
    try:
        run_weekly_synthesis()
    except Exception as exc:
        log.error("scheduler_weekly_failed", error=str(exc))


def start_scheduler() -> None:
    """Configure and run the schedule loop (blocking)."""
    setup_logging()
    log.info(
        "scheduler_starting",
        daily_at=f"{settings.daily_ingest_hour:02d}:{settings.daily_ingest_minute:02d}",
        weekly_day=settings.weekly_synthesis_day,
        weekly_hour=settings.weekly_synthesis_hour,
    )

    daily_time = f"{settings.daily_ingest_hour:02d}:{settings.daily_ingest_minute:02d}"
    schedule.every().day.at(daily_time).do(_daily_job)

    day_map = {
        "monday": schedule.every().monday,
        "tuesday": schedule.every().tuesday,
        "wednesday": schedule.every().wednesday,
        "thursday": schedule.every().thursday,
        "friday": schedule.every().friday,
        "saturday": schedule.every().saturday,
        "sunday": schedule.every().sunday,
    }
    weekly_sched = day_map.get(settings.weekly_synthesis_day.lower(), schedule.every().sunday)
    weekly_time = f"{settings.weekly_synthesis_hour:02d}:00"
    weekly_sched.at(weekly_time).do(_weekly_job)

    log.info("scheduler_running")

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    start_scheduler()
