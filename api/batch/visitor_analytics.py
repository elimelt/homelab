"""
Batch job for computing visitor analytics.

This job processes visitor join/leave events and calculates:
- Time spent per visit (session duration)
- Total time spent per visitor (aggregate across all visits)
- Visit frequency (number of visits per visitor over time)
- Recurring visitor identification (visitors with multiple historical visits)
- Historical time spent trends

Designed to run daily via cron or external scheduler.
Recommended schedule: Daily at 00:05 UTC (after midnight to capture full previous day)

Example cron entry:
    5 0 * * * cd /app && python -m api.batch.visitor_analytics

Usage:
    python -m api.batch.visitor_analytics [--days N] [--dry-run]

Arguments:
    --days N    Number of days to process (default: 1, meaning yesterday)
    --dry-run   Print computed stats without saving to database
"""

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api import db

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO8601 timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _extract_ip_from_payload(payload: dict) -> str | None:
    """Extract visitor IP from event payload."""
    if "visitor" in payload and isinstance(payload["visitor"], dict):
        return payload["visitor"].get("ip")
    return payload.get("ip")


def _extract_location_from_payload(payload: dict) -> tuple[str | None, str | None]:
    """Extract location (country, city) from event payload."""
    location = {}
    if "visitor" in payload and isinstance(payload["visitor"], dict):
        location = payload["visitor"].get("location", {})
    else:
        location = payload.get("location", {})
    return location.get("country"), location.get("city")


async def compute_visitor_stats(
    start_time: datetime,
    end_time: datetime,
) -> list[dict[str, Any]]:
    """
    Compute visitor statistics from events in the given time range.

    Returns a list of computed stats per visitor IP.
    """
    logger.info("Fetching visitor events from %s to %s", start_time, end_time)
    events = await db.fetch_visitor_events_for_analytics(start_time, end_time)
    logger.info("Found %d visitor events", len(events))

    if not events:
        return []

    # Track sessions per visitor IP
    # session = period between join and leave
    visitor_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_sessions: dict[str, datetime] = {}  # IP -> join timestamp
    visitor_locations: dict[str, tuple[str | None, str | None]] = {}

    for event in events:
        event_type = event["type"]
        payload = event["payload"]
        timestamp = _parse_timestamp(event["timestamp"])
        ip = _extract_ip_from_payload(payload)

        if not ip:
            continue

        if event_type == "join":
            active_sessions[ip] = timestamp
            country, city = _extract_location_from_payload(payload)
            if country or city:
                visitor_locations[ip] = (country, city)

        elif event_type == "leave":
            if ip in active_sessions:
                join_time = active_sessions.pop(ip)
                duration = (timestamp - join_time).total_seconds()
                # Cap session duration at 24 hours to handle orphaned sessions
                duration = min(duration, 86400)
                visitor_sessions[ip].append({
                    "join_time": join_time,
                    "leave_time": timestamp,
                    "duration_seconds": duration,
                })

    # Handle active sessions at end of period (assume they left at period end)
    for ip, join_time in active_sessions.items():
        duration = (end_time - join_time).total_seconds()
        duration = min(duration, 86400)
        visitor_sessions[ip].append({
            "join_time": join_time,
            "leave_time": end_time,
            "duration_seconds": duration,
        })

    # Compute statistics per visitor
    stats_list: list[dict[str, Any]] = []
    period_days = max(1, (end_time - start_time).days)

    for ip, sessions in visitor_sessions.items():
        if not sessions:
            continue

        total_visits = len(sessions)
        total_time = sum(s["duration_seconds"] for s in sessions)
        avg_duration = total_time / total_visits if total_visits > 0 else 0
        first_visit = min(s["join_time"] for s in sessions)
        last_visit = max(s["join_time"] for s in sessions)
        is_recurring = total_visits > 1
        frequency = total_visits / period_days

        country, city = visitor_locations.get(ip, (None, None))

        stats_list.append({
            "visitor_ip": ip,
            "period_start": start_time,
            "period_end": end_time,
            "total_visits": total_visits,
            "total_time_seconds": total_time,
            "avg_session_duration_seconds": avg_duration,
            "is_recurring": is_recurring,
            "first_visit_at": first_visit,
            "last_visit_at": last_visit,
            "visit_frequency_per_day": frequency,
            "location_country": country,
            "location_city": city,
        })

    logger.info("Computed stats for %d unique visitors", len(stats_list))
    return stats_list


async def save_visitor_stats(stats_list: list[dict[str, Any]]) -> int:
    """Save computed stats to the database. Returns count of saved records."""
    saved = 0
    for stats in stats_list:
        try:
            await db.upsert_visitor_stats(
                visitor_ip=stats["visitor_ip"],
                period_start=stats["period_start"],
                period_end=stats["period_end"],
                total_visits=stats["total_visits"],
                total_time_seconds=stats["total_time_seconds"],
                avg_session_duration_seconds=stats["avg_session_duration_seconds"],
                is_recurring=stats["is_recurring"],
                first_visit_at=stats["first_visit_at"],
                last_visit_at=stats["last_visit_at"],
                visit_frequency_per_day=stats["visit_frequency_per_day"],
                location_country=stats.get("location_country"),
                location_city=stats.get("location_city"),
            )
            saved += 1
        except Exception as e:
            logger.error("Failed to save stats for %s: %s", stats["visitor_ip"], e)
    return saved


async def run_batch_job(days: int = 1, dry_run: bool = False) -> dict[str, Any]:
    """
    Run the visitor analytics batch job.

    Args:
        days: Number of days to process (1 = yesterday only)
        dry_run: If True, compute but don't save to database

    Returns:
        Summary of the batch job results.
    """
    logger.info("Starting visitor analytics batch job (days=%d, dry_run=%s)", days, dry_run)

    # Initialize database schema if needed
    try:
        await db.init_pool()
    except Exception as e:
        logger.warning("DB init warning (may be ok): %s", e)

    now = datetime.now(UTC)
    results: list[dict[str, Any]] = []

    for day_offset in range(days, 0, -1):
        # Process each day separately
        end_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=day_offset - 1
        )
        start_time = end_time - timedelta(days=1)

        logger.info("Processing day: %s", start_time.date())

        stats_list = await compute_visitor_stats(start_time, end_time)

        if dry_run:
            logger.info("[DRY RUN] Would save %d visitor stats:", len(stats_list))
            for s in stats_list[:5]:  # Show first 5
                logger.info(
                    "  - %s: %d visits, %.1f sec avg duration, recurring=%s",
                    s["visitor_ip"],
                    s["total_visits"],
                    s["avg_session_duration_seconds"],
                    s["is_recurring"],
                )
            if len(stats_list) > 5:
                logger.info("  ... and %d more", len(stats_list) - 5)
            saved = 0
        else:
            saved = await save_visitor_stats(stats_list)
            logger.info("Saved %d visitor stats for %s", saved, start_time.date())

        results.append(
            {
                "date": start_time.date().isoformat(),
                "visitors_processed": len(stats_list),
                "records_saved": saved,
            }
        )

    summary = {
        "job_started_at": now.isoformat(),
        "job_completed_at": datetime.now(UTC).isoformat(),
        "days_processed": days,
        "dry_run": dry_run,
        "daily_results": results,
        "total_visitors": sum(r["visitors_processed"] for r in results),
        "total_records_saved": sum(r["records_saved"] for r in results),
    }

    logger.info("Batch job completed: %s", summary)
    return summary


async def start_analytics_scheduler(stop_event: asyncio.Event) -> list[asyncio.Task]:
    """
    Start the visitor analytics scheduler as a background task.

    The scheduler runs the batch job at a configurable interval (default: daily).
    It can be configured via environment variables:
        - ANALYTICS_INTERVAL_HOURS: Hours between runs (default: 24)
        - ANALYTICS_INITIAL_DELAY_SEC: Seconds to wait before first run (default: 60)
        - ANALYTICS_DAYS: Number of days to process each run (default: 1)

    Args:
        stop_event: Event to signal shutdown.

    Returns:
        List containing the scheduler task.
    """

    async def _scheduler_loop() -> None:
        interval_hours = int(os.getenv("ANALYTICS_INTERVAL_HOURS", "24"))
        days_to_process = int(os.getenv("ANALYTICS_DAYS", "1"))
        interval_seconds = interval_hours * 3600

        logger.info(
            "Visitor analytics scheduler started (interval=%dh, days=%d)",
            interval_hours,
            days_to_process,
        )

        while not stop_event.is_set():
            try:
                logger.info("Running scheduled visitor analytics batch job")
                await run_batch_job(days=days_to_process, dry_run=False)
            except Exception:
                logger.exception("Scheduled analytics batch job failed")

            # Wait for next interval or stop
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # Normal timeout, run again

        logger.info("Visitor analytics scheduler stopped")

    task = asyncio.create_task(_scheduler_loop())
    return [task]


def main() -> None:
    """CLI entry point for the batch job."""
    parser = argparse.ArgumentParser(
        description="Compute visitor analytics from event data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to process (default: 1, meaning yesterday)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed stats without saving to database",
    )
    args = parser.parse_args()

    asyncio.run(run_batch_job(days=args.days, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

