import json
import logging
import secrets
import string
from datetime import UTC, datetime
from typing import Any

from psycopg import errors as pg_errors
from psycopg.types.json import Json

# Import pool management from the centralized core module
from api.db.core import _get_connection, close_pool, init_pool

_logger = logging.getLogger(__name__)


async def _ensure_schema() -> None:
    async with _get_connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                channel TEXT NOT NULL,
                sender TEXT NOT NULL,
                text TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                deleted_at TIMESTAMPTZ NULL,
                message_id TEXT NULL,
                reply_to TEXT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_channel_ts ON chat_messages (channel, ts DESC);
            CREATE TABLE IF NOT EXISTS events (
                id BIGSERIAL PRIMARY KEY,
                topic TEXT NOT NULL,
                type TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL,
                payload JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_topic_ts ON events (topic, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events (type, ts DESC);
            """
        )
        try:
            await conn.execute(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL;"
            )
            await conn.execute(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS message_id TEXT NULL;"
            )
            await conn.execute(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS reply_to TEXT NULL;"
            )
        except Exception:
            pass
        try:
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_message_id ON chat_messages (message_id) WHERE message_id IS NOT NULL;"
            )
        except Exception:
            pass
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS w2m_events (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                dates JSONB NOT NULL,
                time_slots JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                creator_name TEXT
            );
            CREATE TABLE IF NOT EXISTS w2m_availabilities (
                id BIGSERIAL PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES w2m_events(id) ON DELETE CASCADE,
                participant_name TEXT NOT NULL,
                available_slots JSONB NOT NULL,
                password_hash TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                UNIQUE (event_id, participant_name)
            );
            CREATE INDEX IF NOT EXISTS idx_w2m_avail_event ON w2m_availabilities (event_id);
            ALTER TABLE w2m_availabilities ADD COLUMN IF NOT EXISTS password_hash TEXT;
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visitor_stats (
                id BIGSERIAL PRIMARY KEY,
                visitor_ip TEXT NOT NULL,
                computed_at TIMESTAMPTZ NOT NULL,
                period_start TIMESTAMPTZ NOT NULL,
                period_end TIMESTAMPTZ NOT NULL,
                total_visits INTEGER NOT NULL DEFAULT 0,
                total_time_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                avg_session_duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                is_recurring BOOLEAN NOT NULL DEFAULT FALSE,
                first_visit_at TIMESTAMPTZ,
                last_visit_at TIMESTAMPTZ,
                visit_frequency_per_day DOUBLE PRECISION NOT NULL DEFAULT 0,
                location_country TEXT,
                location_city TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_visitor_stats_ip ON visitor_stats (visitor_ip);
            CREATE INDEX IF NOT EXISTS idx_visitor_stats_period ON visitor_stats (period_start, period_end);
            CREATE INDEX IF NOT EXISTS idx_visitor_stats_computed ON visitor_stats (computed_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_visitor_stats_ip_period
                ON visitor_stats (visitor_ip, period_start, period_end);
            """
        )
        await conn.execute(
            """
            -- Notes categories table
            CREATE TABLE IF NOT EXISTS notes_categories (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_notes_categories_name ON notes_categories (name);

            -- Notes documents table
            CREATE TABLE IF NOT EXISTS notes_documents (
                id BIGSERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                category_id BIGINT REFERENCES notes_categories(id) ON DELETE SET NULL,
                description TEXT,
                content TEXT NOT NULL,
                last_modified TIMESTAMPTZ NOT NULL,
                git_commit_sha TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_notes_docs_category ON notes_documents (category_id);
            CREATE INDEX IF NOT EXISTS idx_notes_docs_title ON notes_documents (title);
            CREATE INDEX IF NOT EXISTS idx_notes_docs_path ON notes_documents (file_path);

            -- Notes tags table
            CREATE TABLE IF NOT EXISTS notes_tags (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_notes_tags_name ON notes_tags (name);

            -- Document-Tag junction table (many-to-many)
            CREATE TABLE IF NOT EXISTS notes_document_tags (
                document_id BIGINT NOT NULL REFERENCES notes_documents(id) ON DELETE CASCADE,
                tag_id BIGINT NOT NULL REFERENCES notes_tags(id) ON DELETE CASCADE,
                PRIMARY KEY (document_id, tag_id)
            );
            CREATE INDEX IF NOT EXISTS idx_notes_doc_tags_doc ON notes_document_tags (document_id);
            CREATE INDEX IF NOT EXISTS idx_notes_doc_tags_tag ON notes_document_tags (tag_id);
            """
        )
        await conn.execute(
            """
            -- Sync jobs table - tracks overall sync operations
            CREATE TABLE IF NOT EXISTS notes_sync_jobs (
                id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, paused
                commit_sha TEXT,
                total_items INTEGER NOT NULL DEFAULT 0,
                completed_items INTEGER NOT NULL DEFAULT 0,
                failed_items INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                error_message TEXT,
                rate_limit_reset_at TIMESTAMPTZ,  -- When GitHub rate limit resets
                last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_sync_jobs_status ON notes_sync_jobs (status);
            CREATE INDEX IF NOT EXISTS idx_sync_jobs_created ON notes_sync_jobs (created_at DESC);

            -- Sync job items - tracks individual file sync status
            CREATE TABLE IF NOT EXISTS notes_sync_job_items (
                id BIGSERIAL PRIMARY KEY,
                job_id BIGINT NOT NULL REFERENCES notes_sync_jobs(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',  -- pending, success, failed, skipped
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                last_attempt_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_sync_items_job ON notes_sync_job_items (job_id);
            CREATE INDEX IF NOT EXISTS idx_sync_items_status ON notes_sync_job_items (status);
            CREATE INDEX IF NOT EXISTS idx_sync_items_job_status ON notes_sync_job_items (job_id, status);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_sync_items_job_path ON notes_sync_job_items (job_id, file_path);
            """
        )
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            pass
        try:
            await conn.execute(
                """
                ALTER TABLE notes_documents
                ADD COLUMN IF NOT EXISTS search_vector tsvector
                GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
                    setweight(to_tsvector('english', COALESCE(description, '')), 'B') ||
                    setweight(to_tsvector('english', COALESCE(content, '')), 'C')
                ) STORED;
                """
            )
        except Exception:
            pass
        try:
            await conn.execute(
                "ALTER TABLE notes_documents ADD COLUMN IF NOT EXISTS content_embedding vector(384);"
            )
        except Exception:
            pass
        try:
            await conn.execute(
                "ALTER TABLE notes_documents ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;"
            )
        except Exception:
            pass
        try:
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_docs_search_vector ON notes_documents USING GIN (search_vector);"
            )
        except Exception:
            pass
        try:
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_notes_docs_embedding
                ON notes_documents USING hnsw (content_embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        except Exception:
            pass


async def insert_chat_message(
    channel: str,
    sender: str,
    text: str,
    ts_iso: str,
    message_id: str | None = None,
    reply_to: str | None = None,
) -> None:
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    async with _get_connection() as conn:
        try:
            await conn.execute(
                "INSERT INTO chat_messages (channel, sender, text, ts, message_id, reply_to) VALUES (%s, %s, %s, %s, %s, %s)",
                (channel, sender, text, ts, message_id, reply_to),
            )
        except pg_errors.UndefinedColumn:
            await _ensure_schema()
            await conn.execute(
                "INSERT INTO chat_messages (channel, sender, text, ts) VALUES (%s, %s, %s, %s)",
                (channel, sender, text, ts),
            )


async def insert_event(topic: str, event_type: str, payload: dict, ts_iso: str) -> None:
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    async with _get_connection() as conn:
        await conn.execute(
            "INSERT INTO events (topic, type, ts, payload) VALUES (%s, %s, %s, %s)",
            (topic, event_type, ts, Json(payload)),
        )


async def fetch_chat_history(
    channel: str, before_iso: str | None, limit: int
) -> list[dict[str, Any]]:
    before_ts = (
        datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        if before_iso
        else datetime.now(UTC)
    )
    async with _get_connection() as conn:
        try:
            rows = await conn.execute(
                "SELECT channel, sender, text, ts, message_id, reply_to FROM chat_messages WHERE channel=%s AND ts < %s AND deleted_at IS NULL ORDER BY ts DESC LIMIT %s",
                (channel, before_ts, limit),
            )
            result = []
            async for row in rows:
                channel_v, sender, text, ts, mid, reply_to = row
                result.append(
                    {
                        "type": "chat_message",
                        "channel": channel_v,
                        "sender": sender,
                        "text": text,
                        "timestamp": ts.astimezone(UTC).isoformat(),
                        "id": mid,
                        "reply_to": reply_to,
                    }
                )
            return result
        except pg_errors.UndefinedColumn:
            await _ensure_schema()
            rows = await conn.execute(
                "SELECT channel, sender, text, ts FROM chat_messages WHERE channel=%s AND ts < %s AND deleted_at IS NULL ORDER BY ts DESC LIMIT %s",
                (channel, before_ts, limit),
            )
        result = []
        async for row in rows:
            channel_v, sender, text, ts = row
            result.append(
                {
                    "type": "chat_message",
                    "channel": channel_v,
                    "sender": sender,
                    "text": text,
                    "timestamp": ts.astimezone(UTC).isoformat(),
                }
            )
        return result


async def fetch_events(
    topic: str | None,
    event_type: str | None,
    before_iso: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    before_ts = (
        datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        if before_iso
        else datetime.now(UTC)
    )
    clauses = ["ts < %s"]
    params: list[Any] = [before_ts]
    if topic:
        clauses.append("topic = %s")
        params.append(topic)
    if event_type:
        clauses.append("type = %s")
        params.append(event_type)
    where = " AND ".join(clauses)
    sql = f"SELECT topic, type, ts, payload FROM events WHERE {where} ORDER BY ts DESC LIMIT %s"
    params.append(limit)
    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(params))
        out: list[dict[str, Any]] = []
        async for row in rows:
            topic_v, type_v, ts, payload = row
            out.append(
                {
                    "topic": topic_v,
                    "type": type_v,
                    "timestamp": ts.astimezone(UTC).isoformat(),
                    "payload": payload,
                }
            )
        return out


async def soft_delete_chat_history(
    channel: str | None = None, before_iso: str | None = None
) -> int:
    conditions: list[str] = ["deleted_at IS NULL"]
    params: list[Any] = []
    if channel:
        conditions.append("channel = %s")
        params.append(channel)
    if before_iso:
        before_ts = datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        conditions.append("ts < %s")
        params.append(before_ts)
    where = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"UPDATE chat_messages SET deleted_at = %s WHERE {where} RETURNING id"
    now_ts = datetime.now(UTC)
    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple([now_ts] + params))
        count = 0
        async for _ in rows:
            count += 1
        return count


def _generate_event_id(length: int = 10) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


async def w2m_create_event(
    name: str,
    dates: list[str],
    time_slots: list[str],
    description: str | None = None,
    creator_name: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    async with _get_connection() as conn:
        for _ in range(10):
            event_id = _generate_event_id()
            try:
                await conn.execute(
                    """INSERT INTO w2m_events (id, name, description, dates, time_slots, created_at, creator_name)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        event_id,
                        name,
                        description,
                        json.dumps(dates),
                        json.dumps(time_slots),
                        now,
                        creator_name,
                    ),
                )
                return {
                    "id": event_id,
                    "name": name,
                    "description": description,
                    "dates": dates,
                    "time_slots": time_slots,
                    "created_at": now.isoformat(),
                    "creator_name": creator_name,
                }
            except pg_errors.UniqueViolation:
                continue
        raise RuntimeError("Failed to generate unique event ID")


async def w2m_get_event(event_id: str) -> dict[str, Any] | None:
    async with _get_connection() as conn:
        rows = await conn.execute(
            "SELECT id, name, description, dates, time_slots, created_at, creator_name FROM w2m_events WHERE id = %s",
            (event_id,),
        )
        row = await rows.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "dates": row[3],
            "time_slots": row[4],
            "created_at": row[5].astimezone(UTC).isoformat(),
            "creator_name": row[6],
        }


async def w2m_get_availabilities(event_id: str) -> list[dict[str, Any]]:
    async with _get_connection() as conn:
        rows = await conn.execute(
            "SELECT participant_name, available_slots, created_at, updated_at FROM w2m_availabilities WHERE event_id = %s",
            (event_id,),
        )
        result = []
        async for row in rows:
            result.append(
                {
                    "participant_name": row[0],
                    "available_slots": row[1],
                    "created_at": row[2].astimezone(UTC).isoformat(),
                    "updated_at": row[3].astimezone(UTC).isoformat(),
                }
            )
        return result


async def w2m_get_availability(event_id: str, participant_name: str) -> dict[str, Any] | None:
    async with _get_connection() as conn:
        row = await (
            await conn.execute(
                "SELECT available_slots, password_hash, created_at, updated_at FROM w2m_availabilities WHERE event_id = %s AND participant_name = %s",
                (event_id, participant_name),
            )
        ).fetchone()
        if not row:
            return None
        return {
            "available_slots": row[0],
            "password_hash": row[1],
            "created_at": row[2].astimezone(UTC).isoformat(),
            "updated_at": row[3].astimezone(UTC).isoformat(),
        }


async def w2m_upsert_availability(
    event_id: str,
    participant_name: str,
    available_slots: list[str],
    password_hash: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    async with _get_connection() as conn:
        if password_hash is not None:
            await conn.execute(
                """INSERT INTO w2m_availabilities (event_id, participant_name, available_slots, password_hash, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (event_id, participant_name) DO UPDATE SET available_slots = EXCLUDED.available_slots, updated_at = EXCLUDED.updated_at""",
                (event_id, participant_name, json.dumps(available_slots), password_hash, now, now),
            )
        else:
            await conn.execute(
                """INSERT INTO w2m_availabilities (event_id, participant_name, available_slots, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (event_id, participant_name) DO UPDATE SET available_slots = EXCLUDED.available_slots, updated_at = EXCLUDED.updated_at""",
                (event_id, participant_name, json.dumps(available_slots), now, now),
            )
        return {
            "event_id": event_id,
            "participant_name": participant_name,
            "available_slots": available_slots,
            "updated_at": now.isoformat(),
        }



async def upsert_visitor_stats(
    visitor_ip: str,
    period_start: datetime,
    period_end: datetime,
    total_visits: int,
    total_time_seconds: float,
    avg_session_duration_seconds: float,
    is_recurring: bool,
    first_visit_at: datetime | None,
    last_visit_at: datetime | None,
    visit_frequency_per_day: float,
    location_country: str | None = None,
    location_city: str | None = None,
) -> dict[str, Any]:
    """Insert or update visitor statistics for a given IP and time period."""
    now = datetime.now(UTC)
    async with _get_connection() as conn:
        await conn.execute(
            """
            INSERT INTO visitor_stats (
                visitor_ip, computed_at, period_start, period_end,
                total_visits, total_time_seconds, avg_session_duration_seconds,
                is_recurring, first_visit_at, last_visit_at,
                visit_frequency_per_day, location_country, location_city
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (visitor_ip, period_start, period_end) DO UPDATE SET
                computed_at = EXCLUDED.computed_at,
                total_visits = EXCLUDED.total_visits,
                total_time_seconds = EXCLUDED.total_time_seconds,
                avg_session_duration_seconds = EXCLUDED.avg_session_duration_seconds,
                is_recurring = EXCLUDED.is_recurring,
                first_visit_at = EXCLUDED.first_visit_at,
                last_visit_at = EXCLUDED.last_visit_at,
                visit_frequency_per_day = EXCLUDED.visit_frequency_per_day,
                location_country = EXCLUDED.location_country,
                location_city = EXCLUDED.location_city
            """,
            (
                visitor_ip,
                now,
                period_start,
                period_end,
                total_visits,
                total_time_seconds,
                avg_session_duration_seconds,
                is_recurring,
                first_visit_at,
                last_visit_at,
                visit_frequency_per_day,
                location_country,
                location_city,
            ),
        )
        return {
            "visitor_ip": visitor_ip,
            "computed_at": now.isoformat(),
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_visits": total_visits,
            "total_time_seconds": total_time_seconds,
            "avg_session_duration_seconds": avg_session_duration_seconds,
            "is_recurring": is_recurring,
            "first_visit_at": first_visit_at.isoformat() if first_visit_at else None,
            "last_visit_at": last_visit_at.isoformat() if last_visit_at else None,
            "visit_frequency_per_day": visit_frequency_per_day,
            "location_country": location_country,
            "location_city": location_city,
        }


async def fetch_visitor_stats(
    visitor_ip: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    is_recurring: bool | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch visitor statistics with optional filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if visitor_ip:
        clauses.append("visitor_ip = %s")
        params.append(visitor_ip)
    if start_date:
        start_ts = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        clauses.append("period_end >= %s")
        params.append(start_ts)
    if end_date:
        end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        clauses.append("period_start <= %s")
        params.append(end_ts)
    if is_recurring is not None:
        clauses.append("is_recurring = %s")
        params.append(is_recurring)

    where = " AND ".join(clauses) if clauses else "TRUE"
    sql = f"""
        SELECT visitor_ip, computed_at, period_start, period_end,
               total_visits, total_time_seconds, avg_session_duration_seconds,
               is_recurring, first_visit_at, last_visit_at,
               visit_frequency_per_day, location_country, location_city
        FROM visitor_stats
        WHERE {where}
        ORDER BY computed_at DESC
        LIMIT %s
    """
    params.append(limit)

    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(params))
        result: list[dict[str, Any]] = []
        async for row in rows:
            result.append(
                {
                    "visitor_ip": row[0],
                    "computed_at": row[1].astimezone(UTC).isoformat(),
                    "period_start": row[2].astimezone(UTC).isoformat(),
                    "period_end": row[3].astimezone(UTC).isoformat(),
                    "total_visits": row[4],
                    "total_time_seconds": row[5],
                    "avg_session_duration_seconds": row[6],
                    "is_recurring": row[7],
                    "first_visit_at": row[8].astimezone(UTC).isoformat() if row[8] else None,
                    "last_visit_at": row[9].astimezone(UTC).isoformat() if row[9] else None,
                    "visit_frequency_per_day": row[10],
                    "location_country": row[11],
                    "location_city": row[12],
                }
            )
        return result


async def fetch_visitor_events_for_analytics(
    start_time: datetime,
    end_time: datetime,
) -> list[dict[str, Any]]:
    """Fetch visitor join/leave events from the events table for analytics processing."""
    sql = """
        SELECT ts, type, payload
        FROM events
        WHERE topic = 'visitor_updates'
          AND type IN ('join', 'leave')
          AND ts >= %s AND ts < %s
        ORDER BY ts ASC
    """
    async with _get_connection() as conn:
        rows = await conn.execute(sql, (start_time, end_time))
        result: list[dict[str, Any]] = []
        async for row in rows:
            result.append(
                {
                    "timestamp": row[0].astimezone(UTC).isoformat(),
                    "type": row[1],
                    "payload": row[2],
                }
            )
        return result


async def get_visitor_analytics_summary(
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Get aggregate summary of visitor analytics."""
    clauses: list[str] = []
    params: list[Any] = []

    if start_date:
        start_ts = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        clauses.append("period_end >= %s")
        params.append(start_ts)
    if end_date:
        end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        clauses.append("period_start <= %s")
        params.append(end_ts)

    where = " AND ".join(clauses) if clauses else "TRUE"
    sql = f"""
        SELECT
            COUNT(DISTINCT visitor_ip) as unique_visitors,
            SUM(total_visits) as total_visits,
            AVG(avg_session_duration_seconds) as avg_session_duration,
            SUM(total_time_seconds) as total_time_spent,
            COUNT(*) FILTER (WHERE is_recurring) as recurring_visitors,
            AVG(visit_frequency_per_day) as avg_visit_frequency
        FROM visitor_stats
        WHERE {where}
    """

    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(params))
        row = await rows.fetchone()
        if not row:
            return {
                "unique_visitors": 0,
                "total_visits": 0,
                "avg_session_duration_seconds": 0,
                "total_time_spent_seconds": 0,
                "recurring_visitors": 0,
                "avg_visit_frequency_per_day": 0,
            }
        return {
            "unique_visitors": row[0] or 0,
            "total_visits": row[1] or 0,
            "avg_session_duration_seconds": float(row[2]) if row[2] else 0,
            "total_time_spent_seconds": float(row[3]) if row[3] else 0,
            "recurring_visitors": row[4] or 0,
            "avg_visit_frequency_per_day": float(row[5]) if row[5] else 0,
        }



async def insert_click_events(events: list[dict[str, Any]], client_ip: str) -> int:
    """Insert a batch of click events into the events table.

    Each click event is stored with topic='clicks' and type='click'.
    The client_ip is added to the payload for attribution.

    Returns the number of events inserted.
    """
    if not events:
        return 0

    inserted = 0
    async with _get_connection() as conn:
        for event in events:
            ts_ms = event.get("ts")
            if ts_ms and isinstance(ts_ms, int | float):
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            else:
                ts = datetime.now(UTC)

            payload = {**event, "client_ip": client_ip}

            await conn.execute(
                "INSERT INTO events (topic, type, ts, payload) VALUES (%s, %s, %s, %s)",
                ("clicks", "click", ts, Json(payload)),
            )
            inserted += 1

    return inserted


async def fetch_click_events(
    start_date: str | None = None,
    end_date: str | None = None,
    page_path: str | None = None,
    client_ip: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch click events with optional filters."""
    clauses: list[str] = ["topic = 'clicks'", "type = 'click'"]
    params: list[Any] = []

    if start_date:
        start_ts = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        clauses.append("ts >= %s")
        params.append(start_ts)
    if end_date:
        end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        clauses.append("ts < %s")
        params.append(end_ts)
    if page_path:
        clauses.append("payload->>'page'->>'path' = %s")
        params.append(page_path)
    if client_ip:
        clauses.append("payload->>'client_ip' = %s")
        params.append(client_ip)

    where = " AND ".join(clauses)
    sql = f"""
        SELECT ts, payload
        FROM events
        WHERE {where}
        ORDER BY ts DESC
        LIMIT %s
    """
    params.append(limit)

    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(params))
        result: list[dict[str, Any]] = []
        async for row in rows:
            result.append({
                "timestamp": row[0].astimezone(UTC).isoformat(),
                "event": row[1],
            })
        return result



async def notes_get_or_create_category(name: str) -> int:
    """Get or create a category by name, returns the category id."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            "SELECT id FROM notes_categories WHERE name = %s", (name,)
        )).fetchone()
        if row:
            return row[0]
        row = await (await conn.execute(
            "INSERT INTO notes_categories (name) VALUES (%s) RETURNING id", (name,)
        )).fetchone()
        return row[0]


async def notes_get_or_create_tag(name: str) -> int:
    """Get or create a tag by name, returns the tag id."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            "SELECT id FROM notes_tags WHERE name = %s", (name,)
        )).fetchone()
        if row:
            return row[0]
        row = await (await conn.execute(
            "INSERT INTO notes_tags (name) VALUES (%s) RETURNING id", (name,)
        )).fetchone()
        return row[0]


async def notes_upsert_document(
    file_path: str,
    title: str,
    category_name: str | None,
    description: str | None,
    content: str,
    git_commit_sha: str | None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Insert or update a notes document."""
    now = datetime.now(UTC)

    async with _get_connection() as conn:
        category_id = None
        if category_name:
            cat_row = await (await conn.execute(
                "SELECT id FROM notes_categories WHERE name = %s", (category_name,)
            )).fetchone()
            if cat_row:
                category_id = cat_row[0]
            else:
                cat_row = await (await conn.execute(
                    "INSERT INTO notes_categories (name) VALUES (%s) RETURNING id", (category_name,)
                )).fetchone()
                category_id = cat_row[0]

        row = await (await conn.execute(
            """
            INSERT INTO notes_documents (file_path, title, category_id, description, content, last_modified, git_commit_sha, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE SET
                title = EXCLUDED.title,
                category_id = EXCLUDED.category_id,
                description = EXCLUDED.description,
                content = EXCLUDED.content,
                last_modified = EXCLUDED.last_modified,
                git_commit_sha = EXCLUDED.git_commit_sha,
                updated_at = EXCLUDED.updated_at
            RETURNING id
            """,
            (file_path, title, category_id, description, content, now, git_commit_sha, now, now),
        )).fetchone()
        doc_id = row[0]

        await conn.execute(
            "DELETE FROM notes_document_tags WHERE document_id = %s", (doc_id,)
        )

        if tags:
            for tag_name in tags:
                tag_name = tag_name.strip()
                if not tag_name:
                    continue
                tag_row = await (await conn.execute(
                    "SELECT id FROM notes_tags WHERE name = %s", (tag_name,)
                )).fetchone()
                if tag_row:
                    tag_id = tag_row[0]
                else:
                    tag_row = await (await conn.execute(
                        "INSERT INTO notes_tags (name) VALUES (%s) RETURNING id", (tag_name,)
                    )).fetchone()
                    tag_id = tag_row[0]

                await conn.execute(
                    "INSERT INTO notes_document_tags (document_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (doc_id, tag_id),
                )

        return {
            "id": doc_id,
            "file_path": file_path,
            "title": title,
            "category": category_name,
            "description": description,
            "git_commit_sha": git_commit_sha,
            "updated_at": now.isoformat(),
        }


async def notes_delete_documents_not_in(file_paths: list[str]) -> int:
    """Delete documents whose file_path is not in the provided list. Returns count deleted."""
    if not file_paths:
        return 0
    async with _get_connection() as conn:
        placeholders = ", ".join(["%s"] * len(file_paths))
        result = await conn.execute(
            f"DELETE FROM notes_documents WHERE file_path NOT IN ({placeholders}) RETURNING id",
            tuple(file_paths),
        )
        count = 0
        async for _ in result:
            count += 1
        return count



async def notes_fetch_documents(
    category_id: int | None = None,
    tag_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch documents with optional filters, returns metadata without content."""

    clauses: list[str] = []
    params: list[Any] = []
    joins = ""

    if category_id is not None:
        clauses.append("d.category_id = %s")
        params.append(category_id)

    if tag_id is not None:
        joins = " JOIN notes_document_tags dt ON d.id = dt.document_id"
        clauses.append("dt.tag_id = %s")
        params.append(tag_id)

    where = " AND ".join(clauses) if clauses else "TRUE"

    sql = f"""
        SELECT DISTINCT d.id, d.file_path, d.title, c.name as category, d.description, d.last_modified, d.git_commit_sha
        FROM notes_documents d
        LEFT JOIN notes_categories c ON d.category_id = c.id
        {joins}
        WHERE {where}
        ORDER BY d.title ASC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])

    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(params))
        result: list[dict[str, Any]] = []
        async for row in rows:
            tag_rows = await conn.execute(
                "SELECT t.name FROM notes_tags t JOIN notes_document_tags dt ON t.id = dt.tag_id WHERE dt.document_id = %s",
                (row[0],),
            )
            tags = [t[0] async for t in tag_rows]

            result.append({
                "id": row[0],
                "file_path": row[1],
                "title": row[2],
                "category": row[3],
                "description": row[4],
                "last_modified": row[5].astimezone(UTC).isoformat() if row[5] else None,
                "git_commit_sha": row[6],
                "tags": tags,
            })
        return result


async def notes_count_documents(
    category_id: int | None = None,
    tag_id: int | None = None,
) -> int:
    """Count documents with optional filters."""

    clauses: list[str] = []
    params: list[Any] = []
    joins = ""

    if category_id is not None:
        clauses.append("d.category_id = %s")
        params.append(category_id)

    if tag_id is not None:
        joins = " JOIN notes_document_tags dt ON d.id = dt.document_id"
        clauses.append("dt.tag_id = %s")
        params.append(tag_id)

    where = " AND ".join(clauses) if clauses else "TRUE"

    sql = f"""
        SELECT COUNT(DISTINCT d.id)
        FROM notes_documents d
        {joins}
        WHERE {where}
    """

    async with _get_connection() as conn:
        row = await (await conn.execute(sql, tuple(params))).fetchone()
        return row[0] if row else 0


async def notes_get_document_by_id(doc_id: int) -> dict[str, Any] | None:
    """Get a single document by ID, including full content."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            """
            SELECT d.id, d.file_path, d.title, c.name as category, d.description, d.content, d.last_modified, d.git_commit_sha
            FROM notes_documents d
            LEFT JOIN notes_categories c ON d.category_id = c.id
            WHERE d.id = %s
            """,
            (doc_id,),
        )).fetchone()

        if not row:
            return None

        tag_rows = await conn.execute(
            "SELECT t.name FROM notes_tags t JOIN notes_document_tags dt ON t.id = dt.tag_id WHERE dt.document_id = %s",
            (row[0],),
        )
        tags = [t[0] async for t in tag_rows]

        return {
            "id": row[0],
            "file_path": row[1],
            "title": row[2],
            "category": row[3],
            "description": row[4],
            "content": row[5],
            "last_modified": row[6].astimezone(UTC).isoformat() if row[6] else None,
            "git_commit_sha": row[7],
            "tags": tags,
        }



async def notes_get_all_tags() -> list[dict[str, Any]]:
    """Get all tags with document counts."""
    async with _get_connection() as conn:
        rows = await conn.execute(
            """
            SELECT t.id, t.name, COUNT(dt.document_id) as doc_count
            FROM notes_tags t
            LEFT JOIN notes_document_tags dt ON t.id = dt.tag_id
            GROUP BY t.id, t.name
            ORDER BY t.name
            """
        )
        result: list[dict[str, Any]] = []
        async for row in rows:
            result.append({
                "id": row[0],
                "name": row[1],
                "document_count": row[2],
            })
        return result


async def notes_get_all_categories() -> list[dict[str, Any]]:
    """Get all categories with document counts."""
    async with _get_connection() as conn:
        rows = await conn.execute(
            """
            SELECT c.id, c.name, COUNT(d.id) as doc_count
            FROM notes_categories c
            LEFT JOIN notes_documents d ON c.id = d.category_id
            GROUP BY c.id, c.name
            ORDER BY c.name
            """
        )
        result: list[dict[str, Any]] = []
        async for row in rows:
            result.append({
                "id": row[0],
                "name": row[1],
                "document_count": row[2],
            })
        return result


async def notes_get_category_by_name(name: str) -> dict[str, Any] | None:
    """Get a category by name."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            "SELECT id, name FROM notes_categories WHERE name = %s", (name,)
        )).fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1]}


async def notes_get_tag_by_name(name: str) -> dict[str, Any] | None:
    """Get a tag by name."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            "SELECT id, name FROM notes_tags WHERE name = %s", (name,)
        )).fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1]}


async def notes_get_last_sync_sha() -> str | None:
    """Get the most recent git commit SHA from synced documents."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            "SELECT git_commit_sha FROM notes_documents WHERE git_commit_sha IS NOT NULL ORDER BY updated_at DESC LIMIT 1"
        )).fetchone()
        if not row:
            return None
        return row[0]


async def notes_get_docs_without_embeddings() -> list[int]:
    """Get list of document IDs that don't have embeddings."""
    async with _get_connection() as conn:
        rows = await conn.execute(
            "SELECT id FROM notes_documents WHERE content_embedding IS NULL ORDER BY id"
        )
        return [row[0] async for row in rows]


async def notes_update_embeddings(doc_ids: list[int], embeddings: list) -> int:
    """Update embeddings for multiple documents. Returns count updated."""
    if not doc_ids or not embeddings or len(doc_ids) != len(embeddings):
        return 0
    now = datetime.now(UTC)
    updated = 0
    async with _get_connection() as conn:
        for doc_id, embedding in zip(doc_ids, embeddings):
            vec_str = f"[{','.join(map(str, embedding))}]"
            await conn.execute(
                "UPDATE notes_documents SET content_embedding = %s::vector, embedding_updated_at = %s WHERE id = %s",
                (vec_str, now, doc_id),
            )
            updated += 1
    return updated


async def notes_get_documents_by_ids(doc_ids: list[int]) -> list[dict[str, Any]]:
    """Get full documents by their IDs."""
    if not doc_ids:
        return []
    async with _get_connection() as conn:
        placeholders = ", ".join(["%s"] * len(doc_ids))
        rows = await conn.execute(
            f"""
            SELECT d.id, d.file_path, d.title, c.name as category, d.description, d.content, d.last_modified, d.git_commit_sha
            FROM notes_documents d
            LEFT JOIN notes_categories c ON d.category_id = c.id
            WHERE d.id IN ({placeholders})
            ORDER BY d.id
            """,
            tuple(doc_ids),
        )
        result: list[dict[str, Any]] = []
        async for row in rows:
            tag_rows = await conn.execute(
                "SELECT t.name FROM notes_tags t JOIN notes_document_tags dt ON t.id = dt.tag_id WHERE dt.document_id = %s",
                (row[0],),
            )
            tags = [t[0] async for t in tag_rows]
            result.append({
                "id": row[0],
                "file_path": row[1],
                "title": row[2],
                "category": row[3],
                "description": row[4],
                "content": row[5],
                "last_modified": row[6].astimezone(UTC).isoformat() if row[6] else None,
                "git_commit_sha": row[7],
                "tags": tags,
            })
        return result


async def notes_fulltext_search(
    query: str,
    limit: int = 20,
    offset: int = 0,
    category_id: int | None = None,
    tag_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Full-text search on notes documents using ts_rank."""
    clauses: list[str] = ["d.search_vector @@ plainto_tsquery('english', %s)"]
    params: list[Any] = [query]
    joins = ""

    if category_id is not None:
        clauses.append("d.category_id = %s")
        params.append(category_id)

    if tag_ids:
        joins = " JOIN notes_document_tags dt ON d.id = dt.document_id"
        tag_placeholders = ", ".join(["%s"] * len(tag_ids))
        clauses.append(f"dt.tag_id IN ({tag_placeholders})")
        params.extend(tag_ids)

    where = " AND ".join(clauses)

    sql = f"""
        SELECT DISTINCT d.id, d.file_path, d.title, c.name as category, d.description, d.last_modified, d.git_commit_sha,
               ts_rank(d.search_vector, plainto_tsquery('english', %s)) as rank
        FROM notes_documents d
        LEFT JOIN notes_categories c ON d.category_id = c.id
        {joins}
        WHERE {where}
        ORDER BY rank DESC
        LIMIT %s OFFSET %s
    """
    params_with_rank = [query] + params + [limit, offset]

    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(params_with_rank))
        result: list[dict[str, Any]] = []
        async for row in rows:
            tag_rows = await conn.execute(
                "SELECT t.name FROM notes_tags t JOIN notes_document_tags dt ON t.id = dt.tag_id WHERE dt.document_id = %s",
                (row[0],),
            )
            tags = [t[0] async for t in tag_rows]
            result.append({
                "id": row[0],
                "file_path": row[1],
                "title": row[2],
                "category": row[3],
                "description": row[4],
                "last_modified": row[5].astimezone(UTC).isoformat() if row[5] else None,
                "git_commit_sha": row[6],
                "rank": float(row[7]),
                "tags": tags,
            })
        return result


async def notes_vector_search(
    embedding: list[float],
    limit: int = 20,
    offset: int = 0,
    category_id: int | None = None,
    tag_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Vector similarity search on notes documents using cosine distance."""
    clauses: list[str] = ["d.content_embedding IS NOT NULL"]
    params: list[Any] = []
    joins = ""

    if category_id is not None:
        clauses.append("d.category_id = %s")
        params.append(category_id)

    if tag_ids:
        joins = " JOIN notes_document_tags dt ON d.id = dt.document_id"
        tag_placeholders = ", ".join(["%s"] * len(tag_ids))
        clauses.append(f"dt.tag_id IN ({tag_placeholders})")
        params.extend(tag_ids)

    where = " AND ".join(clauses)
    vec_str = f"[{','.join(map(str, embedding))}]"

    sql = f"""
        SELECT DISTINCT d.id, d.file_path, d.title, c.name as category, d.description, d.last_modified, d.git_commit_sha,
               1 - (d.content_embedding <=> %s::vector) as similarity,
               d.content_embedding <=> %s::vector as distance
        FROM notes_documents d
        LEFT JOIN notes_categories c ON d.category_id = c.id
        {joins}
        WHERE {where}
        ORDER BY distance
        LIMIT %s OFFSET %s
    """
    all_params = [vec_str, vec_str] + params + [limit, offset]

    async with _get_connection() as conn:
        rows = await conn.execute(sql, tuple(all_params))
        result: list[dict[str, Any]] = []
        async for row in rows:
            tag_rows = await conn.execute(
                "SELECT t.name FROM notes_tags t JOIN notes_document_tags dt ON t.id = dt.tag_id WHERE dt.document_id = %s",
                (row[0],),
            )
            tags = [t[0] async for t in tag_rows]
            result.append({
                "id": row[0],
                "file_path": row[1],
                "title": row[2],
                "category": row[3],
                "description": row[4],
                "last_modified": row[5].astimezone(UTC).isoformat() if row[5] else None,
                "git_commit_sha": row[6],
                "similarity": float(row[7]),
                "tags": tags,
            })
        return result


async def notes_get_embedding_stats() -> dict[str, Any]:
    """Get statistics about document embeddings."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            """
            SELECT
                COUNT(*) as total_docs,
                COUNT(content_embedding) as docs_with_embeddings,
                COUNT(*) - COUNT(content_embedding) as docs_without_embeddings,
                MIN(embedding_updated_at) as oldest_embedding_update,
                MAX(embedding_updated_at) as newest_embedding_update
            FROM notes_documents
            """
        )).fetchone()
        return {
            "total_docs": row[0],
            "docs_with_embeddings": row[1],
            "docs_without_embeddings": row[2],
            "oldest_embedding_update": row[3].astimezone(UTC).isoformat() if row[3] else None,
            "newest_embedding_update": row[4].astimezone(UTC).isoformat() if row[4] else None,
        }



async def sync_job_create(commit_sha: str | None, file_paths: list[str]) -> int:
    """Create a new sync job with pending items."""
    async with _get_connection(autocommit=False) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO notes_sync_jobs (status, commit_sha, total_items)
                VALUES ('pending', %s, %s)
                RETURNING id
                """,
                (commit_sha, len(file_paths)),
            )
            row = await cur.fetchone()
            job_id = row[0]

            if file_paths:
                await cur.executemany(
                    "INSERT INTO notes_sync_job_items (job_id, file_path) VALUES (%s, %s)",
                    [(job_id, path) for path in file_paths],
                )

            await conn.commit()
            return job_id


async def sync_job_get(job_id: int) -> dict[str, Any] | None:
    """Get a sync job by ID with item counts."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            """
            SELECT id, status, commit_sha, total_items, completed_items, failed_items,
                   created_at, started_at, completed_at, error_message, rate_limit_reset_at,
                   last_activity_at
            FROM notes_sync_jobs WHERE id = %s
            """,
            (job_id,),
        )).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "status": row[1],
            "commit_sha": row[2],
            "total_items": row[3],
            "completed_items": row[4],
            "failed_items": row[5],
            "created_at": row[6].astimezone(UTC).isoformat() if row[6] else None,
            "started_at": row[7].astimezone(UTC).isoformat() if row[7] else None,
            "completed_at": row[8].astimezone(UTC).isoformat() if row[8] else None,
            "error_message": row[9],
            "rate_limit_reset_at": row[10].astimezone(UTC).isoformat() if row[10] else None,
            "last_activity_at": row[11].astimezone(UTC).isoformat() if row[11] else None,
        }


async def sync_job_list(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    """List sync jobs, optionally filtered by status."""
    async with _get_connection() as conn:
        if status:
            rows = await conn.execute(
                "SELECT id, status, commit_sha, total_items, completed_items, failed_items, created_at, completed_at FROM notes_sync_jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                (status, limit),
            )
        else:
            rows = await conn.execute(
                "SELECT id, status, commit_sha, total_items, completed_items, failed_items, created_at, completed_at FROM notes_sync_jobs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        result = []
        async for row in rows:
            result.append({
                "id": row[0],
                "status": row[1],
                "commit_sha": row[2],
                "total_items": row[3],
                "completed_items": row[4],
                "failed_items": row[5],
                "created_at": row[6].astimezone(UTC).isoformat() if row[6] else None,
                "completed_at": row[7].astimezone(UTC).isoformat() if row[7] else None,
            })
        return result


async def sync_job_update_status(
    job_id: int,
    status: str,
    error_message: str | None = None,
    rate_limit_reset_at: datetime | None = None,
) -> None:
    """Update the status of a sync job."""
    async with _get_connection() as conn:
        now = datetime.now(UTC)
        if status == "running":
            await conn.execute(
                "UPDATE notes_sync_jobs SET status = %s, started_at = %s, last_activity_at = %s WHERE id = %s",
                (status, now, now, job_id),
            )
        elif status in ("completed", "failed"):
            await conn.execute(
                "UPDATE notes_sync_jobs SET status = %s, completed_at = %s, error_message = %s, last_activity_at = %s WHERE id = %s",
                (status, now, error_message, now, job_id),
            )
        elif status == "paused":
            await conn.execute(
                "UPDATE notes_sync_jobs SET status = %s, rate_limit_reset_at = %s, last_activity_at = %s WHERE id = %s",
                (status, rate_limit_reset_at, now, job_id),
            )
        else:
            await conn.execute(
                "UPDATE notes_sync_jobs SET status = %s, last_activity_at = %s WHERE id = %s",
                (status, now, job_id),
            )


async def sync_job_update_counts(job_id: int) -> None:
    """Update the completed/failed counts for a job from its items."""
    async with _get_connection() as conn:
        await conn.execute(
            """
            UPDATE notes_sync_jobs SET
                completed_items = (SELECT COUNT(*) FROM notes_sync_job_items WHERE job_id = %s AND status = 'success'),
                failed_items = (SELECT COUNT(*) FROM notes_sync_job_items WHERE job_id = %s AND status = 'failed'),
                last_activity_at = NOW()
            WHERE id = %s
            """,
            (job_id, job_id, job_id),
        )


async def sync_job_get_pending_items(job_id: int, limit: int = 50) -> list[dict[str, Any]]:
    """Get pending items for a sync job."""
    async with _get_connection() as conn:
        rows = await conn.execute(
            "SELECT id, file_path, retry_count FROM notes_sync_job_items WHERE job_id = %s AND status = 'pending' ORDER BY id LIMIT %s",
            (job_id, limit),
        )
        return [{"id": row[0], "file_path": row[1], "retry_count": row[2]} async for row in rows]


async def sync_job_get_failed_items(job_id: int) -> list[dict[str, Any]]:
    """Get failed items for a sync job (for retry)."""
    async with _get_connection() as conn:
        rows = await conn.execute(
            "SELECT id, file_path, retry_count, last_error FROM notes_sync_job_items WHERE job_id = %s AND status = 'failed' ORDER BY id",
            (job_id,),
        )
        return [{"id": row[0], "file_path": row[1], "retry_count": row[2], "last_error": row[3]} async for row in rows]


async def sync_job_item_update(
    item_id: int,
    status: str,
    error: str | None = None,
) -> None:
    """Update the status of a sync job item."""
    async with _get_connection() as conn:
        now = datetime.now(UTC)
        if status == "success":
            await conn.execute(
                "UPDATE notes_sync_job_items SET status = %s, completed_at = %s, last_attempt_at = %s WHERE id = %s",
                (status, now, now, item_id),
            )
        elif status == "failed":
            await conn.execute(
                "UPDATE notes_sync_job_items SET status = %s, last_error = %s, last_attempt_at = %s, retry_count = retry_count + 1 WHERE id = %s",
                (status, error, now, item_id),
            )
        else:
            await conn.execute(
                "UPDATE notes_sync_job_items SET status = %s, last_attempt_at = %s WHERE id = %s",
                (status, now, item_id),
            )


async def sync_job_reset_failed_items(job_id: int, max_retries: int = 5) -> int:
    """Reset failed items to pending for retry (up to max_retries)."""
    async with _get_connection() as conn:
        result = await conn.execute(
            "UPDATE notes_sync_job_items SET status = 'pending' WHERE job_id = %s AND status = 'failed' AND retry_count < %s",
            (job_id, max_retries),
        )
        return result.rowcount


async def sync_job_get_resumable() -> dict[str, Any] | None:
    """Get the most recent job that can be resumed (paused or running with pending items)."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            """
            SELECT j.id, j.status, j.commit_sha, j.rate_limit_reset_at
            FROM notes_sync_jobs j
            WHERE j.status IN ('paused', 'running', 'pending')
            AND EXISTS (SELECT 1 FROM notes_sync_job_items i WHERE i.job_id = j.id AND i.status = 'pending')
            ORDER BY j.created_at DESC
            LIMIT 1
            """,
        )).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "status": row[1],
            "commit_sha": row[2],
            "rate_limit_reset_at": row[3].astimezone(UTC).isoformat() if row[3] else None,
        }


async def sync_job_get_all_completed_paths(job_id: int) -> list[str]:
    """Get all successfully completed file paths for a job."""
    async with _get_connection() as conn:
        rows = await conn.execute(
            "SELECT file_path FROM notes_sync_job_items WHERE job_id = %s AND status = 'success'",
            (job_id,),
        )
        return [row[0] async for row in rows]


async def sync_job_get_skipped_count(job_id: int) -> int:
    """Get count of skipped items (exceeded max retries) for a job."""
    async with _get_connection() as conn:
        row = await (await conn.execute(
            "SELECT COUNT(*) FROM notes_sync_job_items WHERE job_id = %s AND status = 'skipped'",
            (job_id,),
        )).fetchone()
        return row[0] if row else 0


async def sync_job_list_all_failed_items(
    limit: int = 100,
    job_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    List all failed items across jobs (dead letter queue).

    Args:
        limit: Maximum number of items to return
        job_id: Optional filter by specific job

    Returns:
        List of failed items with job info
    """
    async with _get_connection() as conn:
        if job_id:
            rows = await conn.execute(
                """
                SELECT i.id, i.job_id, i.file_path, i.status, i.retry_count,
                       i.last_error, i.last_attempt_at, i.created_at,
                       j.commit_sha
                FROM notes_sync_job_items i
                JOIN notes_sync_jobs j ON j.id = i.job_id
                WHERE i.job_id = %s AND i.status IN ('failed', 'skipped')
                ORDER BY i.last_attempt_at DESC NULLS LAST
                LIMIT %s
                """,
                (job_id, limit),
            )
        else:
            rows = await conn.execute(
                """
                SELECT i.id, i.job_id, i.file_path, i.status, i.retry_count,
                       i.last_error, i.last_attempt_at, i.created_at,
                       j.commit_sha
                FROM notes_sync_job_items i
                JOIN notes_sync_jobs j ON j.id = i.job_id
                WHERE i.status IN ('failed', 'skipped')
                ORDER BY i.last_attempt_at DESC NULLS LAST
                LIMIT %s
                """,
                (limit,),
            )

        result = []
        async for row in rows:
            result.append({
                "id": row[0],
                "job_id": row[1],
                "file_path": row[2],
                "status": row[3],
                "retry_count": row[4],
                "last_error": row[5],
                "last_attempt_at": row[6].astimezone(UTC).isoformat() if row[6] else None,
                "created_at": row[7].astimezone(UTC).isoformat() if row[7] else None,
                "commit_sha": row[8],
            })
        return result


async def sync_job_item_reset_to_pending(item_id: int) -> bool:
    """
    Reset a specific failed/skipped item to pending for manual retry.

    Returns:
        True if item was reset, False if not found or not in failed/skipped status
    """
    async with _get_connection() as conn:
        result = await conn.execute(
            """
            UPDATE notes_sync_job_items
            SET status = 'pending', last_error = NULL
            WHERE id = %s AND status IN ('failed', 'skipped')
            """,
            (item_id,),
        )
        return result.rowcount > 0


async def sync_job_item_skip(item_id: int, reason: str | None = None) -> bool:
    """
    Permanently skip a failed item (mark as unrecoverable).

    Args:
        item_id: The item ID to skip
        reason: Optional reason for skipping

    Returns:
        True if item was skipped, False if not found
    """
    async with _get_connection() as conn:
        error_msg = reason or "Manually skipped"
        result = await conn.execute(
            """
            UPDATE notes_sync_job_items
            SET status = 'skipped', last_error = %s, last_attempt_at = NOW()
            WHERE id = %s AND status IN ('pending', 'failed')
            """,
            (error_msg, item_id),
        )
        return result.rowcount > 0


async def sync_job_item_delete(item_id: int) -> bool:
    """
    Delete a sync job item entirely (removes from job).

    Use with caution - this removes the item from tracking entirely.
    The file won't be synced unless a new job is created.

    Returns:
        True if item was deleted, False if not found
    """
    async with _get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM notes_sync_job_items WHERE id = %s",
            (item_id,),
        )
        return result.rowcount > 0


async def sync_job_reset_all_failed(job_id: int, include_skipped: bool = False) -> int:
    """
    Reset all failed items in a job to pending for retry.

    Args:
        job_id: The job ID
        include_skipped: If True, also reset skipped items

    Returns:
        Number of items reset
    """
    async with _get_connection() as conn:
        if include_skipped:
            result = await conn.execute(
                """
                UPDATE notes_sync_job_items
                SET status = 'pending', retry_count = 0, last_error = NULL
                WHERE job_id = %s AND status IN ('failed', 'skipped')
                """,
                (job_id,),
            )
        else:
            result = await conn.execute(
                """
                UPDATE notes_sync_job_items
                SET status = 'pending', retry_count = 0, last_error = NULL
                WHERE job_id = %s AND status = 'failed'
                """,
                (job_id,),
            )
        return result.rowcount