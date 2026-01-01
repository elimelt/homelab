import json
import os
import secrets
import string
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg import errors as pg_errors
from psycopg.types.json import Json


def _dsn_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "postgres"))
    port = int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")))
    user = os.getenv("POSTGRES_USER", os.getenv("PGUSER", "devuser"))
    password = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", ""))
    dbname = os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "devdb"))
    return (
        f"host={host} port={port} user={user} password={password} dbname={dbname} sslmode=disable"
    )


async def init_pool() -> None:
    await _ensure_schema()


async def close_pool() -> None:
    return


async def _ensure_schema() -> None:
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
        # Visitor analytics tables
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


async def insert_chat_message(
    channel: str,
    sender: str,
    text: str,
    ts_iso: str,
    message_id: str | None = None,
    reply_to: str | None = None,
) -> None:
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
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
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
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
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    now = datetime.now(UTC)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    now = datetime.now(UTC)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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


# ─────────────────────────────────────────────────────────────────────────────
# Visitor Analytics Functions
# ─────────────────────────────────────────────────────────────────────────────


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
    dsn = _dsn_from_env()
    now = datetime.now(UTC)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
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

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
    sql = """
        SELECT ts, type, payload
        FROM events
        WHERE topic = 'visitor_updates'
          AND type IN ('join', 'leave')
          AND ts >= %s AND ts < %s
        ORDER BY ts ASC
    """
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
    dsn = _dsn_from_env()
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

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
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
