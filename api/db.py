import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import psycopg
from psycopg import errors as pg_errors

# Note: We avoid a global AsyncConnectionPool because pools/locks are loop-bound
# and Starlette/Uvicorn can bind startup and request handling to different loops.
# Instead, we create short-lived async connections per operation.


def _dsn_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "postgres"))
    port = int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")))
    user = os.getenv("POSTGRES_USER", os.getenv("PGUSER", "devuser"))
    password = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", ""))
    dbname = os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "devdb"))
    return f"host={host} port={port} user={user} password={password} dbname={dbname} sslmode=disable"


async def init_pool() -> None:
    # Backwards-compatible no-op initializer that ensures schema exists.
    await _ensure_schema()


async def close_pool() -> None:
    # No global pool to close
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
        # Backfill schema alterations if table existed without columns
        try:
            await conn.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL;")
            await conn.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS message_id TEXT NULL;")
            await conn.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS reply_to TEXT NULL;")
        except Exception:
            pass
        # Ensure index exists (after columns are present)
        try:
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_message_id ON chat_messages (message_id) WHERE message_id IS NOT NULL;")
        except Exception:
            pass


async def insert_chat_message(channel: str, sender: str, text: str, ts_iso: str, message_id: Optional[str] = None, reply_to: Optional[str] = None) -> None:
    # Parse ISO ts (trusted from server build) to timestamptz
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
            (topic, event_type, ts, payload),
        )


async def fetch_chat_history(channel: str, before_iso: Optional[str], limit: int) -> List[Dict[str, Any]]:
    before_ts = (
        datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        if before_iso
        else datetime.now(timezone.utc)
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
                        "timestamp": ts.astimezone(timezone.utc).isoformat(),
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
                    "timestamp": ts.astimezone(timezone.utc).isoformat(),
                }
            )
        return result


async def fetch_events(
    topic: Optional[str],
    event_type: Optional[str],
    before_iso: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    before_ts = (
        datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        if before_iso
        else datetime.now(timezone.utc)
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
        out: List[Dict[str, Any]] = []
        async for row in rows:
            topic_v, type_v, ts, payload = row
            out.append(
                {
                    "topic": topic_v,
                    "type": type_v,
                    "timestamp": ts.astimezone(timezone.utc).isoformat(),
                    "payload": payload,
                }
            )
        return out


async def soft_delete_chat_history(channel: Optional[str] = None, before_iso: Optional[str] = None) -> int:
    """
    Soft-delete chat messages by setting deleted_at, filtering them out from history.
    - If channel is provided, restrict to that channel; otherwise all channels.
    - If before_iso is provided, delete only messages with ts < before; otherwise all.
    Returns number of rows affected (best effort).
    """
    dsn = _dsn_from_env()
    conditions: List[str] = ["deleted_at IS NULL"]
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
    now_ts = datetime.now(timezone.utc)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        rows = await conn.execute(sql, tuple([now_ts] + params))
        count = 0
        async for _ in rows:
            count += 1
        return count


