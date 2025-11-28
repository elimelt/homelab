import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import psycopg
from psycopg_pool import AsyncConnectionPool

POOL: Optional[AsyncConnectionPool] = None


def _dsn_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "postgres"))
    port = int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")))
    user = os.getenv("POSTGRES_USER", os.getenv("PGUSER", "devuser"))
    password = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", ""))
    dbname = os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "devdb"))
    return f"host={host} port={port} user={user} password={password} dbname={dbname} sslmode=disable"


async def init_pool() -> None:
    global POOL
    if POOL is not None:
        return
    dsn = _dsn_from_env()
    POOL = AsyncConnectionPool(conninfo=dsn, min_size=1, max_size=5, open=False, kwargs={"autocommit": True})
    await POOL.open()
    await _ensure_schema()


async def close_pool() -> None:
    global POOL
    if POOL:
        await POOL.close()
        POOL = None


async def _ensure_schema() -> None:
    assert POOL is not None
    async with POOL.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                channel TEXT NOT NULL,
                sender TEXT NOT NULL,
                text TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL
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


async def insert_chat_message(channel: str, sender: str, text: str, ts_iso: str) -> None:
    assert POOL is not None
    # Parse ISO ts (trusted from server build) to timestamptz
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    async with POOL.connection() as conn:
        await conn.execute(
            "INSERT INTO chat_messages (channel, sender, text, ts) VALUES (%s, %s, %s, %s)",
            (channel, sender, text, ts),
        )


async def insert_event(topic: str, event_type: str, payload: dict, ts_iso: str) -> None:
    assert POOL is not None
    ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    async with POOL.connection() as conn:
        await conn.execute(
            "INSERT INTO events (topic, type, ts, payload) VALUES (%s, %s, %s, %s)",
            (topic, event_type, ts, payload),
        )


async def fetch_chat_history(channel: str, before_iso: Optional[str], limit: int) -> List[Dict[str, Any]]:
    assert POOL is not None
    before_ts = (
        datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        if before_iso
        else datetime.now(timezone.utc)
    )
    async with POOL.connection() as conn:
        rows = await conn.execute(
            "SELECT channel, sender, text, ts FROM chat_messages WHERE channel=%s AND ts < %s ORDER BY ts DESC LIMIT %s",
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
    assert POOL is not None
    before_ts = (
        datetime.fromisoformat(before_iso.replace("Z", "+00:00"))
        if before_iso
        else datetime.now(timezone.utc)
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
    async with POOL.connection() as conn:
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


