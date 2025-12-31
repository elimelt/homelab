import os
import secrets
import string
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import psycopg
from psycopg import errors as pg_errors

def _dsn_from_env() -> str:
    host = os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "postgres"))
    port = int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")))
    user = os.getenv("POSTGRES_USER", os.getenv("PGUSER", "devuser"))
    password = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", ""))
    dbname = os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "devdb"))
    return f"host={host} port={port} user={user} password={password} dbname={dbname} sslmode=disable"


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
            await conn.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ NULL;")
            await conn.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS message_id TEXT NULL;")
            await conn.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS reply_to TEXT NULL;")
        except Exception:
            pass
        try:
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_chat_message_id ON chat_messages (message_id) WHERE message_id IS NOT NULL;")
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


async def insert_chat_message(channel: str, sender: str, text: str, ts_iso: str, message_id: Optional[str] = None, reply_to: Optional[str] = None) -> None:
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


def _generate_event_id(length: int = 10) -> str:
    chars = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


async def w2m_create_event(
    name: str,
    dates: List[str],
    time_slots: List[str],
    description: Optional[str] = None,
    creator_name: Optional[str] = None,
) -> Dict[str, Any]:
    dsn = _dsn_from_env()
    now = datetime.now(timezone.utc)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        for _ in range(10):
            event_id = _generate_event_id()
            try:
                await conn.execute(
                    """INSERT INTO w2m_events (id, name, description, dates, time_slots, created_at, creator_name)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (event_id, name, description, json.dumps(dates), json.dumps(time_slots), now, creator_name),
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


async def w2m_get_event(event_id: str) -> Optional[Dict[str, Any]]:
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
            "created_at": row[5].astimezone(timezone.utc).isoformat(),
            "creator_name": row[6],
        }


async def w2m_get_availabilities(event_id: str) -> List[Dict[str, Any]]:
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        rows = await conn.execute(
            "SELECT participant_name, available_slots, created_at, updated_at FROM w2m_availabilities WHERE event_id = %s",
            (event_id,),
        )
        result = []
        async for row in rows:
            result.append({
                "participant_name": row[0],
                "available_slots": row[1],
                "created_at": row[2].astimezone(timezone.utc).isoformat(),
                "updated_at": row[3].astimezone(timezone.utc).isoformat(),
            })
        return result


async def w2m_get_availability(event_id: str, participant_name: str) -> Optional[Dict[str, Any]]:
    dsn = _dsn_from_env()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        row = await (await conn.execute(
            "SELECT available_slots, password_hash, created_at, updated_at FROM w2m_availabilities WHERE event_id = %s AND participant_name = %s",
            (event_id, participant_name),
        )).fetchone()
        if not row:
            return None
        return {
            "available_slots": row[0],
            "password_hash": row[1],
            "created_at": row[2].astimezone(timezone.utc).isoformat(),
            "updated_at": row[3].astimezone(timezone.utc).isoformat(),
        }


async def w2m_upsert_availability(event_id: str, participant_name: str, available_slots: List[str], password_hash: Optional[str] = None) -> Dict[str, Any]:
    dsn = _dsn_from_env()
    now = datetime.now(timezone.utc)
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
