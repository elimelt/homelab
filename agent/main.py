import os
import time
import random
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import redis
import psycopg
from google import genai


def env(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v


def build_pg_conninfo() -> str:
    host = env("POSTGRES_HOST", "postgres")
    port = env("POSTGRES_PORT", "5432")
    user = env("POSTGRES_USER", "devuser")
    password = env("POSTGRES_PASSWORD", "")
    dbname = env("POSTGRES_DB", "devdb")
    return f"host={host} port={port} user={user} password={password} dbname={dbname} sslmode=disable"


def fetch_recent_messages(conn, channel: str, since: datetime, limit: int) -> List[Tuple[str, str, datetime]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT sender, text, ts
            FROM chat_messages
            WHERE channel=%s AND ts >= %s
            ORDER BY ts ASC
            LIMIT %s
            """,
            (channel, since, limit),
        )
        return cur.fetchall()


def insert_chat_message(conn, channel: str, sender: str, text: str, ts: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_messages (channel, sender, text, ts) VALUES (%s, %s, %s, %s)",
            (channel, sender, text, ts),
        )


def insert_event(conn, topic: str, event_type: str, payload: dict, ts: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO events (topic, type, ts, payload) VALUES (%s, %s, %s, %s)",
            (topic, event_type, ts, payload),
        )


def publish_to_redis(r: redis.Redis, channel: str, sender: str, text: str, ts: datetime) -> None:
    payload = {
        "type": "chat_message",
        "channel": channel,
        "sender": sender,
        "text": text,
        "timestamp": ts.astimezone(timezone.utc).isoformat(),
    }
    r.publish(f"chat:{channel}", __import__("json").dumps(payload))


def build_prompt(channel: str, history: List[Tuple[str, str, datetime]]) -> str:
    lines = [f"You are an autonomous agent participating in the #{channel} channel."]
    lines.append("You log on occasionally, read the last day's messages, optionally respond 1-3 times, then log off.")
    lines.append("Keep messages brief and relevant. If no reply is appropriate, say nothing.")
    lines.append("")
    lines.append("Recent messages (oldest first):")
    for sender, text, ts in history[-200:]:
        ts_str = ts.astimezone(timezone.utc).isoformat()
        lines.append(f"[{ts_str}] {sender}: {text}")
    lines.append("")
    lines.append("Now produce your next message. Only return the message text (no prefixes).")
    return "\n".join(lines)


def main():
    # Config
    channels = [c.strip() for c in env("AGENT_CHANNELS", "general").split(",") if c.strip()]
    min_sleep = int(env("AGENT_MIN_SLEEP_SEC", "60"))
    max_sleep = int(env("AGENT_MAX_SLEEP_SEC", "300"))
    max_replies = int(env("AGENT_MAX_REPLIES", "3"))
    history_hours = int(env("AGENT_HISTORY_HOURS", "24"))
    model = env("AGENT_MODEL", "gemini-2.5-flash")
    sender = env("AGENT_SENDER", "agent:gemini")

    # Clients
    r = redis.Redis(
        host=env("REDIS_HOST", "redis"),
        port=int(env("REDIS_PORT", "6379")),
        password=env("REDIS_PASSWORD", "") or None,
        decode_responses=True,
    )
    pg = psycopg.connect(build_pg_conninfo(), autocommit=True)
    ai = genai.Client()

    # Loop forever with random sessions
    while True:
        time.sleep(random.randint(min_sleep, max_sleep))

        # Randomly pick a channel
        channel = random.choice(channels)
        since = datetime.now(timezone.utc) - timedelta(hours=history_hours)
        history = fetch_recent_messages(pg, channel, since, limit=2000)

        # Decide number of replies (0..max)
        n_replies = random.randint(0, max_replies)
        if n_replies == 0:
            continue

        context = build_prompt(channel, history)
        # Multi-turn: send message repeatedly, appending agent outputs for continuity
        running_context = context
        for _ in range(n_replies):
            try:
                resp = ai.models.generate_content(model=model, contents=running_context)
                text = (resp.text or "").strip()
                if not text:
                    break
                now_ts = datetime.now(timezone.utc)
                # Publish to Redis for realtime
                publish_to_redis(r, channel, sender, text, now_ts)
                # Persist to DB (chat_messages + events)
                insert_chat_message(pg, channel, sender, text, now_ts)
                insert_event(pg, f"chat:{channel}", "chat_message", {
                    "type": "chat_message",
                    "channel": channel,
                    "sender": sender,
                    "text": text,
                    "timestamp": now_ts.astimezone(timezone.utc).isoformat(),
                }, now_ts)
                # Extend context for potential next reply
                running_context += f"\n[{now_ts.isoformat()}] {sender}: {text}"
                # Small delay between replies for realism
                time.sleep(random.randint(1, 5))
            except Exception:
                # Fail closed; wait until next session tick
                break


if __name__ == "__main__":
    main()


