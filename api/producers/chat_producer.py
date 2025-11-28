from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
import secrets
import re

from api.bus import EventBus
from api.events import ChatMessageEvent
from api import db


_REPLY_RE = re.compile(r"^\s*REPLY:\s*([A-Za-z0-9_-]{4,32})\s*\n", re.IGNORECASE)


def _gen_message_id() -> str:
    return secrets.token_hex(4)


def _parse_reply_header(text: str) -> Tuple[Optional[str], str]:
    m = _REPLY_RE.match(text or "")
    if not m:
        return None, text
    reply_to = m.group(1)
    cleaned = text[m.end():]
    return reply_to, cleaned


def build_chat_message(channel: str, sender: str, text: str) -> ChatMessageEvent:
    reply_to, cleaned = _parse_reply_header(text)
    mid = _gen_message_id()
    return {
        "type": "chat_message",
        "channel": channel,
        "sender": sender,
        "text": cleaned,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": mid,
        "reply_to": reply_to,
    }


async def publish_chat_message(event_bus: EventBus, channel: str, event: ChatMessageEvent) -> None:
    await event_bus.publish_chat(channel, event)
    # Persist to DB if enabled and pool is initialized
    try:
        await db.insert_chat_message(channel, event["sender"], event["text"], event["timestamp"], message_id=event.get("id"), reply_to=event.get("reply_to"))
        from api.bus import EventBus as _Bus  # local import to avoid cycle at import time
        await db.insert_event(_Bus.chat_channel(channel), "chat_message", event, event["timestamp"])
    except Exception:
        # Don't break real-time if storage fails
        pass


