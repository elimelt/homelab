from datetime import datetime, timezone
from typing import Dict

from api.bus import EventBus
from api.events import ChatMessageEvent
from api import db


def build_chat_message(channel: str, sender: str, text: str) -> ChatMessageEvent:
    return {
        "type": "chat_message",
        "channel": channel,
        "sender": sender,
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def publish_chat_message(event_bus: EventBus, channel: str, event: ChatMessageEvent) -> None:
    await event_bus.publish_chat(channel, event)
    # Persist to DB if enabled and pool is initialized
    try:
        if db.POOL is not None:
            await db.insert_chat_message(channel, event["sender"], event["text"], event["timestamp"])
            from api.bus import EventBus as _Bus  # local import to avoid cycle at import time
            await db.insert_event(_Bus.chat_channel(channel), "chat_message", event, event["timestamp"])
    except Exception:
        # Don't break real-time if storage fails
        pass


