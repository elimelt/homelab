from fastapi import APIRouter, Query
from typing import Optional, Dict, Any

from api import db

router = APIRouter()


@router.get("/events")
async def events_history(
    topic: Optional[str] = Query(None, description="Event topic (e.g., visitor_updates, chat:general)"),
    type: Optional[str] = Query(None, description="Event type (e.g., join, leave, chat_message)"),
    before: Optional[str] = Query(None, description="ISO8601 timestamp; default now"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    if db.POOL is None:
        return {"error": "Event persistence disabled", "events": []}
    events = await db.fetch_events(topic, type, before, limit)
    next_before = events[-1]["timestamp"] if events else before
    return {"events": events, "next_before": next_before}


