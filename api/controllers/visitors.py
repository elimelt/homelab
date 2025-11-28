from fastapi import APIRouter
import json

from api import state

router = APIRouter()


@router.get("/visitors")
async def get_visitors():
    if not state.redis_client:
        return {"error": "Redis not connected"}

    visitor_keys = await state.redis_client.keys("visitor:*")
    active_visitors = [
        json.loads(data)
        for key in visitor_keys
        if (data := await state.redis_client.get(key))
    ]

    visit_log = await state.redis_client.lrange("visit_log", 0, 99)
    visits = [json.loads(v) for v in visit_log]

    return {
        "active_count": len(active_visitors),
        "active_visitors": active_visitors,
        "recent_visits": visits,
    }


