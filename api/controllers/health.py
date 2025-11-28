from fastapi import APIRouter
from typing import Dict

from api import state

router = APIRouter()


@router.get("/health")
async def health() -> Dict[str, str]:
    redis_status = "disconnected"
    if state.redis_client:
        try:
            await state.redis_client.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "unhealthy"

    return {"status": "ok", "redis": redis_status}


