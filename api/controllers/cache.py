from fastapi import APIRouter
from typing import Dict, Optional
from pydantic import BaseModel

from api import state


class CacheValue(BaseModel):
    value: str
    ttl: Optional[int] = 3600


router = APIRouter()


@router.get("/cache/{key}")
async def get_cache(key: str):
    if not state.redis_client:
        return {"error": "Redis not connected"}

    value = await state.redis_client.get(key)
    if value is None:
        return {"key": key, "value": None, "found": False}

    return {"key": key, "value": value, "found": True}


@router.post("/cache/{key}")
async def set_cache(key: str, data: CacheValue):
    if not state.redis_client:
        return {"error": "Redis not connected"}

    await state.redis_client.setex(key, data.ttl, data.value)
    return {"key": key, "value": data.value, "ttl": data.ttl, "success": True}


