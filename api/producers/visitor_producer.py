from datetime import datetime, timezone
import json
from typing import Dict, Any

from api.bus import EventBus
import redis.asyncio as redis
import geoip2.database
from api import db


def get_location(geoip_reader: geoip2.database.Reader | None, ip: str):
    default = {"country": "Unknown", "city": "Unknown", "lat": None, "lon": None}
    if not geoip_reader:
        return default
    try:
        response = geoip_reader.city(ip)
        return {
            "country": response.country.name or "Unknown",
            "city": response.city.name or "Unknown",
            "lat": response.location.latitude,
            "lon": response.location.longitude,
        }
    except Exception:
        return default


async def join_visitor(
    redis_client: redis.Redis, event_bus: EventBus | None, geoip_reader, client_ip: str, visitor_key: str
) -> Dict[str, Any]:
    location = get_location(geoip_reader, client_ip)
    visitor_data = {
        "ip": client_ip,
        "location": location,
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    await redis_client.setex(visitor_key, 30, json.dumps(visitor_data))
    visit_entry = {
        "ip": client_ip,
        "location": location,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await redis_client.lpush("visit_log", json.dumps(visit_entry))
    await redis_client.ltrim("visit_log", 0, 999)
    if event_bus:
        await event_bus.publish({"type": "join", "visitor": visitor_data})
    try:
        if db.POOL is not None:
            await db.insert_event("visitor_updates", "join", {"visitor": visitor_data}, visitor_data["connected_at"])
    except Exception:
        pass
    return visitor_data


async def heartbeat(redis_client: redis.Redis, visitor_key: str, visitor_data: dict) -> None:
    await redis_client.setex(visitor_key, 30, json.dumps(visitor_data))


async def leave_visitor(redis_client: redis.Redis, event_bus: EventBus | None, client_ip: str, visitor_key: str) -> None:
    await redis_client.delete(visitor_key)
    if event_bus:
        await event_bus.publish({"type": "leave", "ip": client_ip})
    try:
        if db.POOL is not None:
            from datetime import datetime, timezone
            await db.insert_event(
                "visitor_updates", "leave", {"ip": client_ip}, datetime.now(timezone.utc).isoformat()
            )
    except Exception:
        pass


