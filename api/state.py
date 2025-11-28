from typing import Optional, Dict
import redis.asyncio as redis
import geoip2.database
from api.bus import EventBus
import asyncio

# Global runtime state initialized in main.lifespan
redis_client: Optional[redis.Redis] = None
event_bus: Optional[EventBus] = None
geoip_reader: Optional[geoip2.database.Reader] = None

# Diagnostics and connection tracking
active_ws_visitors_by_ip: Dict[str, int] = {}
ws_visitors_lock: asyncio.Lock = asyncio.Lock()


