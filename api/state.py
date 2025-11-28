from typing import Optional
import redis.asyncio as redis
import geoip2.database
from api.bus import EventBus

# Global runtime state initialized in main.lifespan
redis_client: Optional[redis.Redis] = None
event_bus: Optional[EventBus] = None
geoip_reader: Optional[geoip2.database.Reader] = None


