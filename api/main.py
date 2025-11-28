from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import redis.asyncio as redis
import geoip2.database
from typing import Optional
from api.bus import EventBus
from api import state
from contextlib import asynccontextmanager
from api.controllers.health import router as health_router
from api.controllers.example import router as example_router
from api.controllers.cache import router as cache_router
from api.controllers.visitors import router as visitors_router
from api.controllers.system import router as system_router
from api.controllers.ws_visitors import router as ws_visitors_router
from api.controllers.ws_chat import router as ws_chat_router
from api.controllers.chat_history import router as chat_history_router
from api.controllers.events_history import router as events_history_router
from api import db

app = FastAPI(title="DevStack Public API", version="1.0.0")

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
cors_regex = os.getenv("CORS_ORIGINS_REGEX", "").strip() or None
allow_credentials = cors_origins != ["*"] and cors_regex is None

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = None
event_bus: Optional[EventBus] = None
geoip_reader = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global redis_client, geoip_reader, event_bus
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "")

    redis_client = await redis.Redis(
        host=redis_host,
        port=redis_port,
        password=redis_password if redis_password else None,
        decode_responses=True
    )
    event_bus = EventBus(redis_client)

    geoip_db_path = os.getenv("GEOIP_DB_PATH", "/app/GeoLite2-City.mmdb")
    if os.path.exists(geoip_db_path):
        geoip_reader = geoip2.database.Reader(geoip_db_path)

    # Mirror into shared state for controllers
    state.redis_client = redis_client
    state.event_bus = event_bus
    state.geoip_reader = geoip_reader

    # Optional DB: enable with ENABLE_CHAT_DB=1
    enable_db = os.getenv("ENABLE_CHAT_DB", "0") == "1"
    if enable_db:
        try:
            await db.init_pool()
        except Exception:
            # Continue without DB
            pass

    try:
        yield
    finally:
        if enable_db:
            try:
                await db.close_pool()
            except Exception:
                pass
        if redis_client:
            await redis_client.aclose()
        if geoip_reader:
            geoip_reader.close()
        state.redis_client = None
        state.event_bus = None
        state.geoip_reader = None

app.router.lifespan_context = lifespan

# Mount controllers (HTTP + WS)
app.include_router(health_router)
app.include_router(example_router)
app.include_router(cache_router)
app.include_router(visitors_router)
app.include_router(system_router)
app.include_router(ws_visitors_router)
app.include_router(ws_chat_router)
app.include_router(chat_history_router)
app.include_router(events_history_router)

