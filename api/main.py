import os
import logging
import redis.asyncio as redis
import geoip2.database

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
from typing import Optional
from collections.abc import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.bus import EventBus
from api import state
from contextlib import asynccontextmanager
from redis.asyncio import BlockingConnectionPool as RedisConnectionPool
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
from api.agents.gemini_agent import start_agents
import asyncio
from api.middleware import HTTPLogMiddleware
from api.redis_debug import wrap_redis_client
from api.controllers.chat_admin import router as chat_admin_router
from api.controllers.when2meet import router as when2meet_router

app = FastAPI(title="DevStack Public API", version="1.0.0")

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
cors_regex_env = os.getenv("CORS_ORIGINS_REGEX", "").strip()
# Allow elimelt.com and any subdomain (http and https)
elimelt_subdomain_regex = r"https?://([a-zA-Z0-9-]+\.)?elimelt\.com"
cors_regex = cors_regex_env if cors_regex_env else elimelt_subdomain_regex
allow_credentials = cors_origins != ["*"] and cors_regex is None

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_regex,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.getenv("REQUEST_DEBUG", "0") == "1":
    logging.getLogger("api.http").setLevel(logging.DEBUG)
    app.add_middleware(HTTPLogMiddleware)

if os.getenv("WS_DEBUG", "0") == "1":
    logging.getLogger("api.ws.visitors").setLevel(logging.INFO)
    logging.getLogger("api.ws.chat").setLevel(logging.INFO)

redis_client = None
event_bus: Optional[EventBus] = None
geoip_reader = None

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global redis_client, geoip_reader, event_bus
    stop_event: Optional[asyncio.Event] = None
    agent_tasks: list[asyncio.Task] = []
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_password = os.getenv("REDIS_PASSWORD", "")

    max_redis_conns = int(os.getenv("REDIS_MAX_CONNECTIONS", "200"))
    pool_timeout = float(os.getenv("REDIS_POOL_TIMEOUT_SEC", "5"))
    redis_pool = RedisConnectionPool(
        host=redis_host,
        port=redis_port,
        password=redis_password if redis_password else None,
        max_connections=max_redis_conns,
        timeout=pool_timeout,
    )
    candidate_client = redis.Redis(connection_pool=redis_pool, decode_responses=True)
    if hasattr(candidate_client, "__await__"):
        redis_client = await candidate_client  # type: ignore[assignment]
    else:
        redis_client = candidate_client
    if os.getenv("REDIS_DEBUG", "0") == "1":
        logging.getLogger("api.redis").setLevel(logging.DEBUG)
        redis_logger = logging.getLogger("api.redis")
        redis_client = wrap_redis_client(redis_client, redis_logger)
    event_bus = EventBus(redis_client)

    geoip_db_path = os.getenv("GEOIP_DB_PATH", "/app/GeoLite2-City.mmdb")
    if os.path.exists(geoip_db_path):
        geoip_reader = geoip2.database.Reader(geoip_db_path)

    state.redis_client = redis_client
    state.event_bus = event_bus
    state.geoip_reader = geoip_reader

    enable_db = os.getenv("ENABLE_CHAT_DB", "0") == "1"
    if enable_db:
        try:
            await db.init_pool()
        except Exception:
            pass
    enable_agent = os.getenv("ENABLE_AGENT", "0") == "1"
    if enable_agent:
        stop_event = asyncio.Event()
        agent_tasks = await start_agents(stop_event)

    try:
        yield
    finally:
        if agent_tasks and stop_event:
            stop_event.set()
            try:
                await asyncio.wait_for(asyncio.gather(*agent_tasks, return_exceptions=True), timeout=5)
            except Exception:
                for t in agent_tasks:
                    t.cancel()
        if enable_db:
            try:
                await db.close_pool()
            except Exception:
                pass
        if redis_client:
            aclose = getattr(redis_client, "aclose", None)
            if callable(aclose):
                await aclose()
            else:
                close = getattr(redis_client, "close", None)
                if callable(close):
                    close()
        if geoip_reader:
            geoip_reader.close()
        state.redis_client = None
        state.event_bus = None
        state.geoip_reader = None

app.router.lifespan_context = lifespan

app.include_router(health_router)
app.include_router(example_router)
app.include_router(cache_router)
app.include_router(visitors_router)
app.include_router(system_router)
app.include_router(ws_visitors_router)
app.include_router(ws_chat_router)
app.include_router(chat_history_router)
app.include_router(events_history_router)
app.include_router(chat_admin_router)
app.include_router(when2meet_router, prefix="/w2m")
