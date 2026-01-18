import logging
import os

import geoip2.database
import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
)
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from redis.asyncio import BlockingConnectionPool as RedisConnectionPool

from api import db, state
from api.agents.gemini_agent import start_agents
from api.batch.visitor_analytics import start_analytics_scheduler
from api.bus import EventBus
from api.controllers.analytics_clicks import router as analytics_clicks_router
from api.controllers.analytics_clicks_write import router as analytics_clicks_write_router
from api.controllers.cache import router as cache_router
from api.controllers.chat_history import router as chat_history_router
from api.controllers.events_history import router as events_history_router
from api.controllers.example import router as example_router
from api.controllers.health import router as health_router
from api.controllers.notes import router as notes_router
from api.controllers.notes_search import router as notes_search_router
from api.controllers.system import router as system_router
from api.controllers.visitor_analytics import router as visitor_analytics_router
from api.controllers.visitors import router as visitors_router
from api.controllers.when2meet import router as when2meet_router
from api.controllers.ws_chat import router as ws_chat_router
from api.controllers.ws_visitors import router as ws_visitors_router
from api.middleware import HTTPLogMiddleware
from api.redis_debug import wrap_redis_client

app = FastAPI(title="DevStack Public API", version="1.0.0")

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
cors_regex_env = os.getenv("CORS_ORIGINS_REGEX", "").strip()
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
event_bus: EventBus | None = None
geoip_reader = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global redis_client, geoip_reader, event_bus
    stop_event: asyncio.Event | None = None
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

    analytics_tasks: list[asyncio.Task] = []
    enable_analytics = os.getenv("ENABLE_ANALYTICS_SCHEDULER", "1") == "1"
    if enable_analytics and enable_db:
        if stop_event is None:
            stop_event = asyncio.Event()
        analytics_tasks = await start_analytics_scheduler(stop_event)

    try:
        yield
    finally:
        if agent_tasks and stop_event:
            stop_event.set()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*agent_tasks, return_exceptions=True), timeout=5
                )
            except Exception:
                for t in agent_tasks:
                    t.cancel()
        if analytics_tasks and stop_event:
            stop_event.set()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*analytics_tasks, return_exceptions=True), timeout=5
                )
            except Exception:
                for t in analytics_tasks:
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
app.include_router(visitor_analytics_router)
app.include_router(analytics_clicks_router)
app.include_router(analytics_clicks_write_router)
app.include_router(when2meet_router, prefix="/w2m")
app.include_router(notes_router)
app.include_router(notes_search_router)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
