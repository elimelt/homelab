import asyncio
import hashlib
import json
import logging
import os
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar
from urllib.parse import urlparse

import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError

from api import db, state
from api.producers.chat_producer import build_chat_message, publish_chat_message


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


_logger = logging.getLogger("api.agents.gemini")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    _logger.addHandler(_handler)
_level_name = _env("AGENT_LOG_LEVEL", "DEBUG" if _env("AGENT_DEBUG", "") == "1" else "INFO").upper()
_logger.setLevel(getattr(logging, _level_name, logging.INFO))
_logger.propagate = False

_VOTE_PREFIX = "[vote]"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


_DAILY_LIMIT_KEY = "agent:daily_request_count"
_DAILY_LIMIT_DATE_KEY = "agent:daily_request_date"


async def _get_daily_request_count() -> int:
    if state.redis_client is None:
        return 0
    try:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        stored_date = await state.redis_client.get(_DAILY_LIMIT_DATE_KEY)
        if stored_date != today:
            await state.redis_client.set(_DAILY_LIMIT_DATE_KEY, today)
            await state.redis_client.set(_DAILY_LIMIT_KEY, "0")
            return 0
        count = await state.redis_client.get(_DAILY_LIMIT_KEY)
        return int(count) if count else 0
    except Exception:
        return 0


async def _increment_daily_request_count() -> int:
    if state.redis_client is None:
        return 0
    try:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        stored_date = await state.redis_client.get(_DAILY_LIMIT_DATE_KEY)
        if stored_date != today:
            await state.redis_client.set(_DAILY_LIMIT_DATE_KEY, today)
            await state.redis_client.set(_DAILY_LIMIT_KEY, "1")
            return 1
        new_count = await state.redis_client.incr(_DAILY_LIMIT_KEY)
        return int(new_count)
    except Exception:
        return 0


async def _can_make_request() -> bool:
    max_daily = int(_env("AGENT_MAX_DAILY_REQUESTS", "20"))
    current = await _get_daily_request_count()
    can_proceed = current < max_daily
    if not can_proceed:
        _logger.warning("Daily request limit reached: %d/%d", current, max_daily)
    return can_proceed


@dataclass
class AgentConfig:
    sender: str
    channels: list[str]
    min_sleep: int
    max_sleep: int
    max_replies: int
    history_token_limit: int
    model: str
    persona: str | None = None


GEMINI_PERSONAS = [
    {
        "key": "analytical",
        "name": "Analytical Thinker",
        "traits": "methodical, data-driven, breaks down complex problems, seeks evidence",
        "style": "You analyze topics systematically. You ask 'what does the data show?' and 'how can we measure this?'. You bring rigor and precision to discussions.",
        "contribution_focus": "empirical evidence, metrics, logical frameworks, systematic analysis",
    },
    {
        "key": "creative",
        "name": "Creative Explorer",
        "traits": "imaginative, makes unexpected connections, thinks laterally, embraces ambiguity",
        "style": "You explore unconventional angles and possibilities. You ask 'what if we tried...?' and 'here's a wild idea...'. You bring fresh perspectives and creative energy.",
        "contribution_focus": "novel approaches, analogies from other domains, thought experiments, creative solutions",
    },
    {
        "key": "pragmatic",
        "name": "Pragmatic Implementer",
        "traits": "practical, action-oriented, focuses on feasibility, grounds ideas in reality",
        "style": "You focus on what can actually be done. You ask 'how would we implement this?' and 'what are the concrete next steps?'. You bring discussions to actionable conclusions.",
        "contribution_focus": "implementation details, practical constraints, real-world examples, action items",
    },
]


ToolFunction = Callable[[dict[str, Any]], Awaitable[str]]
TOOL_MAP: dict[str, ToolFunction] = {}


def _participants_from_history(history: list[tuple[str, str, datetime]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for sender, _text, _ts in history:
        counts[sender] = counts.get(sender, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


async def _fetch_recent_messages_by_tokens(
    channel: str, token_limit: int, limit: int = 500
) -> list[tuple[str, str, datetime]]:
    rows = await db.fetch_chat_history(channel=channel, before_iso=None, limit=limit)
    _logger.debug("[fetch_history] Fetched %d raw rows from DB for channel=%s", len(rows), channel)
    out: list[tuple[str, str, datetime]] = []
    total_tokens = 0

    for m in rows:
        ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
        text = m.get("text") or ""
        sender = m.get("sender") or ""

        msg_tokens = _estimate_tokens(text) + _estimate_tokens(sender) + 30

        if total_tokens + msg_tokens > token_limit:
            break

        out.append((sender, text, ts))
        total_tokens += msg_tokens

    result = list(reversed(out))
    _logger.debug("[fetch_history] Returning %d messages (total_tokens=%d)", len(result), total_tokens)
    for i, (sender, text, ts) in enumerate(result[-5:]):
        _logger.debug("[fetch_history] msg[%d]: sender=%s, text=%s", len(result) - 5 + i, sender, text[:100] if text else "<empty>")
    return result


def _parse_channels(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _load_agent_configs() -> list[AgentConfig]:
    api_key = _env("GEMINI_API_KEY", "")
    if not api_key:
        _logger.info("GEMINI_API_KEY not set; agent disabled")
        return []

    num = int(_env("AGENTS", "3"))
    num = max(1, min(num, 10))
    common_channels = _parse_channels(_env("AGENT_CHANNELS", "general"))
    min_sleep = int(_env("AGENT_MIN_SLEEP_SEC", "3600"))
    max_sleep = int(_env("AGENT_MAX_SLEEP_SEC", "7200"))
    max_replies = int(_env("AGENT_MAX_REPLIES", "1"))
    history_token_limit = int(_env("AGENT_HISTORY_TOKEN_LIMIT", "10000"))
    model = _env("AGENT_MODEL", "gemini-2.5-flash")

    cfgs: list[AgentConfig] = []
    for i in range(1, num + 1):
        sender = _env(f"AGENT_{i}_SENDER", f"agent:gemini-{i}")
        channels = _parse_channels(_env(f"AGENT_{i}_CHANNELS", ",".join(common_channels)))
        persona_idx = (i - 1) % len(GEMINI_PERSONAS)
        persona_data = GEMINI_PERSONAS[persona_idx]
        persona_text = f"{persona_data['name']}: {persona_data['traits']}. {persona_data['style']} Focus on: {persona_data['contribution_focus']}."
        cfgs.append(
            AgentConfig(
                sender=sender,
                channels=channels,
                min_sleep=min_sleep + (i * 300),
                max_sleep=max_sleep + (i * 300),
                max_replies=max_replies,
                history_token_limit=history_token_limit,
                model=model,
                persona=persona_text,
            )
        )

    for cfg in cfgs:
        _logger.info(
            "Configured agent sender=%s channels=%s model=%s sleep=[%ss..%ss] max_replies=%s history_tokens=%d persona=%s",
            cfg.sender,
            cfg.channels,
            cfg.model,
            cfg.min_sleep,
            cfg.max_sleep,
            cfg.max_replies,
            cfg.history_token_limit,
            _safe_trunc(cfg.persona, 120),
        )
    return cfgs


async def _run_single_agent_loop(cfg: AgentConfig, stop_event: asyncio.Event) -> None:
    min_sleep = max(5, cfg.min_sleep)
    max_sleep = max(min_sleep, cfg.max_sleep)
    max_replies = max(0, min(cfg.max_replies, 5))

    wake_prob = float(_env("AGENT_WAKE_PROB", "0.1"))
    wake_cooldown_sec = int(_env("AGENT_WAKE_COOLDOWN_SEC", "300"))
    wake_event = asyncio.Event()
    last_wake_at: datetime | None = None

    _register_wake_handler(cfg, wake_event, wake_prob, wake_cooldown_sec, lambda: last_wake_at)

    while not stop_event.is_set():
        woke = await _wait_for_wake_or_timeout(
            stop_event, wake_event, random.randint(min_sleep, max_sleep)
        )
        if woke is None:
            break

        if state.event_bus is None:
            _logger.debug("[%s] Skipping session; event_bus not ready", cfg.sender)
            continue

        if not await _can_make_request():
            _logger.info("[%s] Skipping session; daily request limit reached", cfg.sender)
            continue

        channel = random.choice(cfg.channels)
        _logger.info(
            "[%s] Session channel=%s token_limit=%d", cfg.sender, channel, cfg.history_token_limit
        )

        last_wake_at = datetime.now(UTC)

        n_replies = min(max_replies, 1)
        if n_replies <= 0:
            continue

        for _ in range(n_replies):
            await _enqueue_generation_job(cfg.sender, channel, cfg.model)
        _logger.debug("[%s] Enqueued %s jobs for channel=%s", cfg.sender, n_replies, channel)


async def start_agents(stop_event: asyncio.Event) -> list[asyncio.Task]:
    cfgs = _load_agent_configs()
    tasks = [asyncio.create_task(_run_single_agent_loop(cfg, stop_event)) for cfg in cfgs]
    if cfgs:
        try:
            tasks.append(asyncio.create_task(_wake_listener(cfgs, stop_event)))
        except Exception:
            _logger.exception("Failed to start wake listener")
    try:
        all_channels = sorted({ch for cfg in cfgs for ch in cfg.channels})
        tasks.append(asyncio.create_task(_generation_worker(all_channels, stop_event)))
    except Exception:
        _logger.exception("Failed to start generation worker")
    return tasks


def _safe_trunc(s: str, n: int) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + "…"


_CHANNEL_TO_HANDLERS: dict[str, list[dict[str, object]]] = {}
_HANDLERS_LOCK = asyncio.Lock()


def _register_wake_handler(
    cfg: AgentConfig,
    wake_event: asyncio.Event,
    wake_prob: float,
    cooldown_sec: int,
    last_wake_getter,
):
    entry = {
        "event": wake_event,
        "prob": wake_prob,
        "cooldown": cooldown_sec,
        "last": last_wake_getter,
    }
    for ch in cfg.channels:
        chan = f"chat:{ch}"
        _CHANNEL_TO_HANDLERS.setdefault(chan, []).append(entry)


async def _wake_listener(cfgs: list[AgentConfig], stop_event: asyncio.Event) -> None:
    if state.redis_client is None:
        _logger.info("Wake listener not started; redis not ready")
        return
    channels: set[str] = set()
    for cfg in cfgs:
        for ch in cfg.channels:
            channels.add(f"chat:{ch}")
    if not channels:
        return
    pubsub = state.redis_client.pubsub()
    await pubsub.subscribe(*channels)
    _logger.info("Wake listener subscribed to %s", list(channels))
    try:
        async for message in pubsub.listen():
            if stop_event.is_set():
                break
            if message.get("type") != "message":
                continue
            chan = message.get("channel")
            if isinstance(chan, bytes):
                chan = chan.decode("utf-8")
            raw_data = message.get("data", "")
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8")
            _logger.debug("[wake_listener] Received message on %s: %s", chan, raw_data[:500] if raw_data else "<empty>")
            try:
                parsed = json.loads(raw_data) if raw_data else {}
                sender = parsed.get("sender", "<unknown>")
                text = parsed.get("text", "")
                _logger.info("[wake_listener] Message from %s: %s", sender, text[:200] if text else "<empty>")
            except Exception as e:
                _logger.debug("[wake_listener] Could not parse message: %s", e)
            handlers = _CHANNEL_TO_HANDLERS.get(chan, [])
            _logger.debug("[wake_listener] Found %d handlers for channel %s", len(handlers), chan)

            for h in handlers:
                try:
                    event: asyncio.Event = h["event"]
                    prob: float = h["prob"]
                    cooldown: int = h["cooldown"]
                    last_fn = h["last"]
                    last_at = last_fn()
                    now = datetime.now(UTC)
                    if last_at and (now - last_at).total_seconds() < cooldown:
                        continue
                    if random.random() < prob:
                        event.set()
                except Exception:
                    pass
    finally:
        try:
            await pubsub.unsubscribe(*channels)
        finally:
            if hasattr(pubsub, "aclose"):
                await pubsub.aclose()
            else:
                await pubsub.close()


async def _wait_or_clear(ev: asyncio.Event) -> None:
    await ev.wait()
    ev.clear()


async def _wait_for_wake_or_timeout(
    stop_event: asyncio.Event, wake_event: asyncio.Event, timeout: int
) -> bool | None:
    stop_task = asyncio.create_task(stop_event.wait())
    wake_task = asyncio.create_task(_wait_or_clear(wake_event))
    try:
        done, _pending = await asyncio.wait(
            [stop_task, wake_task], timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            return None
        return wake_task in done
    finally:
        for t in (stop_task, wake_task):
            if not t.done():
                t.cancel()


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def _remember_agent_message(sender: str, text: str) -> None:
    try:
        if state.redis_client is None:
            return
        key = f"agent:last_msgs:{sender}"
        await state.redis_client.lpush(key, text)
        await state.redis_client.ltrim(key, 0, 4)
        await state.redis_client.expire(key, int(_env("AGENT_LAST_MSG_TTL_SEC", "3600")))
    except Exception:
        pass


async def _can_speak_now(sender: str) -> bool:
    min_gap = int(_env("AGENT_MIN_TURN_GAP_SEC", "10"))
    if min_gap <= 0 or state.redis_client is None:
        return True
    try:
        key = f"agent:last_ts:{sender}"
        last = await state.redis_client.get(key)
        if not last:
            return True
        last_ts = float(last)
        return (datetime.now(UTC).timestamp() - last_ts) >= min_gap
    except Exception:
        return True


async def _mark_spoke(sender: str) -> None:
    try:
        if state.redis_client is None:
            return
        key = f"agent:last_ts:{sender}"
        await state.redis_client.set(
            key, str(datetime.now(UTC).timestamp()), ex=int(_env("AGENT_LAST_TS_TTL_SEC", "7200"))
        )
    except Exception:
        pass


def _queue_key(channel: str) -> str:
    return f"agent:genq:{channel}"


async def _enqueue_generation_job(sender: str, channel: str, model: str) -> None:
    if state.redis_client is None:
        return
    job = json.dumps(
        {
            "sender": sender,
            "channel": channel,
            "model": model,
            "ts": datetime.now(UTC).astimezone(UTC).isoformat(),
        }
    )
    await state.redis_client.rpush(_queue_key(channel), job)

    await state.redis_client.ltrim(_queue_key(channel), -200, -1)


class _channel_mutex:
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(self, channel: str):
        self._channel = channel
        if channel not in _channel_mutex._locks:
            _channel_mutex._locks[channel] = asyncio.Lock()
        self._lock = _channel_mutex._locks[channel]

    async def __aenter__(self):
        await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            self._lock.release()
        except Exception:
            pass


async def _remember_hash_and_check(channel: str, h: str) -> bool:
    try:
        if state.redis_client is None:
            return True
        key = f"agent:recent_hash:{channel}"
        added = await state.redis_client.sadd(key, h)
        await state.redis_client.expire(key, int(_env("AGENT_RECENT_HASH_TTL_SEC", "1800")))

        size = await state.redis_client.scard(key)
        if size and size > 500:
            for _ in range(5):
                await state.redis_client.spop(key)
        return bool(added)
    except Exception:
        return True


async def _mark_channel_spoke(channel: str) -> None:
    try:
        if state.redis_client is None:
            return
        key = f"agent:chan_last_ts:{channel}"
        await state.redis_client.set(
            key,
            str(datetime.now(UTC).timestamp()),
            ex=int(_env("AGENT_CHAN_LAST_TS_TTL_SEC", "7200")),
        )
    except Exception:
        pass


TOOL_URL_FETCH = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_fetch_url",
            description="Fetch the content of a public URL (web page, text file, etc.) for reading. Useful for looking up recent context or information. Returns plain text content.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "url": types.Schema(
                        type=types.Type.STRING,
                        description="The full URL to fetch, e.g., 'https://example.com/data.txt'",
                    ),
                    "max_bytes": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum content size to fetch (default 5000).",
                    ),
                },
                required=["url"],
            ),
        )
    ]
)

TOOL_SEARCH_EVENTS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_search_events",
            description="Search the system's event log for messages related to a topic or event type.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "topic": types.Schema(
                        type=types.Type.STRING, description="A general topic string to search for."
                    ),
                    "type": types.Schema(
                        type=types.Type.STRING,
                        description="A specific event type (e.g., 'user_joined', 'config_changed').",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of events to return (default 20).",
                    ),
                },
            ),
        )
    ]
)

TOOL_QUERY_CHAT = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_query_chat",
            description="Search the chat history for messages in a specific channel matching a keyword.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "channel": types.Schema(
                        type=types.Type.STRING,
                        description="The channel name (e.g., 'general', 'dev').",
                    ),
                    "keyword": types.Schema(
                        type=types.Type.STRING, description="A keyword to filter the messages by."
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of messages to return (default 50).",
                    ),
                },
            ),
        )
    ]
)

TOOL_VISITOR_ANALYTICS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_visitor_analytics",
            description="Query visitor analytics and statistics. Returns metrics about site visitors including visit counts, session durations, recurring visitor status, and visit frequency.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "visitor_ip": types.Schema(
                        type=types.Type.STRING,
                        description="Optional: filter by specific visitor IP address.",
                    ),
                    "start_date": types.Schema(
                        type=types.Type.STRING,
                        description="Optional: start date filter (ISO8601 format, e.g., '2025-01-01').",
                    ),
                    "end_date": types.Schema(
                        type=types.Type.STRING,
                        description="Optional: end date filter (ISO8601 format, e.g., '2025-01-31').",
                    ),
                    "recurring_only": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="Optional: filter to recurring visitors only (true) or non-recurring only (false).",
                    ),
                    "summary": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="If true, return aggregate summary instead of individual records (default: false).",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of records to return (default 50).",
                    ),
                },
            ),
        )
    ]
)

TOOL_SEARCH_NOTES = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_search_notes",
            description="Search the knowledge base of notes and documentation. Returns relevant documents matching the query with titles, descriptions, categories, and relevance scores. Use this to find information on specific topics.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The search query to find relevant notes (e.g., 'machine learning', 'kubernetes deployment').",
                    ),
                    "mode": types.Schema(
                        type=types.Type.STRING,
                        description="Search mode: 'fulltext' (keyword matching), 'semantic' (meaning-based), or 'hybrid' (combined). Default: 'hybrid'.",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of results to return (default 10, max 20).",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Optional: filter by category name (e.g., 'programming', 'devops').",
                    ),
                    "tags": types.Schema(
                        type=types.Type.STRING,
                        description="Optional: comma-separated tags to filter by (e.g., 'python,tutorial').",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)

TOOL_GET_NOTE = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_get_note",
            description="Retrieve the full markdown content of a specific note by its document ID. Use this after searching to read the complete content of a relevant document.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "doc_id": types.Schema(
                        type=types.Type.INTEGER,
                        description="The document ID to retrieve (obtained from search results).",
                    ),
                },
                required=["doc_id"],
            ),
        )
    ]
)

TOOL_RUN_PYTHON = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="tool_run_python",
            description="Execute Python code in a secure sandbox and return the output. Available libraries: numpy, pandas, scipy, sympy, matplotlib, seaborn, scikit-learn, requests, beautifulsoup4, pyyaml. Security restrictions: no network access, no file system access, 30s timeout, 128MB memory limit.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(
                        type=types.Type.STRING,
                        description="The Python code to execute. Can be multiple lines.",
                    ),
                },
                required=["code"],
            ),
        )
    ]
)

ALL_TOOLS: list[types.Tool] = [
    TOOL_URL_FETCH,
    TOOL_SEARCH_EVENTS,
    TOOL_QUERY_CHAT,
    TOOL_VISITOR_ANALYTICS,
    TOOL_SEARCH_NOTES,
    TOOL_GET_NOTE,
    TOOL_RUN_PYTHON,
]


async def _tool_fetch_url(args: dict[str, object]) -> str:
    url = str(args.get("url") or "")
    max_bytes = int(args.get("max_bytes") or 5000)
    timeout = int(args.get("timeout_sec") or 5)
    retries = int(args.get("retries") or int(_env("AGENT_FETCH_RETRIES", "1")))
    backoff_ms = int(args.get("backoff_ms") or int(_env("AGENT_FETCH_BACKOFF_MS", "500")))
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "ERROR: unsupported scheme"

    def _fetch_with_retries_sync() -> str:
        attempt = 0
        delay = backoff_ms / 1000.0

        while attempt <= retries:
            try:
                resp = requests.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()

                size = 0
                chunks = []
                for chunk in resp.iter_content(chunk_size=1024):
                    if not chunk:
                        break
                    if size + len(chunk) > max_bytes:
                        chunks.append(chunk[: max_bytes - size])
                        size = max_bytes
                        break
                    chunks.append(chunk)
                    size += len(chunk)

                data = b"".join(chunks)
                text = data.decode("utf-8", errors="replace")

                return f"status={resp.status_code} content_type={resp.headers.get('Content-Type', '')}\n{text}"

            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (429, 503):
                    ra = e.response.headers.get("Retry-After")
                    return_hint = f"status={e.response.status_code} rate_limited=true retry_after={ra or ''} content_type={e.response.headers.get('Content-Type', '')}\n{_safe_trunc(e.response.text, 500)}"
                    if attempt == retries:
                        return return_hint

                    wait = delay
                    try:
                        if ra:
                            wait = float(ra)
                    except Exception:
                        pass

                    import time

                    time.sleep(wait)
                    attempt += 1
                    delay *= 2
                    continue
                return f"ERROR: HTTP {e.response.status_code}: {e!r}"

            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    return f"ERROR: Request failed: {e!r}"

                import time

                time.sleep(delay)
                attempt += 1
                delay *= 2
        return "ERROR: unknown"

    return await asyncio.to_thread(_fetch_with_retries_sync)


TOOL_MAP["tool_fetch_url"] = _tool_fetch_url


async def _tool_search_events(args: dict[str, object]) -> str:
    topic = args.get("topic")
    typ = args.get("type")
    limit = int(args.get("limit") or 20)
    try:
        rows = await db.fetch_events(
            topic=str(topic) if topic else None,
            event_type=str(typ) if typ else None,
            before_iso=None,
            limit=limit,
        )
        out = {"count": len(rows), "events": rows}
        text = json.dumps(out)[: int(_env("AGENT_TOOL_MAX_OUTPUT_CHARS", "2000"))]
        return text
    except Exception as e:
        _logger.error("tool_search_events failed: %s", e)
        return f"ERROR: {e!r}"


TOOL_MAP["tool_search_events"] = _tool_search_events


async def _tool_query_chat(args: dict[str, object]) -> str:
    channel = str(args.get("channel") or "general")
    keyword = str(args.get("keyword") or "").lower()
    limit = int(args.get("limit") or 50)
    try:
        rows = await db.fetch_chat_history(channel=channel, before_iso=None, limit=limit)
        if keyword:
            rows = [r for r in rows if keyword in (r.get("text") or "").lower()]
        out = {"count": len(rows), "messages": rows}
        return json.dumps(out)[: int(_env("AGENT_TOOL_MAX_OUTPUT_CHARS", "2000"))]
    except Exception as e:
        _logger.error("tool_query_chat failed: %s", e)
        return f"ERROR: {e!r}"


TOOL_MAP["tool_query_chat"] = _tool_query_chat


async def _tool_visitor_analytics(args: dict[str, object]) -> str:
    visitor_ip = args.get("visitor_ip")
    start_date = args.get("start_date")
    end_date = args.get("end_date")
    recurring_only = args.get("recurring_only")
    summary_mode = args.get("summary", False)
    limit = int(args.get("limit") or 50)

    try:
        if summary_mode:
            result = await db.get_visitor_analytics_summary(
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
            )
            out = {"type": "summary", "data": result}
        else:
            rows = await db.fetch_visitor_stats(
                visitor_ip=str(visitor_ip) if visitor_ip else None,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
                is_recurring=bool(recurring_only) if recurring_only is not None else None,
                limit=limit,
            )
            out = {"type": "visitors", "count": len(rows), "visitors": rows}

        return json.dumps(out)[: int(_env("AGENT_TOOL_MAX_OUTPUT_CHARS", "2000"))]
    except Exception as e:
        _logger.error("tool_visitor_analytics failed: %s", e)
        return f"ERROR: {e!r}"


TOOL_MAP["tool_visitor_analytics"] = _tool_visitor_analytics


async def _tool_search_notes(args: dict[str, object]) -> str:
    query = str(args.get("query") or "")
    if not query:
        return "ERROR: query parameter is required"

    mode = str(args.get("mode") or "hybrid").lower()
    if mode not in ("fulltext", "semantic", "hybrid"):
        mode = "hybrid"

    limit = min(int(args.get("limit") or 10), 20)
    category = args.get("category")
    tags = args.get("tags")

    try:
        category_id = None
        if category:
            cat = await db.notes_get_category_by_name(str(category))
            if cat:
                category_id = cat["id"]

        tag_ids: list[int] | None = None
        if tags:
            tag_names = [t.strip() for t in str(tags).split(",") if t.strip()]
            tag_ids = []
            for tag_name in tag_names:
                tag_obj = await db.notes_get_tag_by_name(tag_name)
                if tag_obj:
                    tag_ids.append(tag_obj["id"])

        results: list[dict] = []
        if mode == "fulltext":
            results = await db.notes_fulltext_search(
                query=query, limit=limit, offset=0, category_id=category_id, tag_ids=tag_ids
            )
        elif mode == "semantic":
            try:
                from api.notes_embeddings import generate_query_embedding, is_model_available

                if not is_model_available():
                    results = await db.notes_fulltext_search(
                        query=query, limit=limit, offset=0, category_id=category_id, tag_ids=tag_ids
                    )
                else:
                    query_embedding = generate_query_embedding(query)
                    results = await db.notes_vector_search(
                        embedding=query_embedding, limit=limit, offset=0,
                        category_id=category_id, tag_ids=tag_ids
                    )
            except ImportError:
                results = await db.notes_fulltext_search(
                    query=query, limit=limit, offset=0, category_id=category_id, tag_ids=tag_ids
                )
        else:
            fts_results = await db.notes_fulltext_search(
                query=query, limit=limit + 10, offset=0, category_id=category_id, tag_ids=tag_ids
            )
            results = fts_results[:limit]

        if not results:
            return f"No notes found matching query: '{query}'"

        lines = [f"Found {len(results)} notes matching '{query}':", ""]
        for i, doc in enumerate(results, 1):
            title = doc.get("title") or "Untitled"
            doc_id = doc.get("id", "?")
            category_name = doc.get("category") or "uncategorized"
            description = doc.get("description") or ""
            tags_list = doc.get("tags", [])
            rank = doc.get("rank") or doc.get("similarity") or 0

            lines.append(f"{i}. **{title}** (ID: {doc_id})")
            lines.append(f"   Category: {category_name}")
            if tags_list:
                lines.append(f"   Tags: {', '.join(tags_list)}")
            if description:
                desc_short = description[:150] + "..." if len(description) > 150 else description
                lines.append(f"   {desc_short}")
            if rank:
                lines.append(f"   Relevance: {rank:.4f}")
            lines.append("")

        return "\n".join(lines)[: int(_env("AGENT_TOOL_MAX_OUTPUT_CHARS", "3000"))]
    except Exception as e:
        _logger.error("tool_search_notes failed: %s", e)
        return f"ERROR: {e!r}"


TOOL_MAP["tool_search_notes"] = _tool_search_notes


async def _tool_get_note(args: dict[str, object]) -> str:
    doc_id = args.get("doc_id")
    if doc_id is None:
        return "ERROR: doc_id parameter is required"

    try:
        doc_id_int = int(doc_id)
    except (ValueError, TypeError):
        return f"ERROR: doc_id must be an integer, got: {doc_id}"

    try:
        document = await db.notes_get_document_by_id(doc_id_int)
        if not document:
            return f"ERROR: Note with ID {doc_id_int} not found"

        title = document.get("title") or "Untitled"
        category = document.get("category") or "uncategorized"
        tags = document.get("tags", [])
        content = document.get("content") or ""
        last_modified = document.get("last_modified") or "unknown"

        lines = [
            f"# {title}",
            "",
            f"**Category:** {category}",
        ]
        if tags:
            lines.append(f"**Tags:** {', '.join(tags)}")
        lines.append(f"**Last Modified:** {last_modified}")
        lines.append(f"**Document ID:** {doc_id_int}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(content)

        return "\n".join(lines)[: int(_env("AGENT_TOOL_MAX_OUTPUT_CHARS", "6000"))]
    except Exception as e:
        _logger.error("tool_get_note failed: %s", e)
        return f"ERROR: {e!r}"


TOOL_MAP["tool_get_note"] = _tool_get_note


async def _tool_run_python(args: dict[str, object]) -> str:
    code = str(args.get("code") or "")
    if not code:
        return "ERROR: code parameter is required"

    try:
        from api.sandbox import execute_python, is_sandbox_available

        if not is_sandbox_available():
            return "ERROR: Python sandbox is not available. The sandbox image may need to be built."

        result, success = await asyncio.to_thread(execute_python, code, "gemini-agent")

        if success:
            return result if result else "(no output)"
        else:
            return f"Execution failed:\n{result}"

    except ImportError:
        return "ERROR: Sandbox module not available"
    except Exception as e:
        _logger.error("tool_run_python failed: %s", e)
        return f"ERROR: {e!r}"


TOOL_MAP["tool_run_python"] = _tool_run_python


def _build_system_instruction(
    channel: str, persona: str | None, actors: list[tuple[str, int]]
) -> str:
    lines: list[str] = [
        f"You are an autonomous AI participant in the #{channel} discussion channel.",
        "",
        "## CORE IDENTITY",
        "You are a knowledgeable peer contributing to substantive discussions. Humans take priority when present, but you drive conversations forward autonomously.",
    ]

    if persona:
        lines.append(f"\n## YOUR COGNITIVE STYLE\n{persona}")

    lines.extend([
        "",
        "## CONTRIBUTION PRINCIPLES",
        "1. CONTRIBUTE, DON'T ASK: Never ask 'which topic?' or 'what should we discuss?'. Pick something and contribute.",
        "2. BUILD ON CONTEXT: Reference what others said. Use 'Building on X's point...', 'That connects to...', 'Counterpoint to consider...'",
        "3. ADD UNIQUE VALUE: Every message must contain insight, evidence, a concrete proposal, or a question worth exploring.",
        "4. BE SPECIFIC: Use examples, data, analogies, or references. Ground abstract ideas in concrete reality.",
        "5. ADVANCE THE DISCUSSION: Move topics forward. Introduce new angles, deeper questions, or actionable next steps.",
        "",
        "## CRITICAL ANTI-PATTERNS (NEVER DO THESE)",
        "- NEVER ask 'which topic would you like to explore?' or similar open-ended topic requests",
        "- NEVER repeat the same question format you or others have already used",
        "- NEVER comment on conversation structure or meta-discuss the chat itself",
        "- NEVER use filler like 'Great point!' without adding substance",
        "- NEVER apologize for conversation loops or acknowledge repetition - just contribute something new",
        "- NEVER say the same thing another agent just said, even paraphrased",
        "",
        "## WHEN CONVERSATION STALLS",
        "If the discussion seems stuck, choose ONE of these strategies:",
        "- Share a specific example or case study that illuminates the topic",
        "- Introduce a surprising connection to a different field",
        "- Pose a concrete thought experiment or hypothetical",
        "- Challenge an assumption the group hasn't questioned",
        "- Propose a specific action or next step",
        "",
        "## COORDINATION",
        "Use [vote] prefix to nominate a leader when coordination is needed (e.g., [vote] agent:name for reason).",
        "Respect role differentiation - if another agent is exploring an angle, take a different one.",
        "",
        "## AVAILABLE TOOLS",
        "You have access to the following tools to gather information:",
        "- **tool_fetch_url**: Fetch content from a public URL (web pages, text files)",
        "- **tool_search_events**: Search the system event log for messages by topic or type",
        "- **tool_query_chat**: Search chat history in a specific channel by keyword",
        "- **tool_visitor_analytics**: Query visitor statistics and metrics",
        "- **tool_search_notes**: Search the knowledge base of notes and documentation",
        "- **tool_get_note**: Retrieve full markdown content of a note by document ID",
        "- **tool_run_python**: Execute Python code in a secure sandbox (numpy, pandas, scipy, matplotlib available)",
        "",
        "Use tools proactively - run calculations, search notes, verify claims with code.",
    ])

    if actors:
        lines.append("\n## ACTIVE PARTICIPANTS")
        for sender, count in actors[:8]:
            lines.append(f"- {sender} ({count} msgs)")

    return "\n".join(lines)


def _build_history_contents(history: list[tuple[str, str, datetime]]) -> list[types.Content]:
    contents: list[types.Content] = []

    _logger.info("[gemini_build_contents] Building contents from %d history messages", len(history))
    _logger.info("[gemini_build_contents] Last 3 messages in history:")
    for sender, text, ts in history[-3:]:
        _logger.info("  - %s: %s", sender, text[:100] if text else "<empty>")

    for sender, text, ts in history:
        role = "user" if not sender.startswith("agent:") else "model"
        prefixed_text = f"[{ts.astimezone(UTC).isoformat()}] {sender}: {text}"

        contents.append(types.Content(role=role, parts=[types.Part(text=prefixed_text)]))

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="Your turn. Contribute something substantive that advances the discussion. Do NOT ask which topic to discuss - just contribute. Do NOT repeat what's already been said. Return ONLY your message text or a tool call."
                )
            ],
        )
    )

    return contents


def _sync_generate_content(
    client: genai.Client,
    model: str,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _env("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        _client = genai.Client(api_key=api_key)
    return _client


async def _call_gemini_with_retry(
    cfg: AgentConfig,
    contents: list[types.Content],
    system_instruction: str,
    call_name: str = "call",
) -> types.GenerateContentResponse | None:
    max_retry_sec = int(_env("AGENT_LLM_MAX_RETRY_SEC", "3600"))
    base_delay = 5
    elapsed = 0
    attempt = 0
    while elapsed < max_retry_sec:
        try:
            response = await asyncio.to_thread(
                _sync_generate_content,
                _get_client(),
                cfg.model,
                contents,
                types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=ALL_TOOLS,
                ),
            )
            return response
        except APIError as e:
            if hasattr(e, "code") and e.code == 429:
                delay = min(base_delay * (2**attempt), max_retry_sec - elapsed, 300)
                _logger.warning(
                    "[%s] %s rate limited, retrying in %.0fs (attempt %d)",
                    cfg.sender,
                    call_name,
                    delay,
                    attempt + 1,
                )
                await asyncio.sleep(delay)
                elapsed += delay
                attempt += 1
                continue
            _logger.error("[%s] %s failed: %s", cfg.sender, call_name, e)
            return None
        except Exception:
            _logger.exception("[%s] Unexpected error in %s", cfg.sender, call_name)
            return None
    _logger.error("[%s] %s exceeded max retry time of %ds", cfg.sender, call_name, max_retry_sec)
    return None


async def _call_gemini_async(
    cfg: AgentConfig, channel: str, history: list[tuple[str, str, datetime]]
) -> str | None:
    actors = _participants_from_history(history)
    system_instruction = _build_system_instruction(channel, cfg.persona, actors)
    contents = _build_history_contents(history)

    _logger.debug(
        "[%s] Calling Gemini with history_count=%d, model=%s", cfg.sender, len(contents), cfg.model
    )

    response = await _call_gemini_with_retry(cfg, contents, system_instruction, "first Gemini call")
    if response is None:
        return None

    if response.function_calls:
        _logger.info(
            "[%s] Received %d tool calls for channel=%s",
            cfg.sender,
            len(response.function_calls),
            channel,
        )

        tool_results: list[types.Part] = []
        for call in response.function_calls:
            function_name = call.name
            tool_func = TOOL_MAP.get(function_name)

            if not tool_func:
                _logger.warning("[%s] Unknown function call: %s", cfg.sender, function_name)
                result = f"ERROR: Unknown function {function_name}"
            else:
                _logger.debug(
                    "[%s] Executing tool: %s(%s)", cfg.sender, function_name, dict(call.args)
                )
                try:
                    result = await tool_func(dict(call.args))
                    _logger.debug(
                        "[%s] Tool %s result length: %d", cfg.sender, function_name, len(result)
                    )
                except Exception as e:
                    _logger.exception("[%s] Tool %s failed", cfg.sender, function_name)
                    result = f"ERROR: Tool execution failed with exception: {e}"

            tool_results.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=function_name,
                        response={"result": result},
                    )
                )
            )

        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="tool", parts=tool_results))

        if not await _can_make_request():
            _logger.warning(
                "[%s] Daily limit reached before second call; skipping tool response", cfg.sender
            )
            return None

        new_count = await _increment_daily_request_count()
        _logger.debug(
            "[%s] Second call to Gemini with tool results (request #%d).", cfg.sender, new_count
        )
        response = await _call_gemini_with_retry(
            cfg, contents, system_instruction, "second Gemini call"
        )
        if response is None:
            return None

    if not response.candidates:
        _logger.warning("[%s] Gemini returned no candidates.", cfg.sender)
        return None

    for part in response.candidates[0].content.parts:
        if part.text:
            text = part.text
            _logger.debug(
                "[%s] Gemini text_len=%s snippet=%s", cfg.sender, len(text), _safe_trunc(text, 200)
            )
            return text.strip()

    return None


async def _generation_worker(channels: list[str], stop_event: asyncio.Event) -> None:
    if state.redis_client is None or state.event_bus is None:
        _logger.info("Generation worker not started; redis/event_bus not ready")
        return

    api_key = _env("GEMINI_API_KEY", "")
    if not api_key:
        _logger.error("GEMINI_API_KEY is not set. Generation worker disabled.")
        return

    cfgs = _load_agent_configs()
    cfg_map = {c.sender: c for c in cfgs}

    keys = [_queue_key(ch) for ch in channels] or [_queue_key("general")]
    while not stop_event.is_set():
        try:
            res = await state.redis_client.brpop(keys, timeout=5)
            if res is None:
                continue
            _key, raw = res

            try:
                job = json.loads(raw)
            except Exception:
                continue

            sender = job.get("sender") or ""
            channel = job.get("channel") or "general"

            cfg = cfg_map.get(sender)
            if not cfg:
                _logger.warning("Job for unknown sender %s received.", sender)
                continue

            async with _channel_mutex(channel):
                if not await _can_speak_now(sender):
                    _logger.debug("[%s] Skipping message due to minimum turn gap.", sender)
                    continue

                if not await _can_make_request():
                    _logger.info("[%s] Skipping generation; daily request limit reached", sender)
                    continue

                history = await _fetch_recent_messages_by_tokens(channel, cfg.history_token_limit)

                new_count = await _increment_daily_request_count()
                _logger.info(
                    "[%s] Making API request %d for channel=%s with %d messages",
                    sender,
                    new_count,
                    channel,
                    len(history),
                )

                text = await _call_gemini_async(cfg, channel, history)

                if not text:
                    continue

                norm = _normalize_text(text)
                h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                if not await _remember_hash_and_check(channel, h):
                    _logger.debug("[%s] Skipping message due to recent hash match.", sender)
                    continue

                event = build_chat_message(channel=channel, sender=sender, text=text)
                await publish_chat_message(state.event_bus, channel, event)
                await _mark_spoke(sender)
                await _remember_agent_message(sender, text)
                await _mark_channel_spoke(channel)

        except APIError as e:
            _logger.error("Gemini API Error in worker: %s", e)
            await asyncio.sleep(1)
        except Exception as e:
            _logger.exception("Unexpected error in generation worker: %s", e)
            await asyncio.sleep(0.25)
