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
    """Estimate token count for a string. Rough approximation: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


_DAILY_LIMIT_KEY = "agent:daily_request_count"
_DAILY_LIMIT_DATE_KEY = "agent:daily_request_date"


async def _get_daily_request_count() -> int:
    """Get current daily request count, resetting if date changed."""
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
    """Increment and return the new daily request count."""
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
    """Check if we're under the daily request limit."""
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
    """Fetch recent messages up to a token limit instead of time-based."""
    rows = await db.fetch_chat_history(channel=channel, before_iso=None, limit=limit)
    out: list[tuple[str, str, datetime]] = []
    total_tokens = 0

    for m in reversed(rows):
        ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
        text = m.get("text") or ""
        sender = m.get("sender") or ""

        msg_tokens = _estimate_tokens(text) + _estimate_tokens(sender) + 30

        if total_tokens + msg_tokens > token_limit:
            break

        out.append((sender, text, ts))
        total_tokens += msg_tokens

    return list(reversed(out))


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
        cfgs.append(
            AgentConfig(
                sender=sender,
                channels=channels,
                min_sleep=min_sleep,
                max_sleep=max_sleep,
                max_replies=max_replies,
                history_token_limit=history_token_limit,
                model=model,
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
            handlers = _CHANNEL_TO_HANDLERS.get(chan, [])

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

ALL_TOOLS: list[types.Tool] = [
    TOOL_URL_FETCH,
    TOOL_SEARCH_EVENTS,
    TOOL_QUERY_CHAT,
    TOOL_VISITOR_ANALYTICS,
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
            type=str(typ) if typ else None,
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
    """Query visitor analytics from the database."""
    visitor_ip = args.get("visitor_ip")
    start_date = args.get("start_date")
    end_date = args.get("end_date")
    recurring_only = args.get("recurring_only")
    summary_mode = args.get("summary", False)
    limit = int(args.get("limit") or 50)

    try:
        if summary_mode:
            # Return aggregate summary
            result = await db.get_visitor_analytics_summary(
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None,
            )
            out = {"type": "summary", "data": result}
        else:
            # Return individual visitor records
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


def _build_system_instruction(
    channel: str, persona: str | None, actors: list[tuple[str, int]]
) -> str:
    lines: list[str] = [
        f"You are an autonomous agent named {channel} in the @{channel} channel.",
        "Your goal is to be a social peer and productive team member.",
        "Humans are the boss; prioritize their needs and decisions.",
        "Reply only when you have meaningful value to add (new info, proposal, action step). Avoid filler or repetitive messages.",
        "Keep messages brief, useful, and non-repetitive.",
        "Coordinate like a self-organizing team: adopt roles, establish hierarchy, and respect different agendas.",
        "Drive towards concrete proposals, decisions, and actions. Avoid meta-debates.",
        "If useful for coordination, elect a leader by starting your reply with the [vote] prefix, followed by the candidate's name and reason (e.g., [vote] agent:leader_name for coordinating next steps).",
    ]
    if persona:
        lines.append(f"\nYour specific persona and role: {persona}")
    if actors:
        lines.append("\nRecently active participants:")
        for sender, count in actors[:8]:
            lines.append(f"- {sender} ({count} msgs)")
    return "\n".join(lines)


def _build_history_contents(history: list[tuple[str, str, datetime]]) -> list[types.Content]:
    """Build history contents from already token-limited history."""
    contents: list[types.Content] = []

    for sender, text, ts in history:
        role = "user" if not sender.startswith("agent:") else "model"
        prefixed_text = f"[{ts.astimezone(UTC).isoformat()}] {sender}: {text}"

        contents.append(types.Content(role=role, parts=[types.Part(text=prefixed_text)]))

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text="Review the conversation history and your instructions. Now produce your next message (or call a tool). Return ONLY the message text or tool call."
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


_client = genai.Client(api_key=_env("GEMINI_API_KEY", ""))


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
                _client,
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
