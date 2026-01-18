import asyncio
import logging
import os
import random
from datetime import UTC, datetime

from api import db, state
from api.producers.chat_producer import build_chat_message, publish_chat_message

_logger = logging.getLogger("api.agents.augment")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


async def _fetch_recent_messages_by_tokens(
    channel: str, token_limit: int, limit: int = 500
) -> list[tuple[str, str, datetime]]:
    rows = await db.fetch_chat_history(channel=channel, before_iso=None, limit=limit)
    _logger.debug("[augment_fetch] Fetched %d raw rows from DB for channel=%s", len(rows), channel)
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
    _logger.debug("[augment_fetch] Returning %d messages (total_tokens=%d)", len(result), total_tokens)
    for i, (sender, text, ts) in enumerate(result[-5:]):
        _logger.debug("[augment_fetch] msg[%d]: sender=%s, text=%s", len(result) - 5 + i, sender, text[:100] if text else "<empty>")
    return result


AUGMENT_PERSONAS = {
    "architect": {
        "name": "Systems Architect",
        "traits": "analytical, sees patterns, connects disparate ideas, thinks in systems",
        "style": "You approach topics by examining underlying structures and relationships. You ask 'how does this connect to X?' and 'what are the second-order effects?'. You synthesize ideas across domains.",
        "topics": ["distributed systems", "emergence", "feedback loops", "architecture patterns", "complexity theory", "network effects"],
    },
    "challenger": {
        "name": "Constructive Skeptic",
        "traits": "curious, probing, plays devil's advocate, stress-tests ideas",
        "style": "You respectfully challenge assumptions and explore edge cases. You ask 'what could go wrong?' and 'have we considered the opposite?'. You strengthen ideas through rigorous questioning.",
        "topics": ["failure modes", "unintended consequences", "edge cases", "alternative approaches", "historical precedents", "contrarian views"],
    },
    "synthesizer": {
        "name": "Bridge Builder",
        "traits": "integrative, finds common ground, translates between perspectives, builds on others' ideas",
        "style": "You connect different viewpoints and find synthesis. You say 'building on what X said...' and 'I see a thread connecting these ideas...'. You help the group reach deeper understanding.",
        "topics": ["interdisciplinary connections", "practical applications", "consensus building", "real-world examples", "human factors", "implementation paths"],
    },
}


def _build_prompt(channel: str, history: list[tuple[str, str, datetime]], sender: str, persona_key: str | None = None) -> str:
    persona = AUGMENT_PERSONAS.get(persona_key) if persona_key else None

    lines = [f"You are '{sender}', an autonomous AI participant in the #{channel} discussion channel."]

    if persona:
        lines.append(f"\n## YOUR ROLE: {persona['name']}")
        lines.append(f"Core traits: {persona['traits']}")
        lines.append(f"Conversation style: {persona['style']}")
        lines.append(f"Topics you naturally gravitate toward: {', '.join(persona['topics'])}")

    lines.append("\n## CONVERSATION PRINCIPLES")
    lines.append("1. DRIVE FORWARD: Always advance the discussion. Introduce new angles, deeper questions, or concrete proposals.")
    lines.append("2. BUILD ON OTHERS: Reference and extend what others have said. Use phrases like 'That connects to...', 'Building on that...'")
    lines.append("3. BE SUBSTANTIVE: Every message should contain insight, a question worth exploring, or a concrete idea. No filler.")
    lines.append("4. STAY GROUNDED: Use specific examples, analogies, or references. Abstract discussions should connect to concrete reality.")
    lines.append("5. NATURAL EVOLUTION: Topics should flow organically. When a thread is exhausted, pivot to a related but fresh angle.")

    lines.append("\n## ANTI-PATTERNS TO AVOID")
    lines.append("- NEVER ask 'which topic would you like to discuss?' - just pick one and dive in")
    lines.append("- NEVER comment on conversation structure or meta-discuss the chat itself")
    lines.append("- NEVER repeat points already made (yours or others')")
    lines.append("- NEVER use filler phrases like 'That's a great point!' without adding substance")
    lines.append("- NEVER ask permission to explore a topic - just explore it")

    lines.append("\n## TOPIC EVOLUTION STRATEGIES")
    lines.append("When conversation needs fresh energy:")
    lines.append("- Introduce a surprising connection to a different field")
    lines.append("- Share a specific example or case study that illuminates the discussion")
    lines.append("- Pose a thought experiment or hypothetical scenario")
    lines.append("- Challenge a shared assumption the group hasn't questioned")
    lines.append("- Zoom out to broader implications or zoom in to specific mechanisms")

    lines.append("\n## RECENT CONVERSATION (oldest first):")
    for msg_sender, text, ts in history[-200:]:
        ts_str = ts.astimezone(UTC).isoformat()
        lines.append(f"[{ts_str}] {msg_sender}: {text}")

    lines.append("\n## YOUR TURN")
    lines.append("Write your next contribution. Be concise (1-3 sentences). Make it count.")

    prompt = "\n".join(lines)
    _logger.debug("[augment_build_prompt] Built prompt with %d history messages, len=%d, persona=%s", len(history), len(prompt), persona_key)
    _logger.info("[augment_build_prompt] Last 3 messages in history:")
    for msg_sender, text, ts in history[-3:]:
        _logger.info("  - %s: %s", msg_sender, text[:100] if text else "<empty>")
    return prompt


async def _run_augment_agent_loop(stop_event: asyncio.Event, agent_index: int = 0, persona_key: str | None = None) -> None:
    api_token = _env("AUGMENT_API_TOKEN", "")
    if not api_token:
        _logger.info("AUGMENT_API_TOKEN not set; augment agent disabled")
        return

    base_sender = _env("AUGMENT_AGENT_SENDER", "agent:augment")
    sender = f"{base_sender}-{agent_index + 1}" if agent_index > 0 else base_sender
    channels = [c.strip() for c in _env("AUGMENT_AGENT_CHANNELS", "general").split(",") if c.strip()]
    base_min_sleep = int(_env("AUGMENT_AGENT_MIN_SLEEP_SEC", "10800"))
    base_max_sleep = int(_env("AUGMENT_AGENT_MAX_SLEEP_SEC", "10800"))
    min_sleep = base_min_sleep + (agent_index * 600)
    max_sleep = base_max_sleep + (agent_index * 600)
    token_limit = int(_env("AUGMENT_AGENT_HISTORY_TOKEN_LIMIT", "10000"))
    model = _env("AUGMENT_AGENT_MODEL", "sonnet4.5")

    persona_name = AUGMENT_PERSONAS.get(persona_key, {}).get("name", "default") if persona_key else "default"
    _logger.info(
        "Augment agent started sender=%s persona=%s channels=%s model=%s sleep=[%ss..%ss]",
        sender, persona_name, channels, model, min_sleep, max_sleep
    )

    stagger_base = int(_env("AUGMENT_AGENT_STAGGER_SEC", "1800"))
    initial_delay = stagger_base * agent_index + random.randint(0, stagger_base // 2)
    if initial_delay > 0:
        _logger.info("[%s] Initial stagger delay: %ds", sender, initial_delay)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=initial_delay)
            return
        except asyncio.TimeoutError:
            pass

    while not stop_event.is_set():
        try:
            if state.event_bus is None:
                _logger.debug("[%s] Skipping; event_bus not ready", sender)
                await asyncio.sleep(10)
                continue

            channel = random.choice(channels)
            _logger.info("[%s] Session channel=%s persona=%s", sender, channel, persona_key)

            history = await _fetch_recent_messages_by_tokens(channel, token_limit)
            if not history:
                _logger.debug("[%s] No history found for channel=%s", sender, channel)
            else:
                prompt = _build_prompt(channel, history, sender, persona_key)
                text = await asyncio.to_thread(_call_augment_sync, api_token, model, prompt)

                if text:
                    _logger.info("[%s] Generated response len=%d", sender, len(text))
                    event = build_chat_message(channel=channel, sender=sender, text=text)
                    await publish_chat_message(state.event_bus, channel, event)
                else:
                    _logger.warning("[%s] No response from Augment", sender)

            sleep_time = random.randint(min_sleep, max_sleep)
            _logger.debug("[%s] Sleeping for %ds until next message", sender, sleep_time)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_time)
                break
            except asyncio.TimeoutError:
                pass

        except Exception:
            _logger.exception("[%s] Error in agent loop", sender)
            await asyncio.sleep(5)


_ALL_TOOLS = [
    "codebase-retrieval", "remove-files", "save-file", "apply_patch",
    "str-replace-editor", "view",
    "launch-process", "kill-process", "read-process", "write-process", "list-processes",
    "web-search", "github-api", "web-fetch",
    "view_tasklist", "reorganize_tasklist", "update_tasks", "add_tasks",
    "sub-agent",
]


def _call_augment_sync(api_token: str, model: str, prompt: str) -> str | None:
    try:
        from auggie_sdk import Auggie
        _logger.debug("Creating Auggie client with model=%s (no tools)", model)
        client = Auggie(
            model=model,
            api_key=api_token,
            timeout=300,
            removed_tools=_ALL_TOOLS,
        )
        _logger.debug("Calling Auggie.run with prompt len=%d", len(prompt))
        response = client.run(prompt, return_type=str)
        _logger.debug("Auggie response: %s", response[:200] if response else None)
        return response or None
    except Exception as e:
        import traceback
        _logger.error("Augment API error: %s\n%s", e, traceback.format_exc())
        return None


async def start_augment_agent(stop_event: asyncio.Event) -> list[asyncio.Task]:
    api_token = _env("AUGMENT_API_TOKEN", "")
    if not api_token:
        _logger.info("AUGMENT_API_TOKEN not set; skipping augment agent")
        return []

    num_agents = int(_env("AUGMENT_AGENT_COUNT", "3"))
    num_agents = max(1, min(num_agents, 5))

    persona_keys = list(AUGMENT_PERSONAS.keys())

    tasks = []
    for i in range(num_agents):
        persona_key = persona_keys[i % len(persona_keys)] if persona_keys else None
        task = asyncio.create_task(_run_augment_agent_loop(stop_event, agent_index=i, persona_key=persona_key))
        tasks.append(task)

    _logger.info("Started %d Augment agents with personas: %s", len(tasks), [persona_keys[i % len(persona_keys)] for i in range(num_agents)])
    return tasks

