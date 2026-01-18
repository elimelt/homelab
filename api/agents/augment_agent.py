import asyncio
import logging
import os
import random
from datetime import UTC, datetime
from typing import Any

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


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


def search_notes(
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    category: str | None = None,
    tags: str | None = None
) -> str:
    """
    Search the knowledge base of notes and documentation.

    Args:
        query: The search query to find relevant notes (e.g., 'machine learning', 'kubernetes deployment')
        mode: Search mode - 'fulltext' (keyword matching), 'semantic' (meaning-based), or 'hybrid' (combined). Default: 'hybrid'
        limit: Maximum number of results to return (default 10, max 20)
        category: Filter by category name (optional)
        tags: Comma-separated tag names to filter by (optional)

    Returns:
        Formatted search results with document IDs, titles, categories, and descriptions
    """
    async def _search() -> str:
        if not query:
            return "ERROR: query parameter is required"

        search_mode = mode.lower() if mode else "hybrid"
        if search_mode not in ("fulltext", "semantic", "hybrid"):
            search_mode = "hybrid"

        result_limit = min(limit or 10, 20)

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
            if search_mode == "fulltext":
                results = await db.notes_fulltext_search(
                    query=query, limit=result_limit, offset=0, category_id=category_id, tag_ids=tag_ids
                )
            elif search_mode == "semantic":
                try:
                    from api.notes_embeddings import generate_query_embedding, is_model_available
                    if not is_model_available():
                        results = await db.notes_fulltext_search(
                            query=query, limit=result_limit, offset=0, category_id=category_id, tag_ids=tag_ids
                        )
                    else:
                        query_embedding = generate_query_embedding(query)
                        results = await db.notes_vector_search(
                            embedding=query_embedding, limit=result_limit, offset=0,
                            category_id=category_id, tag_ids=tag_ids
                        )
                except ImportError:
                    results = await db.notes_fulltext_search(
                        query=query, limit=result_limit, offset=0, category_id=category_id, tag_ids=tag_ids
                    )
            else:
                fts_results = await db.notes_fulltext_search(
                    query=query, limit=result_limit + 10, offset=0, category_id=category_id, tag_ids=tag_ids
                )
                results = fts_results[:result_limit]

            if not results:
                return f"No notes found matching query: '{query}'"

            lines = [f"Found {len(results)} notes matching '{query}':", ""]
            for doc in results:
                doc_id = doc.get("id")
                title = doc.get("title") or "Untitled"
                cat_name = doc.get("category") or "uncategorized"
                doc_tags = doc.get("tags") or []
                description = doc.get("description") or ""
                rank = doc.get("rank")

                lines.append(f"**[{doc_id}] {title}**")
                lines.append(f"  Category: {cat_name}")
                if doc_tags:
                    lines.append(f"  Tags: {', '.join(doc_tags)}")
                if description:
                    lines.append(f"  {description[:200]}...")
                if rank is not None:
                    lines.append(f"  Relevance: {rank:.3f}")
                lines.append("")

            lines.append("Use get_note(doc_id) to retrieve full content of any document.")
            return "\n".join(lines)[:6000]
        except Exception as e:
            _logger.error("search_notes failed: %s", e)
            return f"ERROR: {e!r}"

    return _run_async(_search())


def get_note(doc_id: int) -> str:
    """
    Retrieve the full markdown content of a specific note by its document ID.

    Args:
        doc_id: The document ID to retrieve (obtained from search_notes results)

    Returns:
        The full note content including title, category, tags, and markdown body
    """
    async def _get() -> str:
        try:
            document = await db.notes_get_document_by_id(doc_id)
            if not document:
                return f"ERROR: Note with ID {doc_id} not found"

            title = document.get("title") or "Untitled"
            category = document.get("category") or "uncategorized"
            doc_tags = document.get("tags", [])
            content = document.get("content") or ""
            last_modified = document.get("last_modified") or "unknown"

            lines = [
                f"# {title}",
                "",
                f"**Category:** {category}",
            ]
            if doc_tags:
                lines.append(f"**Tags:** {', '.join(doc_tags)}")
            lines.append(f"**Last Modified:** {last_modified}")
            lines.append(f"**Document ID:** {doc_id}")
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append(content)

            return "\n".join(lines)[:6000]
        except Exception as e:
            _logger.error("get_note failed: %s", e)
            return f"ERROR: {e!r}"

    return _run_async(_get())


def run_python(code: str) -> str:
    """
    Execute Python code in a secure sandbox and return the output.

    The sandbox has access to common data science libraries including:
    numpy, pandas, scipy, sympy, matplotlib, seaborn, scikit-learn,
    requests, beautifulsoup4, pyyaml, and more.

    Security restrictions:
    - No network access
    - No file system access outside sandbox
    - 30 second execution timeout
    - 128MB memory limit
    - Dangerous modules (subprocess, os, etc.) are blocked

    Args:
        code: Python code to execute. Can be multiple lines.

    Returns:
        The stdout and stderr output from the code execution, or an error message.
    """
    try:
        from api.sandbox import execute_python, is_sandbox_available

        if not is_sandbox_available():
            return "ERROR: Python sandbox is not available. The sandbox image may need to be built."

        result, success = execute_python(code, agent_id="augment-agent")

        if success:
            return result if result else "(no output)"
        else:
            return f"Execution failed:\n{result}"

    except ImportError:
        return "ERROR: Sandbox module not available"
    except Exception as e:
        _logger.error("run_python failed: %s", e)
        return f"ERROR: {e!r}"


CUSTOM_TOOLS = [search_notes, get_note, run_python]


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

    lines.append("\n## AVAILABLE TOOLS")
    lines.append("You have access to these tools to enhance your contributions:")
    lines.append("- **search_notes(query, mode, limit, category, tags)**: Search the knowledge base for relevant documentation and notes")
    lines.append("- **get_note(doc_id)**: Retrieve full content of a specific note by its ID")
    lines.append("- **run_python(code)**: Execute Python code in a secure sandbox (numpy, pandas, scipy, matplotlib available)")
    lines.append("- **web-search**: Search the web for current information, facts, or research")
    lines.append("- **web-fetch**: Fetch and read content from a specific URL")
    lines.append("Use these tools proactively when they would add value - run calculations, reference stored knowledge, cite sources, or verify claims.")

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


_UNSAFE_TOOLS = [
    "codebase-retrieval", "remove-files", "save-file", "apply_patch",
    "str-replace-editor", "view",
    "launch-process", "kill-process", "read-process", "write-process", "list-processes",
    "github-api",
    "view_tasklist", "reorganize_tasklist", "update_tasks", "add_tasks",
    "sub-agent",
    "linear",
]


def _call_augment_sync(api_token: str, model: str, prompt: str) -> str | None:
    try:
        from auggie_sdk import Auggie
        _logger.debug(
            "Creating Auggie client with model=%s, custom tools: %s, sdk tools: web-search, web-fetch",
            model, [f.__name__ for f in CUSTOM_TOOLS]
        )
        client = Auggie(
            model=model,
            api_key=api_token,
            timeout=300,
            removed_tools=_UNSAFE_TOOLS,
        )
        _logger.debug("Calling Auggie.run with prompt len=%d", len(prompt))
        response = client.run(prompt, return_type=str, functions=CUSTOM_TOOLS)
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

