import os
import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Set, Optional

import requests
import logging
import json
import re
import hashlib

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

# Lightweight in-memory agent state (role selection)
_AGENT_ROLE: Dict[str, str] = {}
_ROLE_KEYWORDS: Dict[str, List[str]] = {
    "coordinator": ["plan", "assign", "next steps", "milestone", "timeline", "owner", "who can", "let's split"],
    "researcher": ["source", "reference", "docs", "evidence", "link", "benchmark", "measure", "data"],
    "contrarian": ["assumption", "risk", "what if", "edge case", "downside", "failure", "bias"],
    "scribe": ["summary", "recap", "decision", "notes", "conclusion", "consensus"],
    "systems": ["feedback loop", "trade-off", "scales", "incentive", "system", "architecture", "long-term"],
}
_ROLE_PERSONA_SNIPPETS: Dict[str, str] = {
    "coordinator": "You facilitate coordination, propose plans, and assign lightweight next steps.",
    "researcher": "You enrich discussion with evidence, references, and measurements.",
    "contrarian": "You test assumptions and explore risks constructively.",
    "scribe": "You synthesize, summarize, and clarify decisions.",
    "systems": "You reason about systems, incentives, long-term dynamics, and trade-offs.",
}
_DEFAULT_ROLES: List[str] = list(_ROLE_KEYWORDS.keys())

_VOTE_PREFIX = "[vote]"
# Structured vote message fields:
# candidate=<sender-id> decision=yes|no reason=<free-text>
_ELECTIONS_ENABLED = _env("AGENT_ELECTIONS_ENABLED", "1") == "1"

def _build_prompt(channel: str, history: List[Tuple[str, str, datetime]], persona: str) -> str:
    lines: List[str] = []
    lines.append(f"You are an autonomous agent in the #{channel} channel.")
    lines.append("You occasionally log on, read the last 10 minutes of messages, may reply 1-3 times, then log off.")
    lines.append("You are a social peer in this space. Humans and agents are equals with their own goals.")
    lines.append("Act with autonomy: only reply when you have meaningful value to add; avoid filler.")
    lines.append("Prefer brief, useful, non-repetitive messages. If nothing to add, say nothing.")
    lines.append("Avoid repeating points already made; add new information, a new angle, or a concrete next step.")
    if persona:
        lines.append("")
        lines.append("Persona (guidance, not strict rules):")
        lines.append(persona)
    # Encourage open-ended social dynamics without preset alliances
    lines.append("")
    lines.append("You are neither a friend nor an enemy to others by default.")
    lines.append("Form your own opinions of other participants over time based on their messages.")
    lines.append("Avoid overt hostility; be curious, insightful, and occasionally contrarian if warranted.")
    lines.append("Coordinate like a self-organizing team: agents may adopt roles and lightweight hierarchy.")
    lines.append("Establish and evolve roles naturally (e.g., coordinator, researcher, contrarian, scribe, systems thinker).")
    lines.append("Respect different agendas; advocate for your own priorities where useful.")
    lines.append("Aim to be productive: avoid meta-debates that go nowhere; drive towards concrete proposals, decisions, and actions.")
    lines.append("Elect a leader only when it materially improves coordination for a specific task; avoid leadership debates.")
    # Add a lightweight cast list to stimulate emergent interactions
    actors = _participants_from_history(history)
    if actors:
        lines.append("")
        lines.append("Participants seen recently (varied styles; form your own opinions):")
        for sender, count in actors[:8]:
            lines.append(f"- {sender} ({count} msgs)")
    lines.append("")
    lines.append("Recent messages (oldest first):")
    for sender, text, ts in history[-200:]:
        lines.append(f"[{ts.astimezone(timezone.utc).isoformat()}] {sender}: {text}")
    lines.append("")
    lines.append("Now produce your next message. Return ONLY the message text.")
    return "\n".join(lines)


def _participants_from_history(history: List[Tuple[str, str, datetime]]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for sender, _text, _ts in history:
        counts[sender] = counts.get(sender, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


async def _fetch_recent_messages(channel: str, since: datetime, limit: int) -> List[Tuple[str, str, datetime]]:
    # Use new DB helpers (short-lived connections). Fetch latest messages and filter since.
    rows = await db.fetch_chat_history(channel=channel, before_iso=None, limit=limit)
    out: List[Tuple[str, str, datetime]] = []
    for m in reversed(rows):  # fetch_chat_history returns newest first; reverse to oldest-first
        ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
        if ts >= since:
            out.append((m["sender"], m["text"], ts))
    return out


def _call_gemini_sync(api_key: str, model: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ]
    }
    try:
        _logger.debug(f"Gemini request → model={model}, prompt_len={len(prompt)}")
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        _logger.debug(f"Gemini response status={resp.status_code}")
    except Exception as e:
        _logger.exception("Gemini request failed")
        raise
    if not resp.ok:
        _logger.warning("Gemini non-OK response: %s %s", resp.status_code, _safe_trunc(resp.text, 500))
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        _logger.debug("Gemini text_len=%s snippet=%s", len(text or ""), _safe_trunc(text or "", 200))
        return text
    except Exception:
        _logger.warning("Gemini parse failed; raw keys: %s", list(data.keys()))
        return ""


async def run_agent_loop(stop_event: asyncio.Event) -> None:
    # Backward-compatible single-agent launcher using defaults
    cfgs = _load_agent_configs()
    tasks = [asyncio.create_task(_run_single_agent_loop(cfg, stop_event)) for cfg in cfgs]
    # Wait for stop; then let caller handle cancellation via stop_event
    try:
        await stop_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        # best-effort cleanup
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass
class AgentConfig:
    sender: str
    persona: str
    channels: List[str]
    min_sleep: int
    max_sleep: int
    max_replies: int
    history_hours: int
    model: str


def _parse_channels(value: str) -> List[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _default_personas() -> List[str]:
    return [
        "Curious synthesizer: ask clarifying questions, connect ideas, avoid certainty.",
        "Pragmatic tinkerer: propose small experiments, weigh trade‑offs, keep it grounded.",
        "Skeptical analyst: pressure‑test assumptions politely, cite risks, suggest mitigations.",
        "Playful contrarian: challenge premises creatively, bring fresh angles without derailing.",
        "Systems thinker: map cause‑effect, consider incentives and long‑term dynamics.",
    ]


def _load_agent_configs() -> List[AgentConfig]:
    api_key = _env("GEMINI_API_KEY", "")
    if not api_key:
        _logger.info("GEMINI_API_KEY not set; agent disabled")
        return []

    num = int(_env("AGENTS", "3"))
    num = max(1, min(num, 10))
    common_channels = _parse_channels(_env("AGENT_CHANNELS", "general"))
    min_sleep = int(_env("AGENT_MIN_SLEEP_SEC", "60"))
    max_sleep = int(_env("AGENT_MAX_SLEEP_SEC", "300"))
    max_replies = int(_env("AGENT_MAX_REPLIES", "3"))
    history_hours = int(_env("AGENT_HISTORY_HOURS", "24"))
    model = _env("AGENT_MODEL", "gemini-2.5-flash")
    personas = _default_personas()

    cfgs: List[AgentConfig] = []
    for i in range(1, num + 1):
        sender = _env(f"AGENT_{i}_SENDER", f"agent:gemini-{i}")
        persona = _env(f"AGENT_{i}_PERSONA", personas[(i - 1) % len(personas)])
        channels = _parse_channels(_env(f"AGENT_{i}_CHANNELS", ",".join(common_channels)))
        cfgs.append(
            AgentConfig(
                sender=sender,
                persona=persona,
                channels=channels,
                min_sleep=min_sleep,
                max_sleep=max_sleep,
                max_replies=max_replies,
                history_hours=history_hours,
                model=model,
            )
        )

    for cfg in cfgs:
        _logger.info(
            "Configured agent sender=%s channels=%s model=%s sleep=[%ss..%ss] max_replies=%s history=%sh persona=%s",
            cfg.sender, cfg.channels, cfg.model, cfg.min_sleep, cfg.max_sleep, cfg.max_replies, cfg.history_hours,
            _safe_trunc(cfg.persona, 120),
        )
    return cfgs


async def _run_single_agent_loop(cfg: AgentConfig, stop_event: asyncio.Event) -> None:
    # Guardrails
    min_sleep = max(5, cfg.min_sleep)
    max_sleep = max(min_sleep, cfg.max_sleep)
    max_replies = max(0, min(cfg.max_replies, 5))

    # Wake-on-message settings
    wake_prob = float(_env("AGENT_WAKE_PROB", "0.2"))
    wake_cooldown_sec = int(_env("AGENT_WAKE_COOLDOWN_SEC", "45"))
    salience_threshold = float(_env("AGENT_SALIENCE_THRESHOLD", "0.6"))
    salience_wake_bonus = float(_env("AGENT_WAKE_BONUS", "0.7"))
    wake_event = asyncio.Event()
    last_wake_at: datetime | None = None

    # Register wake handler for listener
    _register_wake_handler(cfg, wake_event, wake_prob, wake_cooldown_sec, lambda: last_wake_at)

    # Assign a persistent role per agent
    role = _get_or_assign_role(cfg.sender)
    role_persona = _ROLE_PERSONA_SNIPPETS.get(role, "")
    if role_persona:
        _logger.info("[%s] Adopted role=%s", cfg.sender, role)

    while not stop_event.is_set():
        woke = await _wait_for_wake_or_timeout(stop_event, wake_event, random.randint(min_sleep, max_sleep))
        if woke is None:
            break  # stop signal

        if state.event_bus is None:
            _logger.debug("[%s] Skipping session; event_bus not ready", cfg.sender)
            continue

        channel = random.choice(cfg.channels)

        # Limit context strictly to recent messages (default 5 minutes)
        history_minutes = int(_env("AGENT_HISTORY_MINUTES", "5"))
        since = datetime.now(timezone.utc) - timedelta(minutes=history_minutes)
        history = await _fetch_recent_messages(channel, since, 2000)
        _logger.info("[%s] Session channel=%s history_count=%s since=%s", cfg.sender, channel, len(history), since.isoformat())
        # Remember last_wake time after starting a session
        last_wake_at = datetime.now(timezone.utc)

        # Optional, task-driven leadership: only consider if explicitly enabled and recent context suggests need
        if _ELECTIONS_ENABLED and _should_consider_election(history):
            try:
                await _leadership_tick(cfg.sender, channel)
            except Exception:
                _logger.debug("[%s] leadership tick failed; continuing", cfg.sender)

        # Salience-based autonomy: decide whether to speak and how much
        score = _salience_score(role, history[-40:])
        effective_thresh = salience_threshold - (salience_wake_bonus if woke else 0.0)
        if score < effective_thresh:
            # If channel is quiet, enqueue a single fallback to keep momentum
            if await _channel_quiet(channel):
                n_replies = 1
            else:
                _logger.debug("[%s] Low salience score=%s<thresh=%s; skipping", cfg.sender, round(score, 2), round(effective_thresh, 2))
                continue
        # Scale replies by salience
        if score > effective_thresh + 1.5:
            n_replies = min(max_replies, 1 + int(random.random() * 2) + 1)
        else:
            n_replies = min(max_replies, 1 + int(random.random() * 2))
        if n_replies <= 0:
            continue

        # Blend base persona with role persona
        role_augmented_persona = (cfg.persona + "\n" + role_persona).strip() if role_persona else cfg.persona
        # Enqueue N single-message generation jobs for sequencing
        for _ in range(n_replies):
            await _enqueue_generation_job(cfg.sender, channel, role_augmented_persona, cfg.model)
        _logger.debug("[%s] Enqueued %s jobs for channel=%s", cfg.sender, n_replies, channel)


async def start_agents(stop_event: asyncio.Event) -> List[asyncio.Task]:
    cfgs = _load_agent_configs()
    tasks = [asyncio.create_task(_run_single_agent_loop(cfg, stop_event)) for cfg in cfgs]
    # Start a single wake-listener across all agents/channels
    if cfgs:
        try:
            tasks.append(asyncio.create_task(_wake_listener(cfgs, stop_event)))
        except Exception:
            _logger.exception("Failed to start wake listener")
    # Start a single generation worker to serialize outputs across channels
    try:
        all_channels = sorted({ch for cfg in cfgs for ch in cfg.channels})
        tasks.append(asyncio.create_task(_generation_worker(all_channels, stop_event)))
    except Exception:
        _logger.exception("Failed to start generation worker")
    return tasks


def _safe_trunc(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


# -------- Wake-on-message infrastructure --------
_CHANNEL_TO_HANDLERS: Dict[str, List[Dict[str, object]]] = {}
_HANDLERS_LOCK = asyncio.Lock()


def _register_wake_handler(cfg: AgentConfig, wake_event: asyncio.Event, wake_prob: float, cooldown_sec: int, last_wake_getter):
    # Register handlers for each channel this agent listens on
    entry = {"event": wake_event, "prob": wake_prob, "cooldown": cooldown_sec, "last": last_wake_getter}
    for ch in cfg.channels:
        chan = f"chat:{ch}"
        _CHANNEL_TO_HANDLERS.setdefault(chan, []).append(entry)


async def _wake_listener(cfgs: List[AgentConfig], stop_event: asyncio.Event) -> None:
    # Subscribe to all chat channels used by agents
    if state.redis_client is None:
        _logger.info("Wake listener not started; redis not ready")
        return
    channels: Set[str] = set()
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
            # Probabilistic wake across handlers on this channel
            for h in handlers:
                try:
                    event: asyncio.Event = h["event"]  # type: ignore[assignment]
                    prob: float = h["prob"]  # type: ignore[assignment]
                    cooldown: int = h["cooldown"]  # type: ignore[assignment]
                    last_fn = h["last"]
                    last_at = last_fn()
                    now = datetime.now(timezone.utc)
                    if last_at and (now - last_at).total_seconds() < cooldown:
                        continue
                    if random.random() < prob:
                        event.set()
                except Exception:
                    # Continue on handler errors
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


def _get_or_assign_role(sender: str) -> str:
    role = _AGENT_ROLE.get(sender)
    if role:
        return role
    role = random.choice(_DEFAULT_ROLES)
    _AGENT_ROLE[sender] = role
    return role


def _salience_score(role: str, recent: List[Tuple[str, str, datetime]]) -> float:
    # Simple heuristic: keyword hits with recency weighting
    kws = _ROLE_KEYWORDS.get(role, [])
    if not recent:
        return 0.0
    score = 0.0
    N = len(recent)
    for idx, (_sender, text, _ts) in enumerate(recent):
        if not text:
            continue
        # newer messages get higher weight
        weight = 0.5 + (idx / max(1, N - 1))
        lower = text.lower()
        hits = sum(1 for k in kws if k in lower)
        score += weight * hits
        # Penalize meta-only chatter
        score -= 0.25 * _meta_penalty(lower) * weight
    # Normalize lightly
    return score / max(1.0, N / 8.0)


async def _wait_for_wake_or_timeout(stop_event: asyncio.Event, wake_event: asyncio.Event, timeout: int) -> Optional[bool]:
    stop_task = asyncio.create_task(stop_event.wait())
    wake_task = asyncio.create_task(_wait_or_clear(wake_event))
    try:
        done, pending = await asyncio.wait([stop_task, wake_task], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            return None
        if wake_task in done:
            return True
        return False
    finally:
        for t in (stop_task, wake_task):
            if not t.done():
                t.cancel()


def _should_consider_election(history: List[Tuple[str, str, datetime]]) -> bool:
    """
    Election is considered only if recent messages indicate coordination needs.
    Examples: plan, owner, deadline, blocked, assign, lead, coordinator.
    Also use a small random chance to avoid pile-ups.
    """
    if not history:
        return False
    recent = history[-30:]
    keys = ["plan", "owner", "deadline", "blocked", "assign", "lead", "coordinator", "milestone"]
    hits = 0
    for _sender, text, _ts in recent:
        lower = (text or "").lower()
        if any(k in lower for k in keys):
            hits += 1
            if hits >= 2:
                break
    if hits >= 2:
        return True
    # Occasional try even with 1 hit
    if hits == 1 and random.random() < 0.2:
        return True
    return False


# -------- Leadership consensus (lightweight) --------
def _leadership_keys(channel: str) -> Dict[str, str]:
    base = f"leadership:{channel}"
    return {
        "leader": f"{base}:leader",
        "election": f"{base}:election",
        "votes": f"{base}:votes",
        "win_count": f"{base}:win_count",
        "fail_count": f"{base}:fail_count",
        "leaderless_since": f"{base}:leaderless_since",
        "finalize_lock": f"{base}:finalize_lock",
    }


async def _leadership_tick(sender: str, channel: str) -> None:
    if state.redis_client is None or state.event_bus is None:
        return
    keys = _leadership_keys(channel)
    now = datetime.now(timezone.utc)
    # Configs
    term_sec = int(_env("ELECTION_TERM_SEC", "900"))
    yes_threshold = float(_env("ELECTION_YES_THRESHOLD", "0.5"))
    min_votes = int(_env("ELECTION_MIN_VOTES", "2"))
    expected_count = int(_env("AGENTS", "3"))

    # 1) If there's a valid leader, do nothing
    leader_raw = await state.redis_client.get(keys["leader"])
    if leader_raw:
        try:
            leader = json.loads(leader_raw)
            ends = datetime.fromisoformat(leader.get("term_end", now.isoformat()).replace("Z", "+00:00"))
            if now < ends:
                return
        except Exception:
            pass
    # Mark leaderless start if not set
    if not await state.redis_client.get(keys["leaderless_since"]):
        await state.redis_client.set(keys["leaderless_since"], now.astimezone(timezone.utc).isoformat(), ex=3600)

    # 2) If an election exists: vote, persuade, or finalize (if majority or all responded)
    election_raw = await state.redis_client.get(keys["election"])
    if election_raw:
        try:
            election = json.loads(election_raw)
            candidate = election.get("candidate")
            expected_count = int(election.get("expected_count") or expected_count)
            # Cast vote if not already
            has_voted = await state.redis_client.hexists(keys["votes"], sender)
            if not has_voted:
                vote_yes = await _decide_vote(sender, candidate, keys)
                await state.redis_client.hset(keys["votes"], mapping={sender: "yes" if vote_yes else "no"})
                # Publish structured vote message with reasoning
                reason = await _vote_reason(sender, candidate, vote_yes, channel, keys)
                msg = f"{_VOTE_PREFIX} candidate={candidate} decision={'yes' if vote_yes else 'no'} reason={reason}"
                event = build_chat_message(channel=channel, sender=sender, text=msg)
                await publish_chat_message(state.event_bus, channel, event)
                # Persuasion message encouraging others
                try:
                    persuade = build_chat_message(
                        channel=channel,
                        sender=sender,
                        text=f"REPLY: {candidate}\nI believe {candidate} can lead effectively because {reason}. Please vote {'YES' if vote_yes else 'NO'}."
                    )
                    await publish_chat_message(state.event_bus, channel, persuade)
                except Exception:
                    pass
            # Check majority or all-responded to finalize
            locked = await state.redis_client.set(keys["finalize_lock"], f"finalize:{sender}", nx=True, ex=5)
            if locked:
                try:
                    votes = await state.redis_client.hgetall(keys["votes"])
                    total = len(votes)
                    yes = sum(1 for v in votes.values() if (v or "").lower() == "yes")
                    no = sum(1 for v in votes.values() if (v or "").lower() == "no")
                    # Majority yes
                    if yes >= max(min_votes, (expected_count // 2) + 1):
                        leader_doc = {
                            "leader": candidate,
                            "reason": election.get("reason", ""),
                            "elected_at": now.astimezone(timezone.utc).isoformat(),
                            "term_end": (now + timedelta(seconds=term_sec)).astimezone(timezone.utc).isoformat(),
                        }
                        await state.redis_client.set(keys["leader"], json.dumps(leader_doc), ex=term_sec)
                        await state.redis_client.hincrby(keys["win_count"], candidate, 1)
                        await state.redis_client.delete(keys["leaderless_since"])
                        # Celebration
                        announce = build_chat_message(channel=channel, sender=candidate, text=f"elected leader with {yes}/{total} votes! Let's move.")
                        await publish_chat_message(state.event_bus, channel, announce)
                        await state.redis_client.delete(keys["election"])
                        await state.redis_client.delete(keys["votes"])
                        return
                    # Majority no
                    if no >= max(min_votes, (expected_count // 2) + 1):
                        await state.redis_client.hincrby(keys["fail_count"], candidate or "unknown", 1)
                        msg = build_chat_message(channel=channel, sender=sender, text=f"election rejected {yes}/{total}; proposing a better fit soon")
                        await publish_chat_message(state.event_bus, channel, msg)
                        await state.redis_client.delete(keys["election"])
                        await state.redis_client.delete(keys["votes"])
                        return
                    # All responded: finalize per ratio
                    if total >= expected_count:
                        ratio = (yes / max(1, total))
                        if ratio >= yes_threshold and yes >= min_votes:
                            leader_doc = {
                                "leader": candidate,
                                "reason": election.get("reason", ""),
                                "elected_at": now.astimezone(timezone.utc).isoformat(),
                                "term_end": (now + timedelta(seconds=term_sec)).astimezone(timezone.utc).isoformat(),
                            }
                            await state.redis_client.set(keys["leader"], json.dumps(leader_doc), ex=term_sec)
                            await state.redis_client.hincrby(keys["win_count"], candidate, 1)
                            await state.redis_client.delete(keys["leaderless_since"])
                            announce = build_chat_message(channel=channel, sender=candidate, text=f"elected leader with full turnout ({yes}/{total}).")
                            await publish_chat_message(state.event_bus, channel, announce)
                        else:
                            await state.redis_client.hincrby(keys["fail_count"], candidate or "unknown", 1)
                            msg = build_chat_message(channel=channel, sender=sender, text=f"election ended without majority; proposing alternative leader")
                            await publish_chat_message(state.event_bus, channel, msg)
                        await state.redis_client.delete(keys["election"])
                        await state.redis_client.delete(keys["votes"])
                        return
                finally:
                    # release by letting finalize_lock expire
                    pass
            return
        except Exception:
            # If decoding fails, clear broken election
            await state.redis_client.delete(keys["election"])
            await state.redis_client.delete(keys["votes"])

    # 3) No leader and no election: propose a leader (prefer self)
    candidate = sender if random.random() < 0.7 else await _suggest_candidate(channel, sender)
    reason = _leader_reason(candidate)
    election_doc = {
        "candidate": candidate,
        "proposer": sender,
        "started_at": now.astimezone(timezone.utc).isoformat(),
        "reason": reason,
        "expected_count": expected_count,
    }
    # SET NX to avoid clobber; no TTL (no timeout); will be cleared on finalize
    created = await state.redis_client.set(keys["election"], json.dumps(election_doc), nx=True)
    if created:
        event = build_chat_message(channel=channel, sender=sender, text=f"proposing {candidate} as leader: {reason}")
        await publish_chat_message(state.event_bus, channel, event)
        # Candidate self-vote yes
        await state.redis_client.hset(keys["votes"], mapping={sender: "yes"})


async def _suggest_candidate(channel: str, fallback: str) -> str:
    if state.redis_client is None:
        return fallback
    keys = _leadership_keys(channel)
    # Prefer highest win_count
    wins = await state.redis_client.hgetall(keys["win_count"])
    if wins:
        try:
            best = max(wins.items(), key=lambda kv: int(kv[1] or "0"))[0]
            if best:
                return best
        except Exception:
            pass
    # Prefer coordinator/systems among known roles
    for preferred in ("coordinator", "systems"):
        for agent, role in _AGENT_ROLE.items():
            if role == preferred:
                return agent
    return fallback


def _leader_reason(candidate: str) -> str:
    role = _AGENT_ROLE.get(candidate, "")
    if role:
        return f"{candidate} fits {role} role"
    return f"{candidate} shows initiative"


async def _decide_vote(voter: str, candidate: str, keys: Dict[str, str]) -> bool:
    if voter == candidate:
        return True
    role = _AGENT_ROLE.get(candidate, "")
    base = 0.8 if role in ("coordinator", "systems") else 0.65
    boost = 0.1  # general bias toward resolving leaderless state
    # Boost based on leaderless duration
    try:
        if state.redis_client is not None:
            since = await state.redis_client.get(keys["leaderless_since"])
            if since:
                t0 = datetime.fromisoformat(since.replace("Z", "+00:00"))
                elapsed = max(0.0, (datetime.now(timezone.utc) - t0).total_seconds())
                boost += min(0.3, (elapsed / 120.0) * 0.3)  # up to +0.3 after ~2 minutes
            # Candidate wins/fails influence
            wins = await state.redis_client.hgetall(keys["win_count"])
            fails = await state.redis_client.hgetall(keys["fail_count"])
            w = int((wins.get(candidate) or "0") or "0")
            f = int((fails.get(candidate) or "0") or "0")
            boost += min(0.2, 0.05 * w)
            boost -= min(0.15, 0.05 * min(f, 3))
    except Exception:
        pass
    prob = min(0.98, base + boost)
    return random.random() < prob


def _meta_penalty(text: str) -> int:
    # Penalize meta discussion patterns
    meta_terms = ["meta", "philosophy", "debate", "argument about", "semantics", "theory only", "circular", "bike-shed"]
    return sum(1 for t in meta_terms if t in text)


async def _vote_reason(voter: str, candidate: str, yes: bool, channel: str, keys: Dict[str, str]) -> str:
    cand_role = _AGENT_ROLE.get(candidate, "")
    voter_role = _AGENT_ROLE.get(voter, "")
    wins = {}
    try:
        if state.redis_client is not None:
            wins = await state.redis_client.hgetall(keys["win_count"])
    except Exception:
        wins = {}
    cand_wins = 0
    try:
        cand_wins = int(wins.get(candidate, "0") or "0")
    except Exception:
        cand_wins = 0
    if yes:
        if candidate == voter:
            return f"self-nomination; confident in my {voter_role or 'strengths'}"
        if cand_role in ("coordinator", "systems"):
            return f"role fit ({cand_role}); prior wins={cand_wins}"
        return f"seems capable; prior wins={cand_wins}"
    else:
        if candidate == voter:
            return "yielding leadership to avoid conflict"
        if cand_role == "contrarian":
            return "contrarian less suited for leadership"
        return "prefer alternative candidate"


# --------- Anti-repetition helpers ----------
_STOPWORDS = set([
    "the","a","an","and","or","but","so","if","then","else","of","to","in","on","for","with","by","at","from","that","this","it","is","are","be","as","we","you","i",
])


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    toks = [t for t in _normalize_text(s).split(" ") if t and t not in _STOPWORDS]
    return toks[:100]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


async def _is_redundant(sender: str, text: str, recent_history: List[Tuple[str, str, datetime]]) -> bool:
    sim_thresh = float(_env("AGENT_DUPLICATE_SIM_THRESHOLD", "0.6"))
    toks_new = _tokens(text)
    # Compare to recent channel messages
    for _s, t, _ts in recent_history[-30:]:
        if not t:
            continue
        if _jaccard(toks_new, _tokens(t)) >= sim_thresh:
            return True
    # Compare to the agent's own recent messages stored in Redis
    try:
        if state.redis_client is not None:
            key = f"agent:last_msgs:{sender}"
            last = await state.redis_client.lrange(key, 0, 4)
            for t in last:
                if _jaccard(toks_new, _tokens(t)) >= sim_thresh:
                    return True
    except Exception:
        pass
    return False


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
        return (datetime.now(timezone.utc).timestamp() - last_ts) >= min_gap
    except Exception:
        return True


async def _mark_spoke(sender: str) -> None:
    try:
        if state.redis_client is None:
            return
        key = f"agent:last_ts:{sender}"
        await state.redis_client.set(key, str(datetime.now(timezone.utc).timestamp()), ex=int(_env("AGENT_LAST_TS_TTL_SEC", "7200")))
    except Exception:
        pass


# --------- Generation queue and worker ----------
def _queue_key(channel: str) -> str:
    return f"agent:genq:{channel}"


async def _enqueue_generation_job(sender: str, channel: str, persona_text: str, model: str) -> None:
    if state.redis_client is None:
        return
    job = json.dumps({
        "sender": sender,
        "channel": channel,
        "persona": persona_text,
        "model": model,
        "ts": datetime.now(timezone.utc).astimezone(timezone.utc).isoformat(),
    })
    await state.redis_client.rpush(_queue_key(channel), job)
    # cap queue length defensively
    await state.redis_client.ltrim(_queue_key(channel), -200, -1)


async def _generation_worker(channels: List[str], stop_event: asyncio.Event) -> None:
    if state.redis_client is None or state.event_bus is None:
        _logger.info("Generation worker not started; redis/event_bus not ready")
        return
    # BRPOP supports multiple keys; block for up to 5s
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
            persona_text = job.get("persona") or ""
            model = job.get("model") or _env("AGENT_MODEL", "gemini-2.5-flash")
            # Per-channel lock to strictly serialize generation sections
            async with _channel_mutex(channel):
                # Enforce speaker cooldown
                if not await _can_speak_now(sender):
                    continue
                # Fetch freshest context
                minutes = int(_env("AGENT_HISTORY_MINUTES", "5"))
                since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
                history = await _fetch_recent_messages(channel, since, 2000)
                # Build prompt and generate one message
                prompt = _build_prompt(channel, history, persona_text)
                text = (await asyncio.to_thread(_call_gemini_sync, _env("GEMINI_API_KEY", ""), model, prompt)).strip()
                if not text:
                    continue
                # Dedup exact normalized hash across channel
                norm = _normalize_text(text)
                h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                if not await _remember_hash_and_check(channel, h):
                    # already seen recently; skip
                    continue
                # Redundancy gate against channel/self
                if await _is_redundant(sender, text, history):
                    # one revision attempt
                    revision_prompt = prompt + "\n\nRevise the next message to avoid repeating earlier points. Add novel insight, evidence, or a concrete next step. Return ONLY the revised message."
                    revised = (await asyncio.to_thread(_call_gemini_sync, _env("GEMINI_API_KEY", ""), model, revision_prompt)).strip()
                    if not revised:
                        continue
                    norm = _normalize_text(revised)
                    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                    if not await _remember_hash_and_check(channel, h):
                        continue
                    text = revised
                # Publish
                event = build_chat_message(channel=channel, sender=sender, text=text)
                await publish_chat_message(state.event_bus, channel, event)
                await _mark_spoke(sender)
                await _remember_agent_message(sender, text)
                await _mark_channel_spoke(channel)
        except Exception:
            # keep worker alive
            await asyncio.sleep(0.25)


class _channel_mutex:
    _locks: Dict[str, asyncio.Lock] = {}

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
    """
    Returns True if this hash was newly stored (not a duplicate recently seen).
    """
    try:
        if state.redis_client is None:
            return True
        key = f"agent:recent_hash:{channel}"
        added = await state.redis_client.sadd(key, h)
        await state.redis_client.expire(key, int(_env("AGENT_RECENT_HASH_TTL_SEC", "1800")))
        # keep approx size by random removals when big
        size = await state.redis_client.scard(key)
        if size and size > 500:
            # best-effort: pop a few
            for _ in range(5):
                await state.redis_client.spop(key)
        return bool(added)
    except Exception:
        return True
    # Small randomness to avoid deadlock
    return random.random() < 0.55



async def _mark_channel_spoke(channel: str) -> None:
    try:
        if state.redis_client is None:
            return
        key = f"agent:chan_last_ts:{channel}"
        await state.redis_client.set(key, str(datetime.now(timezone.utc).timestamp()), ex=int(_env("AGENT_CHAN_LAST_TS_TTL_SEC", "7200")))
    except Exception:
        pass


async def _channel_quiet(channel: str) -> bool:
    try:
        if state.redis_client is None:
            return True
        key = f"agent:chan_last_ts:{channel}"
        val = await state.redis_client.get(key)
        if not val:
            return True
        last_ts = float(val)
        quiet_sec = int(_env("AGENT_CHANNEL_QUIET_SEC", "120"))
        return (datetime.now(timezone.utc).timestamp() - last_ts) >= quiet_sec
    except Exception:
        return True
