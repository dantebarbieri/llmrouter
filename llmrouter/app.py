"""OpenAI-compatible router in front of LiteLLM + request log UI.

Classifies each /v1/chat/completions request and picks a tier based on
configured policy (default: local / local-thinking / haiku / sonnet / opus),
then forwards to LiteLLM with the model field rewritten. Two classification
modes (configurable):

- classifier.mode=local: one call to a local model (via LiteLLM) returns
  {has_secret, secret_values, complexity} as JSON. Regex secret_patterns are
  still run as a safety-net union. Fallback to heuristic + tiebreaker on any
  error.
- classifier.mode=hybrid|haiku: legacy pipeline — regex+heuristic first,
  Haiku tiebreaker when ambiguous.

On local-tier selection with an unreachable Ollama, transparently promotes
to fallback.cloud_tier and fires an NTFY alert on health transitions. Every
routed request is logged to a local SQLite DB (FTS5-indexed on keywords + a
message preview, redacted when a secret was detected), exposed via a minimal
`/ui/` web UI at the same port.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from llmrouter.config import Config, RuntimeEnv, load_config

# --- bootstrap config + env at import time ---------------------------------

ENV: RuntimeEnv = RuntimeEnv.from_env()
CFG: Config = load_config(ENV.config_path)

logging.basicConfig(
    level=ENV.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOG = logging.getLogger("llmrouter")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Health-probe state (process-local, ttl'd).
_ollama_state: dict[str, Any] = {"ok": True, "checked_at": 0.0}
_ollama_lock = asyncio.Lock()


# --- health + ntfy ---------------------------------------------------------


async def ollama_healthy() -> bool:
    now = time.time()
    if now - _ollama_state["checked_at"] < CFG.limits.health_ttl_s:
        return _ollama_state["ok"]
    async with _ollama_lock:
        if time.time() - _ollama_state["checked_at"] < CFG.limits.health_ttl_s:
            return _ollama_state["ok"]
        was_ok = _ollama_state["ok"]
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{ENV.ollama_url}/api/tags")
            ok = r.status_code == 200
        except Exception as e:
            ok = False
            LOG.warning("ollama probe failed: %s", e)
        _ollama_state["ok"] = ok
        _ollama_state["checked_at"] = time.time()
        if ok != was_ok and CFG.fallback.ntfy_on_transition:
            asyncio.create_task(_notify_ntfy(
                title=f"Ollama {'recovered' if ok else 'unreachable'}",
                message=f"Ollama at {ENV.ollama_url} is now {'reachable' if ok else 'unreachable'}",
                priority="default" if ok else "high",
            ))
        return ok


async def _notify_ntfy(title: str, message: str, priority: str = "default") -> None:
    if not ENV.ntfy_url:
        return
    headers = {"Title": title, "Priority": priority, "Tags": "robot"}
    if ENV.ntfy_token:
        headers["Authorization"] = f"Bearer {ENV.ntfy_token}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            await c.post(f"{ENV.ntfy_url}/{ENV.ntfy_topic}", content=message.encode(), headers=headers)
    except Exception as e:
        LOG.warning("ntfy send failed: %s", e)


# --- classification --------------------------------------------------------


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _messages_text(messages: list[dict[str, Any]], include_system: bool = True) -> str:
    parts: list[str] = []
    for m in messages:
        if not include_system and m.get("role") == "system":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
    return "\n".join(parts)


def _scrub_secrets(text: str) -> str:
    """Replace each secret_patterns match with a [REDACTED:<name>] marker."""
    for sp in CFG.secret_patterns:
        text = sp.compiled.sub(f"[REDACTED:{sp.name}]", text)
    return text


def _scrub_with_values(text: str, values: list[str], marker: str = "llm_classifier") -> str:
    """Substring-replace each given value with [REDACTED:<marker>]."""
    if not text or not values:
        return text
    for v in sorted({v for v in values if v and len(v) >= 3}, key=len, reverse=True):
        text = text.replace(v, f"[REDACTED:{marker}]")
    return text


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _messages_text([m])
    return ""


def _clean_user_text(text: str) -> str:
    """Apply each metadata_strip_patterns regex to the latest user message."""
    for ms in CFG.metadata_strip_patterns:
        text = ms.compiled.sub("\n", text)
    return text.strip()


def regex_secret_hit(body: dict[str, Any]) -> str | None:
    """Run secret_patterns against the latest user message. Returns pattern name or None."""
    latest_user = _latest_user_text(body.get("messages", []))
    return next((sp.name for sp in CFG.secret_patterns if sp.compiled.search(latest_user)), None)


def heuristic_tier(body: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """First-match-wins rules. Returns (tier_or_None, signals)."""
    h = CFG.heuristic
    messages = body.get("messages", [])
    text = _messages_text(messages)
    latest_user = _latest_user_text(messages)
    latest_lower = latest_user.lower()
    tokens = _approx_tokens(text)
    has_tools = bool(body.get("tools") or body.get("functions") or body.get("tool_choice"))
    hit_secret = regex_secret_hit(body)
    hit_step = next((k for k in h.step_keywords if k in latest_lower), None)
    has_code = "```" in text
    signals: dict[str, Any] = {
        "tokens": tokens,
        "has_tools": has_tools,
        "has_code": has_code,
        "step_keyword": hit_step,
        "secret_keyword": hit_secret,
    }
    if hit_secret:
        return h.secret_hit_tier, signals
    if tokens > h.large_token_threshold:
        return h.large_token_tier, signals
    if has_tools:
        return h.tools_tier, signals
    if tokens < h.small_token_threshold and not has_code and not hit_step:
        return h.small_token_tier, signals
    if has_code or hit_step:
        return h.ambiguous_tier, signals
    return CFG.classifier.heuristic_default_tier, signals


async def haiku_tiebreaker(
    body: dict[str, Any],
    classification_text: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    if classification_text is None:
        classification_text = _messages_text(body.get("messages", []))
    text = classification_text[:4000]
    rate_prompt = (
        "Rate the complexity of this request on a 1-5 scale. "
        "1=trivial factual, 2=simple generation, 3=moderate reasoning, "
        "4=complex multi-step, 5=deep architectural/novel. Output ONLY the digit.\n\n"
        f"---\n{text}\n---"
    )
    details: dict[str, Any] = {"prompt": rate_prompt, "response": None, "digit": None, "error": None}
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.post(
                f"{ENV.litellm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {ENV.litellm_api_key}"},
                json={
                    "model": CFG.classifier.tiebreaker_model,
                    "messages": [{"role": "user", "content": rate_prompt}],
                    "max_tokens": 4,
                    "temperature": 0,
                },
            )
        content = r.json()["choices"][0]["message"]["content"]
        details["response"] = content
        digit = next((ch for ch in content if ch.isdigit()), None)
        details["digit"] = digit
        if digit is None:
            raise ValueError(f"no digit in tiebreaker response: {content!r}")
    except Exception as e:
        LOG.warning("tiebreaker failed, defaulting to %s: %s", CFG.fallback.cloud_tier, e)
        details["error"] = str(e)
        return CFG.fallback.cloud_tier, "tiebreaker-failed-default", details
    tier = CFG.classifier.tiebreaker_to_tier.get(digit, CFG.fallback.cloud_tier)
    return tier, "haiku-tiebreaker", details


_CLASSIFIER_PROMPT_TEMPLATE = (
    "You are a request classifier. Classify the user request below and return\n"
    "ONLY a JSON object. No prose, no markdown fences, no thinking tags.\n"
    "\n"
    "Fields:\n"
    "- has_secret (bool): true iff the request literally contains a credential,\n"
    "  API key, password, bearer token, private key, or .env content. Discussion\n"
    "  OF secrets is NOT a secret. When in doubt, return false — a regex catches\n"
    "  obvious cases too.\n"
    "- secret_values (array of strings): when has_secret=true, the literal\n"
    "  exact substring(s) from the request containing the credential value(s),\n"
    "  verbatim — preserving case, punctuation, and surrounding quotes/braces\n"
    "  ONLY if they're part of the secret. The caller will find-and-replace\n"
    "  these strings, so they must match exactly. Include just the value, not\n"
    "  the surrounding label (e.g. for `password: hunter2`, return [\"hunter2\"]).\n"
    "  Empty array when has_secret=false.\n"
    "- complexity (int 1-5):\n"
    "    1=trivial factual, 2=simple generation, 3=moderate reasoning,\n"
    "    4=complex multi-step, 5=deep architectural / novel.\n"
    "  Calibration: simple tool-use requests (\"use tool X to look up Y\",\n"
    "  \"search for Z\", \"geocode this address\") are complexity 2 even when\n"
    "  they sound technical — the tool does the heavy lifting. Reserve 3+ for\n"
    "  requests that require planning across multiple tool calls, recovering\n"
    "  from tool failures, or reasoning about tool output. Reserve 4 for\n"
    "  genuinely hard multi-step reasoning, and 5 for novel architectural or\n"
    "  research-level work.\n"
    "- reason (string, <=80 chars): short phrase justifying the complexity pick.\n"
    "  Do NOT echo the secret value in the reason field.\n"
    "\n"
    "Respond exactly with: {{\"has_secret\": <bool>, \"secret_values\": [...], "
    "\"complexity\": <int>, \"reason\": \"<str>\"}}\n"
    "\n"
    "---\n{text}\n---"
)


_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_BLOB_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

_TRUTHY = {True, "1", "true", "True", "TRUE", "yes", "Yes", "YES"}
_FALSY = {False, "0", "false", "False", "FALSE", "no", "No", "NO"}


def _parse_classifier_json(content: str) -> dict[str, Any]:
    """Parse the classifier's response into {has_secret, secret_values, complexity, reason}.

    Defensive: strips <think> tags and markdown fences; falls back to regex-extracting
    the first JSON object if the raw content doesn't parse. Coerces has_secret to bool
    and clamps complexity to [1, 5]. Raises ValueError if unsalvageable.
    """
    if not content:
        raise ValueError("empty classifier response")
    stripped = _THINK_RE.sub("", content).strip()
    stripped = _FENCE_RE.sub("", stripped).strip()
    parsed: Any = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        m = _JSON_BLOB_RE.search(stripped)
        if m is None:
            raise ValueError(f"no JSON object in classifier response: {content!r}") from None
        parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"classifier JSON is not an object: {parsed!r}")
    if "has_secret" not in parsed or "complexity" not in parsed:
        raise ValueError(f"classifier JSON missing required fields: {parsed!r}")

    raw_secret = parsed["has_secret"]
    if raw_secret in _TRUTHY:
        has_secret = True
    elif raw_secret in _FALSY:
        has_secret = False
    else:
        raise ValueError(f"cannot coerce has_secret={raw_secret!r}")

    try:
        complexity = int(parsed["complexity"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"cannot coerce complexity={parsed['complexity']!r}") from e
    complexity = max(1, min(5, complexity))

    reason = str(parsed.get("reason", ""))[:120]

    raw_values = parsed.get("secret_values", [])
    if not isinstance(raw_values, list):
        raw_values = []
    secret_values = [
        str(v) for v in raw_values
        if isinstance(v, (str, int, float)) and len(str(v)) >= 3
    ]

    return {
        "has_secret": has_secret,
        "secret_values": secret_values,
        "complexity": complexity,
        "reason": reason,
    }


def _map_tier(has_secret: bool, complexity: int) -> str:
    """Map classifier output to a tier. has_secret routes to classifier.secret_tier."""
    if has_secret:
        return CFG.classifier.secret_tier
    return CFG.classifier.complexity_to_tier.get(complexity, CFG.classifier.complexity_default_tier)


# --- tier hysteresis -------------------------------------------------------
#
# Rationale: large local thinking/instruct models often can't both stay
# resident in VRAM, so swapping pays a multi-second latency cost per turn.
# When the classifier wobbles between tiers in a single conversation, this
# cache holds the conversation on its sticky tier (per config sticky_pairs).

_LAST_TIER_BY_CHAT: dict[str, tuple[str, float]] = {}


def _extract_chat_id(body: dict[str, Any]) -> str | None:
    """Pull a stable session id from the latest user message via configured regex."""
    if not CFG.hysteresis.enabled:
        return None
    latest = _latest_user_text(body.get("messages", []))
    m = CFG.hysteresis.chat_id_re.search(latest)
    return m.group(1) if m else None


def _apply_hysteresis(tier: str, chat_id: str | None, now: float) -> str:
    """Apply configured sticky_pairs to keep a conversation on a sticky tier."""
    if not CFG.hysteresis.enabled or not chat_id:
        return tier
    ttl = CFG.hysteresis.ttl_s
    stale = [k for k, (_, t) in _LAST_TIER_BY_CHAT.items() if now - t > ttl * 2]
    for k in stale:
        _LAST_TIER_BY_CHAT.pop(k, None)
    prev = _LAST_TIER_BY_CHAT.get(chat_id)
    if prev is None:
        return tier
    prev_tier, prev_ts = prev
    if now - prev_ts > ttl:
        return tier
    for sp in CFG.hysteresis.sticky_pairs:
        if prev_tier == sp.from_prev and tier == sp.when_now:
            return sp.keep
    return tier


def _record_tier(chat_id: str | None, tier: str, now: float) -> None:
    """Remember the tier a chat was routed to, for later hysteresis lookups."""
    if chat_id and tier in CFG.local_tiers:
        _LAST_TIER_BY_CHAT[chat_id] = (tier, now)


# --- thread-aware routing (issue #1) ---------------------------------------
# Public-ish helpers (also unit-tested via tests/test_threading.py):
#   _sanitize_header_thread_id  : header validation/normalization
#   _extract_thread_id          : pure function, source-prefixed key
#   _claim_thread_state         : admission-time race-safe parent lookup
#   _record_thread_state        : commit thread state after routing decided
#   _apply_parent_tier_policy   : enforce inform/cap/ignore policy
#   _apply_thread_sticky        : per-thread sticky-pair downgrade
#   _build_classification_text  : build sub-call-isolated prompt input

_HEADER_BAD_CHAR_RE = re.compile(r"[^\w\-:.@$+/=]")

# Process-local state cache: thread_id -> {
#   origin_request_id, previous_request_id, previous_tier, previous_complexity, ts
# }. Bounded only by `threading.ttl_s` eviction during access.
_THREAD_STATE: dict[str, dict[str, Any]] = {}
_THREAD_STATE_LOCK = asyncio.Lock()


def _sanitize_header_thread_id(raw: str) -> str | None:
    """Return a sanitized thread-id from a raw header value, or None.

    Strips control chars, replaces non-safe chars with '_', caps at 128.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    cleaned = _HEADER_BAD_CHAR_RE.sub("_", raw)
    cleaned = cleaned[:128]
    return cleaned or None


def _last_user_message_text(messages: list[dict[str, Any]]) -> str:
    """Return the textual content of the most recent user-role message, or ''."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return _messages_text([m])
    return ""


def _trailing_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Last message in the array (any role), or None."""
    for m in reversed(messages):
        if m.get("role") in ("user", "assistant", "tool", "function", "system"):
            return m
    return messages[-1] if messages else None


def _extract_thread_id(body: dict[str, Any], header_value: str | None) -> str | None:
    """Compute the source-prefixed thread id for this request, or None.

    Priority: header → extractor regex list (against the LAST user message
    only) → fallback hash of (system prompt prefix, latest user prefix).
    """
    spec = CFG.threading
    if not spec.enabled:
        return None
    if header_value:
        cleaned = _sanitize_header_thread_id(header_value)
        if cleaned:
            return f"hdr:{cleaned}"
    messages = body.get("messages") or []
    last_user = _last_user_message_text(messages)
    for ex in spec.extractors:
        if ex.source != "last_user_text":
            continue
        m = ex.compiled.search(last_user)
        if m and m.group(1):
            return f"{ex.name}:{m.group(1)}"
    if spec.fallback_hash:
        sys_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                sys_text = _messages_text([msg])[:512]
                break
        digest = hashlib.sha256(
            (sys_text + "\x1f" + last_user[:2048]).encode("utf-8", errors="replace")
        ).hexdigest()[:24]
        return f"fallback:{digest}"
    return None


def _lookup_thread_state_in_db(
    thread_id: str, ttl: float, now: float,
) -> dict[str, Any] | None:
    """Find the most-recent same-thread row within ttl. Cold-cache fallback."""
    if not ENV.log_requests:
        return None
    try:
        _init_db()
        conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT id, origin_request_id, tier, local_complexity, ts "
                "FROM requests WHERE thread_id = ? AND ts >= ? "
                "ORDER BY ts DESC LIMIT 1",
                (thread_id, now - ttl),
            ).fetchone()
            if not row:
                return None
            return {
                "origin_request_id": row["origin_request_id"] or row["id"],
                "previous_request_id": row["id"],
                "previous_tier": row["tier"],
                "previous_complexity": row["local_complexity"],
                "ts": row["ts"],
            }
        finally:
            conn.close()
    except Exception as e:
        LOG.debug("thread-state DB lookup failed: %s", e)
        return None


async def _claim_thread_state(thread_id: str, now: float) -> dict[str, Any]:
    """Race-safe admission-time claim. Reserves a slot if cache is empty.

    Returns:
      {
        is_originating: bool,
        origin_request_id: int | None,
        previous_request_id: int | None,
        previous_tier: str | None,
        previous_complexity: int | None,
      }
    """
    spec = CFG.threading
    ttl = float(spec.ttl_s)
    async with _THREAD_STATE_LOCK:
        cached = _THREAD_STATE.get(thread_id)
        # A "reserved" slot has reserved=True and no previous_request_id —
        # this means another concurrent request is mid-flight and hasn't
        # finished its DB lookup or insert yet. We treat it as originating
        # too (no parent state to inform classification) so siblings don't
        # silently get origin_request_id=None subcalls.
        if cached and (now - cached.get("ts", 0.0)) <= ttl and not cached.get("reserved"):
            return {
                "is_originating": False,
                "origin_request_id": cached.get("origin_request_id"),
                "previous_request_id": cached.get("previous_request_id"),
                "previous_tier": cached.get("previous_tier"),
                "previous_complexity": cached.get("previous_complexity"),
            }
        # Reserve the slot so a sibling arriving microseconds later sees us.
        # Mark `reserved=True` so a sibling hitting this same window doesn't
        # treat the half-initialized slot as a real parent.
        _THREAD_STATE[thread_id] = {
            "origin_request_id": cached.get("origin_request_id") if cached else None,
            "previous_request_id": cached.get("previous_request_id") if cached else None,
            "previous_tier": None,
            "previous_complexity": None,
            "ts": now,
            "reserved": True,
        }
    # Cache miss (cold or expired). Look in DB (no lock — async friendly).
    db_state = await asyncio.to_thread(_lookup_thread_state_in_db, thread_id, ttl, now)
    if db_state and (now - db_state["ts"]) <= ttl:
        async with _THREAD_STATE_LOCK:
            slot = _THREAD_STATE.get(thread_id)
            if slot is not None:
                slot.update({
                    "origin_request_id": db_state["origin_request_id"],
                    "previous_request_id": db_state["previous_request_id"],
                    "previous_tier": db_state["previous_tier"],
                    "previous_complexity": db_state["previous_complexity"],
                    "reserved": False,
                })
        return {
            "is_originating": False,
            "origin_request_id": db_state["origin_request_id"],
            "previous_request_id": db_state["previous_request_id"],
            "previous_tier": db_state["previous_tier"],
            "previous_complexity": db_state["previous_complexity"],
        }
    return {
        "is_originating": True,
        "origin_request_id": None,
        "previous_request_id": None,
        "previous_tier": None,
        "previous_complexity": None,
    }


def _record_thread_state(
    thread_id: str | None,
    origin_request_id: int | None,
    previous_request_id: int | None,
    tier: str | None,
    complexity: int | None,
    now: float,
) -> None:
    """Persist routing outcome for this thread so the next request inherits it.

    NOTE: Caller MUST hold no other locks. This grabs the asyncio lock via
    a sync shortcut: we only mutate primitive dict fields, but to keep the
    cache consistent with `_claim_thread_state` (which reads under the
    lock), we acquire it here too. Because this function may be called
    from sync contexts (post-INSERT finalize) we use the lock's underlying
    primitive directly via a try/except.
    """
    if not thread_id:
        return
    # asyncio.Lock isn't reentrant and isn't sync-callable. The mutations
    # are atomic for primitive dict[str, Any] writes in CPython (GIL), so
    # the practical race is between a writer here and a reader in
    # `_claim_thread_state` seeing a half-updated slot. We avoid that by
    # building the new slot dict in one go and assigning by reference.
    new_slot = {
        "origin_request_id": origin_request_id,
        "previous_request_id": previous_request_id,
        "previous_tier": tier,
        "previous_complexity": complexity,
        "ts": now,
        "reserved": False,
    }
    _THREAD_STATE[thread_id] = new_slot


def _tier_rank(tier: str | None) -> int:
    """Insertion-order rank from CFG.tiers. Unknown tiers get -1."""
    if tier is None:
        return -1
    for i, name in enumerate(CFG.tiers.keys()):
        if name == tier:
            return i
    return -1


def _apply_parent_tier_policy(child_tier: str, parent_tier: str | None) -> str:
    """Enforce parent_tier_policy from CFG.threading.

    - inform / ignore: child unchanged.
    - cap: child rank cannot exceed parent rank; if it would, drop to parent.
    """
    spec = CFG.threading
    if not spec.enabled or spec.parent_tier_policy in ("inform", "ignore"):
        return child_tier
    if (
        spec.parent_tier_policy == "cap"
        and parent_tier
        and _tier_rank(child_tier) > _tier_rank(parent_tier)
    ):
        return parent_tier
    return child_tier


def _apply_thread_sticky(
    tier: str,
    parent_state: dict[str, Any] | None,
    now: float,  # noqa: ARG001 — kept for symmetry with _apply_hysteresis
) -> str:
    """Per-thread sticky_pairs adoption (mirrors `_apply_hysteresis`)."""
    spec = CFG.threading
    if not spec.enabled or not spec.adopt_sticky_pairs or not parent_state:
        return tier
    prev = parent_state.get("previous_tier")
    if not prev:
        return tier
    for sp in CFG.hysteresis.sticky_pairs:
        if prev == sp.from_prev and tier == sp.when_now:
            return sp.keep
    return tier


def _build_classification_text(body: dict[str, Any], is_subcall: bool) -> str:
    """Return the text fed into the local classifier.

    For originating requests this is just the latest user message (cleaned).
    For sub-calls, when `classify_subcall_isolated` is True, the text is
    composed of the original user ask + the trailing message (role-labeled),
    so the classifier scores the *current* sub-step rather than re-scoring
    the original ask via a long history.
    """
    spec = CFG.threading
    messages = body.get("messages") or []
    user_text = _clean_user_text(_last_user_message_text(messages))
    if not is_subcall or not spec.classify_subcall_isolated:
        return user_text
    trailing = _trailing_message(messages)
    if trailing is None or trailing.get("role") == "user":
        return user_text
    trailing_text = _messages_text([trailing])
    return (
        f"Original user ask: {user_text}\n\n"
        f"Current trailing message (role={trailing.get('role')}): {trailing_text}"
    )


async def local_classify(
    body: dict[str, Any],
    classification_text: str | None = None,
) -> tuple[bool, int, dict[str, Any]]:
    """Ask the local model to classify the request. Returns (has_secret, complexity, details)."""
    if classification_text is None:
        classification_text = _clean_user_text(_latest_user_text(body.get("messages", [])))
    text = classification_text[:4000]
    prompt = _CLASSIFIER_PROMPT_TEMPLATE.format(text=text)
    details: dict[str, Any] = {"prompt": prompt, "response": None, "error": None}
    try:
        async with httpx.AsyncClient(timeout=CFG.classifier.local_timeout_s) as c:
            r = await c.post(
                f"{ENV.litellm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {ENV.litellm_api_key}"},
                json={
                    "model": CFG.classifier.local_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "temperature": 0,
                    "stream": False,
                },
            )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = _parse_classifier_json(content)
        secret_values = parsed.get("secret_values", [])
        if parsed["has_secret"] and secret_values:
            details["prompt"] = _scrub_with_values(prompt, secret_values)
            details["response"] = _scrub_with_values(content, secret_values)
        else:
            details["response"] = content
        details["secret_values"] = secret_values
        return parsed["has_secret"], parsed["complexity"], details
    except Exception as e:
        details["error"] = str(e)
        raise


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI-style message content (str or list-of-parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


_SUMMARY_MAX_CHARS = 240


def _shorten(text: str, limit: int = _SUMMARY_MAX_CHARS) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _extract_subcall_summary(
    messages: list[dict[str, Any]],
    secret_keyword: str | None,
    llm_secret_values: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build a compact 'what triggered this LLM call' summary from the
    most recent message in the conversation history.

    Returns a dict with keys: kind, tool_name, agent_says, summary.
    None when there's nothing meaningful to display (empty messages, or
    secret_keyword=='llm_classifier' without scrub values — same gating
    as _serialize_full_messages).
    """
    if not messages:
        return None
    if secret_keyword == "llm_classifier" and not llm_secret_values:
        return None

    regex_scrub = secret_keyword is not None and secret_keyword != "llm_classifier"
    has_values = bool(llm_secret_values)

    def scrub(s: str) -> str:
        if not s:
            return s
        if regex_scrub:
            s = _scrub_secrets(s)
        if has_values:
            s = _scrub_with_values(s, llm_secret_values or [])
        return s

    last = messages[-1]
    role = last.get("role")
    out: dict[str, Any] = {
        "kind": role or "unknown",
        "tool_name": None,
        "agent_says": None,
        "summary": "",
    }

    if role == "tool":
        tool_call_id = last.get("tool_call_id")
        tool_name = last.get("name")
        agent_says = None
        if not tool_name or tool_call_id:
            for m in reversed(messages[:-1]):
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                matched = None
                if tool_call_id:
                    for tc in tcs:
                        if tc.get("id") == tool_call_id:
                            matched = tc
                            break
                if matched is None and tcs and not tool_call_id:
                    matched = tcs[0]
                if matched and not tool_name:
                    fn = matched.get("function") or {}
                    tool_name = fn.get("name")
                if matched is not None:
                    agent_says = _content_to_text(m.get("content")) or None
                    break
        out["kind"] = "tool_result"
        out["tool_name"] = tool_name or None
        if agent_says:
            out["agent_says"] = _shorten(scrub(agent_says), 160)
        out["summary"] = _shorten(scrub(_content_to_text(last.get("content"))))
    elif role == "user":
        out["kind"] = "user_continuation"
        out["summary"] = _shorten(scrub(_content_to_text(last.get("content"))))
    elif role == "assistant":
        text = _content_to_text(last.get("content"))
        tcs = last.get("tool_calls") or []
        if tcs and not text:
            fn = (tcs[0].get("function") or {})
            out["kind"] = "tool_call"
            out["tool_name"] = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                out["summary"] = _shorten(scrub(args), 160)
        else:
            out["kind"] = "assistant_text"
            out["summary"] = _shorten(scrub(text))
    else:
        out["summary"] = _shorten(scrub(_content_to_text(last.get("content"))))

    if not out["summary"] and not out["tool_name"] and not out["agent_says"]:
        return None
    return out


def _serialize_full_messages(
    messages: list[dict[str, Any]],
    secret_keyword: str | None,
    llm_secret_values: list[str] | None = None,
) -> str | None:
    """Serialize the full message array for the log, scrubbing secrets."""
    regex_scrub = secret_keyword is not None and secret_keyword != "llm_classifier"
    has_values = bool(llm_secret_values)
    if secret_keyword == "llm_classifier" and not has_values:
        return None

    def scrub(s: str) -> str:
        if regex_scrub:
            s = _scrub_secrets(s)
        if has_values:
            s = _scrub_with_values(s, llm_secret_values or [])
        return s

    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            new_content: Any = scrub(content) if (regex_scrub or has_values) else content
        elif isinstance(content, list):
            new_content = []
            for p in content:
                if isinstance(p, dict) and "text" in p and (regex_scrub or has_values):
                    new_content.append({**p, "text": scrub(p["text"])})
                else:
                    new_content.append(p)
        else:
            new_content = content
        out.append({"role": m.get("role"), "content": new_content})
    text = json.dumps(out, ensure_ascii=False)
    if len(text) > CFG.limits.messages_full_max_chars:
        text = text[:CFG.limits.messages_full_max_chars] + "…[truncated]"
    return text


def _redact_for_log(
    full_text: str,
    secret_keyword: str | None,
    llm_secret_values: list[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Return (messages_preview, keywords_list) for logging."""
    pc = CFG.limits.messages_preview_chars
    if secret_keyword is None:
        return full_text[:pc], extract_keywords(full_text)
    if secret_keyword == "llm_classifier":
        if llm_secret_values:
            scrubbed = _scrub_with_values(full_text, llm_secret_values)
            return scrubbed[:pc], extract_keywords(scrubbed)
        return None, []
    scrubbed = _scrub_secrets(full_text)
    if llm_secret_values:
        scrubbed = _scrub_with_values(scrubbed, llm_secret_values)
    return scrubbed[:pc], extract_keywords(scrubbed)


# --- storage ---------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")
_db_ready = False

_REQUESTS_ADDED_COLUMNS: list[tuple[str, str]] = [
    ("step_keyword", "TEXT"),
    ("secret_keyword", "TEXT"),
    ("tiebreaker_prompt", "TEXT"),
    ("tiebreaker_response", "TEXT"),
    ("tiebreaker_digit", "TEXT"),
    ("tiebreaker_error", "TEXT"),
    ("classifier_source", "TEXT"),
    ("classifier_model", "TEXT"),
    ("local_classifier_prompt", "TEXT"),
    ("local_classifier_response", "TEXT"),
    ("local_classifier_error", "TEXT"),
    ("local_complexity", "INTEGER"),
    ("local_secret", "INTEGER"),
    ("messages_full", "TEXT"),
    # Threading (issue #1). All NULL when threading.enabled=false.
    # `thread_id`: opaque source-prefixed key, e.g. `openclaw_message_id:$rus-...`
    # or `fallback:<sha256>`. Indexed for thread-grouped queries.
    ("thread_id", "TEXT"),
    # `origin_request_id`: id of the originating row in the same thread.
    # Equals the row's own id for originating rows. NULL when thread_id is NULL.
    ("origin_request_id", "INTEGER"),
    # `thread_role`: 'originating' | 'subcall'. NULL when thread_id is NULL.
    ("thread_role", "TEXT"),
    # Snapshot of the parent's local_complexity at routing time. Logged
    # under all parent_tier_policy values (informational).
    ("parent_complexity", "INTEGER"),
    # Snapshot of the parent's tier at routing time.
    ("parent_tier", "TEXT"),
    # Per-row 'what triggered this LLM call' summary (JSON object with
    # keys: kind, tool_name, agent_says, summary). Populated for sub-calls
    # so the UI can render a trajectory tree without repeating the user
    # prompt. NULL for originating rows and pre-threading rows.
    ("subcall_summary", "TEXT"),
]


def _migrate_requests_schema(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
    for name, sql_type in _REQUESTS_ADDED_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {name} {sql_type}")


def _init_db() -> None:
    global _db_ready
    if _db_ready:
        return
    Path(ENV.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ENV.db_path, check_same_thread=False)
    try:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                duration_ms INTEGER,
                requested_model TEXT,
                tier TEXT,
                reason TEXT,
                target_model TEXT,
                fallback_applied INTEGER DEFAULT 0,
                is_stream INTEGER DEFAULT 0,
                upstream_status INTEGER,
                approx_tokens INTEGER,
                has_tools INTEGER DEFAULT 0,
                has_code INTEGER DEFAULT 0,
                step_keyword TEXT,
                secret_keyword TEXT,
                tiebreaker_prompt TEXT,
                tiebreaker_response TEXT,
                tiebreaker_digit TEXT,
                tiebreaker_error TEXT,
                classifier_source TEXT,
                classifier_model TEXT,
                local_classifier_prompt TEXT,
                local_classifier_response TEXT,
                local_classifier_error TEXT,
                local_complexity INTEGER,
                local_secret INTEGER,
                keywords TEXT,
                messages_preview TEXT,
                messages_full TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_requests_ts   ON requests(ts DESC);
            CREATE INDEX IF NOT EXISTS idx_requests_tier ON requests(tier);

            CREATE VIRTUAL TABLE IF NOT EXISTS requests_fts USING fts5(
                keywords, messages_preview,
                content='requests', content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS requests_ai AFTER INSERT ON requests BEGIN
                INSERT INTO requests_fts(rowid, keywords, messages_preview)
                VALUES (new.id, coalesce(new.keywords, ''), coalesce(new.messages_preview, ''));
            END;

            CREATE TRIGGER IF NOT EXISTS requests_ad AFTER DELETE ON requests BEGIN
                INSERT INTO requests_fts(requests_fts, rowid, keywords, messages_preview)
                VALUES ('delete', old.id, coalesce(old.keywords, ''), coalesce(old.messages_preview, ''));
            END;
        """)
        _migrate_requests_schema(conn)
        # Thread index — created post-migration so the column exists. IF NOT
        # EXISTS makes this no-op on subsequent boots.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_thread_ts "
                "ON requests(thread_id, ts DESC)"
            )
        conn.commit()
    finally:
        conn.close()
    _db_ready = True


def extract_keywords(text: str, max_n: int = 30) -> list[str]:
    freq: dict[str, int] = {}
    for w in _WORD_RE.findall(text.lower()):
        if w in CFG.heuristic.stopwords:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:max_n]]


def log_request(row: dict[str, Any]) -> int | None:
    """Insert a request row and return its new id, or None if logging is off."""
    if not ENV.log_requests:
        return None
    try:
        _init_db()
        conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=5.0)
        try:
            keys = list(row.keys())
            cur = conn.execute(
                f"INSERT INTO requests ({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                [row[k] for k in keys],
            )
            conn.commit()
            return int(cur.lastrowid) if cur.lastrowid is not None else None
        finally:
            conn.close()
    except Exception as e:
        LOG.warning("failed to log request: %s", e)
        return None


def delete_request(req_id: int) -> bool:
    _init_db()
    conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=5.0)
    try:
        cur = conn.execute("DELETE FROM requests WHERE id = ?", (req_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _fts_match(q: str) -> str:
    tokens = _FTS_TOKEN_RE.findall(q)
    return " ".join(f'"{t}"*' for t in tokens)


def search_requests(
    q: str = "",
    tier: list[str] | str | None = None,
    reason: list[str] | str | None = None,
    classifier_source: list[str] | str | None = None,
    fallback: int | None = None,
    has_secret: int | None = None,
    is_stream: int | None = None,
    min_duration_ms: int | None = None,
    since: float | None = None,
    until: float | None = None,
    since_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    _init_db()
    conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        where: list[str] = []
        params: list[Any] = []
        joins = ""

        match = _fts_match(q) if q else ""
        if match:
            joins = " JOIN requests_fts ON requests_fts.rowid = requests.id "
            where.append("requests_fts MATCH ?")
            params.append(match)

        def _in_clause(column: str, values: list[str] | str | None) -> None:
            if values is None:
                return
            vs = [values] if isinstance(values, str) else [v for v in values if v]
            if not vs:
                return
            where.append(f"requests.{column} IN ({','.join('?' * len(vs))})")
            params.extend(vs)

        _in_clause("tier", tier)
        _in_clause("reason", reason)
        _in_clause("classifier_source", classifier_source)
        if fallback in (0, 1):
            where.append("requests.fallback_applied = ?")
            params.append(fallback)
        if has_secret == 1:
            where.append("requests.secret_keyword IS NOT NULL")
        elif has_secret == 0:
            where.append("requests.secret_keyword IS NULL")
        if is_stream in (0, 1):
            where.append("requests.is_stream = ?")
            params.append(is_stream)
        if min_duration_ms is not None:
            where.append("requests.duration_ms >= ?")
            params.append(min_duration_ms)
        if since is not None:
            where.append("requests.ts >= ?")
            params.append(since)
        if until is not None:
            where.append("requests.ts <= ?")
            params.append(until)
        if since_id is not None:
            where.append("requests.id > ?")
            params.append(since_id)
        where_sql = f" WHERE {' AND '.join(where)}" if where else ""

        total = int(conn.execute(
            f"SELECT COUNT(*) AS n FROM requests {joins}{where_sql}", params,
        ).fetchone()["n"])

        rows = conn.execute(
            f"""SELECT requests.id, requests.ts, requests.duration_ms,
                       requests.requested_model, requests.tier, requests.reason,
                       requests.target_model, requests.fallback_applied,
                       requests.is_stream, requests.upstream_status,
                       requests.approx_tokens, requests.has_tools, requests.has_code,
                       requests.step_keyword, requests.secret_keyword,
                       requests.tiebreaker_digit, requests.classifier_source,
                       requests.local_complexity, requests.local_secret,
                       requests.keywords,
                       requests.thread_id, requests.origin_request_id,
                       requests.thread_role, requests.parent_complexity,
                       requests.parent_tier,
                       requests.subcall_summary,
                       substr(coalesce(requests.messages_preview, ''), 1, 240) AS snippet
                FROM requests {joins}{where_sql}
                ORDER BY requests.ts DESC
                LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [dict(r) for r in rows],
        }
    finally:
        conn.close()


def get_request(req_id: int) -> dict[str, Any] | None:
    _init_db()
    conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM requests WHERE id = ?", (req_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def stats() -> dict[str, Any]:
    _init_db()
    conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        total = int(conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"])
        by_tier = [
            {"tier": r["tier"], "count": r["n"]}
            for r in conn.execute(
                "SELECT tier, COUNT(*) AS n FROM requests GROUP BY tier ORDER BY n DESC"
            ).fetchall()
        ]
        by_reason = [
            {"reason": r["reason"], "count": r["n"]}
            for r in conn.execute(
                "SELECT reason, COUNT(*) AS n FROM requests GROUP BY reason ORDER BY n DESC"
            ).fetchall()
        ]
        return {"total": total, "by_tier": by_tier, "by_reason": by_reason}
    finally:
        conn.close()


# --- live event hub --------------------------------------------------------

_HUB_QUEUE_MAX = 100
_hub_clients: set[asyncio.Queue] = set()


async def _hub_publish(event: dict[str, Any]) -> None:
    for q in list(_hub_clients):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            LOG.debug("hub: dropping event for slow client (queue full)")


def _project_list_item(row_id: int, row: dict[str, Any]) -> dict[str, Any]:
    preview = row.get("messages_preview") or ""
    return {
        "id": row_id,
        "ts": row.get("ts"),
        "duration_ms": row.get("duration_ms"),
        "requested_model": row.get("requested_model"),
        "tier": row.get("tier"),
        "reason": row.get("reason"),
        "target_model": row.get("target_model"),
        "fallback_applied": row.get("fallback_applied"),
        "is_stream": row.get("is_stream"),
        "upstream_status": row.get("upstream_status"),
        "approx_tokens": row.get("approx_tokens"),
        "has_tools": row.get("has_tools"),
        "has_code": row.get("has_code"),
        "step_keyword": row.get("step_keyword"),
        "secret_keyword": row.get("secret_keyword"),
        "tiebreaker_digit": row.get("tiebreaker_digit"),
        "classifier_source": row.get("classifier_source"),
        "local_complexity": row.get("local_complexity"),
        "local_secret": row.get("local_secret"),
        "keywords": row.get("keywords"),
        "snippet": preview[:240] if preview else "",
        "thread_id": row.get("thread_id"),
        "origin_request_id": row.get("origin_request_id"),
        "thread_role": row.get("thread_role"),
        "parent_complexity": row.get("parent_complexity"),
        "parent_tier": row.get("parent_tier"),
        "subcall_summary": row.get("subcall_summary"),
    }


# --- app -------------------------------------------------------------------



@asynccontextmanager
async def _lifespan(app_):  # noqa: ARG001
    try:
        _init_db()
    except Exception as e:
        LOG.warning("db init failed at startup: %s", e)
    yield


app = FastAPI(title="llmrouter", lifespan=_lifespan)


def _check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if authorization[7:].strip() != ENV.router_api_key:
        raise HTTPException(401, "invalid api key")


@app.get("/health")
async def health() -> dict[str, Any]:
    ok = await ollama_healthy()
    return {"status": "ok", "ollama_healthy": ok}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_auth(authorization)
    now = int(time.time())
    ids = ["auto", *CFG.tier_to_model.keys()]
    return {
        "object": "list",
        "data": [
            {"id": i, "object": "model", "created": now, "owned_by": "llmrouter"}
            for i in ids
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
):
    _check_auth(authorization)
    body = await request.json()
    requested = body.get("model", "auto")
    t0 = time.time()

    tiebreaker: dict[str, Any] | None = None
    local_details: dict[str, Any] | None = None
    classifier_source: str | None = None
    classifier_model: str | None = None
    local_complexity: int | None = None
    local_secret: int | None = None

    heuristic, signals = heuristic_tier(body)

    # Threading: extract thread_id and claim parent state at admission time
    # (before any I/O so siblings race-safely see each other).
    thread_spec = CFG.threading
    thread_header_name = (thread_spec.header_name or "").lower()
    thread_header_value = request.headers.get(thread_header_name) if thread_header_name else None
    thread_id = _extract_thread_id(body, thread_header_value)
    parent_state: dict[str, Any] | None = None
    is_subcall = False
    if thread_id:
        parent_state = await _claim_thread_state(thread_id, t0)
        is_subcall = not parent_state["is_originating"]

    # Classification text: when threading is enabled and classify_subcall_isolated
    # is on for sub-calls, isolate the trailing message from the original ask
    # so the classifier scores the actual sub-step.
    classify_text: str | None = None
    if thread_spec.enabled:
        classify_text = _build_classification_text(body, is_subcall=is_subcall)

    parent_tier_for_log: str | None = (parent_state or {}).get("previous_tier") if is_subcall else None
    parent_complexity_for_log: int | None = (
        (parent_state or {}).get("previous_complexity") if is_subcall else None
    )

    if requested == "auto":
        used_local = False
        if CFG.classifier.mode == "local" and await ollama_healthy():
            try:
                l_secret, l_complex, local_details = await asyncio.wait_for(
                    local_classify(body, classification_text=classify_text),
                    timeout=CFG.classifier.local_timeout_s + 0.5,
                )
                local_secret = 1 if l_secret else 0
                local_complexity = int(l_complex)
                regex_hit = signals.get("secret_keyword")
                union_secret = bool(l_secret) or bool(regex_hit)
                if union_secret and not regex_hit:
                    signals["secret_keyword"] = "llm_classifier"
                raw_tier = _map_tier(union_secret, local_complexity)
                tier = _apply_hysteresis(raw_tier, _extract_chat_id(body), time.time())
                reason = "local-classifier+hysteresis" if tier != raw_tier else "local-classifier"
                classifier_source = "local_model"
                classifier_model = CFG.classifier.local_model
                used_local = True
            except Exception as e:
                err_str = str(e) or type(e).__name__
                LOG.warning("local classifier failed, falling back: %s", err_str)
                if local_details is None:
                    local_details = {"prompt": None, "response": None, "error": err_str}
                else:
                    local_details["error"] = err_str
                classifier_source = "fallback"

        if not used_local:
            if heuristic is not None:
                tier, reason = heuristic, "heuristic"
                classifier_source = classifier_source or "heuristic"
            elif CFG.classifier.mode in ("hybrid", "haiku", "local"):
                tier, reason, tiebreaker = await haiku_tiebreaker(
                    body, classification_text=classify_text,
                )
                classifier_source = classifier_source or "haiku_tiebreaker"
            else:
                tier, reason = CFG.classifier.heuristic_default_tier, "heuristic-default"
                classifier_source = classifier_source or "heuristic"
    elif requested in CFG.tier_to_model:
        tier, reason = requested, "explicit"
        classifier_source = "explicit"
    else:
        tier, reason = None, "passthrough"
        classifier_source = "passthrough"

    # Threading policy: parent_tier_policy + per-thread sticky pairs.
    # Both no-op when threading.enabled=false.
    if thread_id and is_subcall and tier:
        capped = _apply_parent_tier_policy(tier, parent_tier_for_log)
        if capped != tier:
            reason = (reason or "") + "+parent-cap"
            tier = capped
        sticky = _apply_thread_sticky(tier, parent_state, time.time())
        if sticky != tier:
            reason = (reason or "") + "+thread-sticky"
            tier = sticky

    if classifier_source == "local_model":
        _record_tier(_extract_chat_id(body), tier, time.time())

    fallback_applied = False
    if tier in CFG.local_tiers and not await ollama_healthy():
        tier = CFG.fallback.cloud_tier
        fallback_applied = True

    target_model = CFG.tier_to_model[tier] if tier else requested
    forward_body = {**body, "model": target_model}
    is_stream = bool(body.get("stream"))

    LOG.info(
        "route tier=%s reason=%s requested=%s target=%s stream=%s fallback=%s thread=%s",
        tier, reason, requested, target_model, is_stream, fallback_applied, thread_id,
    )

    response_headers = {
        "x-llmrouter-tier": str(tier),
        "x-llmrouter-reason": reason,
        "x-llmrouter-target": target_model,
    }
    if fallback_applied:
        response_headers["x-llmrouter-fallback"] = "ollama-unreachable"
    if thread_id:
        response_headers["x-llmrouter-thread-id"] = thread_id
        response_headers["x-llmrouter-thread-role"] = (
            "subcall" if is_subcall else "originating"
        )

    llm_secret_values = (local_details or {}).get("secret_values") or None
    log_text = _clean_user_text(_latest_user_text(body.get("messages", [])))
    messages_preview, keywords_list = _redact_for_log(
        log_text, signals.get("secret_keyword"), llm_secret_values,
    )
    messages_full = _serialize_full_messages(
        body.get("messages", []), signals.get("secret_keyword"), llm_secret_values,
    )

    client = httpx.AsyncClient(timeout=None)
    upstream_req = client.build_request(
        "POST",
        f"{ENV.litellm_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {ENV.litellm_api_key}",
            "Content-Type": "application/json",
        },
        json=forward_body,
    )
    upstream = await client.send(upstream_req, stream=is_stream)

    thread_role: str | None = None
    if thread_id:
        thread_role = "subcall" if is_subcall else "originating"
    subcall_summary_obj: dict[str, Any] | None = None
    if thread_id and is_subcall:
        subcall_summary_obj = _extract_subcall_summary(
            body.get("messages", []),
            signals.get("secret_keyword"),
            llm_secret_values,
        )
    base_row: dict[str, Any] = {
        "ts": t0,
        "duration_ms": None,
        "requested_model": requested,
        "tier": tier,
        "reason": reason,
        "target_model": target_model,
        "fallback_applied": 1 if fallback_applied else 0,
        "is_stream": 1 if is_stream else 0,
        "upstream_status": upstream.status_code,
        "approx_tokens": signals.get("tokens"),
        "has_tools": 1 if signals.get("has_tools") else 0,
        "has_code": 1 if signals.get("has_code") else 0,
        "step_keyword": signals.get("step_keyword"),
        "secret_keyword": signals.get("secret_keyword"),
        "tiebreaker_prompt": (tiebreaker or {}).get("prompt"),
        "tiebreaker_response": (tiebreaker or {}).get("response"),
        "tiebreaker_digit": (tiebreaker or {}).get("digit"),
        "tiebreaker_error": (tiebreaker or {}).get("error"),
        "classifier_source": classifier_source,
        "classifier_model": classifier_model,
        "local_classifier_prompt": (local_details or {}).get("prompt"),
        "local_classifier_response": (local_details or {}).get("response"),
        "local_classifier_error": (local_details or {}).get("error"),
        "local_complexity": local_complexity,
        "local_secret": local_secret,
        "keywords": " ".join(keywords_list) if keywords_list else None,
        "messages_preview": messages_preview,
        "messages_full": messages_full,
        "thread_id": thread_id,
        "origin_request_id": (parent_state or {}).get("origin_request_id") if is_subcall else None,
        "thread_role": thread_role,
        "parent_complexity": parent_complexity_for_log,
        "parent_tier": parent_tier_for_log,
        "subcall_summary": json.dumps(subcall_summary_obj, ensure_ascii=False)
        if subcall_summary_obj else None,
    }

    async def _finalize_thread_state(inserted_id: int | None) -> None:
        """Backfill origin_request_id on originating rows (it == self.id) and
        update the in-process state cache so siblings see this row."""
        if not thread_id or inserted_id is None:
            return
        origin_id = base_row.get("origin_request_id") or inserted_id
        if not is_subcall:
            # Originating: backfill self-reference. Run in threadpool so we
            # don't block the event loop on sqlite I/O under load.
            def _backfill_origin_request_id(row_id: int) -> None:
                conn = sqlite3.connect(ENV.db_path, check_same_thread=False, timeout=2.0)
                try:
                    conn.execute(
                        "UPDATE requests SET origin_request_id = ? WHERE id = ?",
                        (row_id, row_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

            try:
                await asyncio.to_thread(_backfill_origin_request_id, inserted_id)
            except Exception as e:
                LOG.debug("origin backfill failed: %s", e)
        _record_thread_state(
            thread_id,
            origin_request_id=origin_id,
            previous_request_id=inserted_id,
            tier=tier,
            complexity=local_complexity,
            now=time.time(),
        )

    if not is_stream:
        data = await upstream.aread()
        await client.aclose()
        base_row["duration_ms"] = int((time.time() - t0) * 1000)
        inserted_id = await asyncio.to_thread(log_request, base_row)
        if inserted_id is not None:
            if not is_subcall and thread_id:
                base_row["origin_request_id"] = inserted_id
            await _finalize_thread_state(inserted_id)
            await _hub_publish({
                "type": "insert",
                "id": inserted_id,
                "item": _project_list_item(inserted_id, base_row),
            })
        try:
            content = json.loads(data) if data else {}
        except json.JSONDecodeError:
            content = {"raw": data.decode("utf-8", errors="replace")}
        return JSONResponse(
            status_code=upstream.status_code,
            content=content,
            headers=response_headers,
        )

    async def iter_chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            base_row["duration_ms"] = int((time.time() - t0) * 1000)
            inserted_id = await asyncio.to_thread(log_request, base_row)
            if inserted_id is not None:
                if not is_subcall and thread_id:
                    base_row["origin_request_id"] = inserted_id
                await _finalize_thread_state(inserted_id)
                await _hub_publish({
                    "type": "insert",
                    "id": inserted_id,
                    "item": _project_list_item(inserted_id, base_row),
                })

    return StreamingResponse(
        iter_chunks(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=response_headers,
    )


# --- UI --------------------------------------------------------------------


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
async def ui_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/ui/api/stats")
async def ui_stats(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_auth(authorization)
    return stats()


@app.get("/ui/api/requests")
async def ui_list_requests(
    q: str = Query(default=""),
    tier: list[str] | None = Query(default=None),
    reason: list[str] | None = Query(default=None),
    classifier_source: list[str] | None = Query(default=None),
    fallback: int | None = Query(default=None, ge=0, le=1),
    has_secret: int | None = Query(default=None, ge=0, le=1),
    is_stream: int | None = Query(default=None, ge=0, le=1),
    min_duration_ms: int | None = Query(default=None, ge=0),
    since: float | None = Query(default=None),
    until: float | None = Query(default=None),
    since_id: int | None = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    return search_requests(
        q=q,
        tier=tier,
        reason=reason,
        classifier_source=classifier_source,
        fallback=fallback,
        has_secret=has_secret,
        is_stream=is_stream,
        min_duration_ms=min_duration_ms,
        since=since,
        until=until,
        since_id=since_id,
        limit=limit,
        offset=offset,
    )


@app.get("/ui/api/requests/{req_id}")
async def ui_get_request(
    req_id: int,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    row = get_request(req_id)
    if not row:
        raise HTTPException(404, "not found")
    return row


@app.delete("/ui/api/requests/{req_id}", status_code=204)
async def ui_delete_request(
    req_id: int,
    authorization: str | None = Header(default=None),
) -> Response:
    _check_auth(authorization)
    deleted = await asyncio.to_thread(delete_request, req_id)
    if not deleted:
        raise HTTPException(404, "not found")
    await _hub_publish({"type": "delete", "id": req_id})
    return Response(status_code=204)


@app.websocket("/ui/ws")
async def ui_ws(ws: WebSocket, token: str = Query(default="")) -> None:
    if not secrets.compare_digest(token, ENV.router_api_key):
        await ws.close(code=4401)
        return
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=_HUB_QUEUE_MAX)
    _hub_clients.add(queue)
    disconnect = asyncio.Event()

    async def reader() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "ping":
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait({"type": "pong"})
        except WebSocketDisconnect:
            pass
        except Exception as e:
            LOG.debug("ws reader error: %s", e)
        finally:
            disconnect.set()

    reader_task = asyncio.create_task(reader())
    try:
        await ws.send_json({"type": "hello"})
        while not disconnect.is_set():
            get_task = asyncio.create_task(queue.get())
            dc_task = asyncio.create_task(disconnect.wait())
            done, pending = await asyncio.wait(
                {get_task, dc_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if disconnect.is_set():
                break
            event = get_task.result()
            await ws.send_json(event)
    except Exception as e:
        LOG.debug("ws sender error: %s", e)
    finally:
        reader_task.cancel()
        _hub_clients.discard(queue)
        with contextlib.suppress(Exception):
            await ws.close()
