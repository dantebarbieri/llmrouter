"""Unit tests for thread-aware routing helpers in app.py.

No network, no real DB. We monkeypatch CFG to flip threading on per-test
where needed. Mirrors the pure-function style of test_classifier.py.
"""
from __future__ import annotations

from typing import Any

import pytest

from llmrouter import app
from llmrouter.config import ThreadExtractorSpec, ThreadingSpec

# --- helpers ---------------------------------------------------------------


def _enable_threading(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> ThreadingSpec:
    """Replace CFG with a copy whose threading section is enabled."""
    extractors = (
        ThreadExtractorSpec.model_validate({
            "name": "openclaw_message_id",
            "source": "last_user_text",
            "pattern": r'"message_id"\s*:\s*"([^"]+)"',
        }),
        ThreadExtractorSpec.model_validate({
            "name": "cron_task_id",
            "source": "last_user_text",
            "pattern": r"\[cron:([a-fA-F0-9-]{8,})",
        }),
    )
    spec = ThreadingSpec.model_validate({
        "enabled": True,
        "extractors": list(extractors),
        **overrides,
    })
    new_cfg = app.CFG.model_copy(update={"threading": spec})
    monkeypatch.setattr(app, "CFG", new_cfg)
    return spec


def _user(msg_id: str | None, text: str) -> dict[str, Any]:
    blob = ""
    if msg_id is not None:
        blob = (
            'Conversation info (untrusted metadata):\n'
            '```json\n'
            '{\n'
            f'  "message_id": "{msg_id}",\n'
            '  "chat_id": "room:!abc:host"\n'
            '}\n'
            '```\n\n'
        )
    return {"role": "user", "content": blob + text}


def _system(text: str) -> dict[str, Any]:
    return {"role": "system", "content": text}


def _assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _tool(text: str) -> dict[str, Any]:
    return {"role": "tool", "content": text}


# --- _sanitize_header_thread_id -------------------------------------------


def test_sanitize_header_strips_control_chars():
    out = app._sanitize_header_thread_id("hello\x00world\n")
    assert out is not None
    assert "\x00" not in out
    assert "\n" not in out


def test_sanitize_header_caps_length():
    raw = "x" * 500
    out = app._sanitize_header_thread_id(raw)
    assert out is not None
    assert len(out) <= 128


def test_sanitize_header_empty_returns_none():
    assert app._sanitize_header_thread_id("") is None
    assert app._sanitize_header_thread_id("   ") is None


# --- _extract_thread_id ----------------------------------------------------


def test_extract_disabled_returns_none(monkeypatch: pytest.MonkeyPatch):
    spec = ThreadingSpec.model_validate({"enabled": False})
    new_cfg = app.CFG.model_copy(update={"threading": spec})
    monkeypatch.setattr(app, "CFG", new_cfg)
    body = {"messages": [_user("X", "hi")]}
    assert app._extract_thread_id(body, header_value="ignored") is None


def test_extract_header_takes_priority(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch)
    body = {"messages": [_user("X", "hi")]}
    out = app._extract_thread_id(body, header_value="my-thread-id")
    assert out == "hdr:my-thread-id"


def test_extract_openclaw_message_id(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch)
    body = {"messages": [_user("ABC123", "hello")]}
    out = app._extract_thread_id(body, header_value=None)
    assert out == "openclaw_message_id:ABC123"


def test_extract_picks_LAST_user_message_id(monkeypatch: pytest.MonkeyPatch):
    """Critical: with multiple user turns, only the LAST one's id matters."""
    _enable_threading(monkeypatch)
    body = {
        "messages": [
            _system("you are a bot"),
            _user("OLD", "first ask"),
            _assistant("ok"),
            _tool("tool result"),
            _user("NEW", "follow-up ask"),
        ],
    }
    out = app._extract_thread_id(body, header_value=None)
    assert out == "openclaw_message_id:NEW"


def test_extract_subcall_finds_originating_user_id(monkeypatch: pytest.MonkeyPatch):
    """After a tool result, the LAST role is `tool`. Walk back to user."""
    _enable_threading(monkeypatch)
    body = {
        "messages": [
            _system("you are a bot"),
            _user("ORIG", "do the thing"),
            _assistant("calling tool"),
            _tool("result data"),
        ],
    }
    out = app._extract_thread_id(body, header_value=None)
    assert out == "openclaw_message_id:ORIG"


def test_extract_cron_task_id(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch)
    body = {"messages": [_user(None, "[cron:d592d4a6-0b79-40b5-be09-5f2cf40cf4a1 watch] go")]}
    out = app._extract_thread_id(body, header_value=None)
    assert out == "cron_task_id:d592d4a6-0b79-40b5-be09-5f2cf40cf4a1"


def test_extract_fallback_hash_when_no_extractor_matches(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, fallback_hash=True)
    body = {"messages": [_system("sys"), _user(None, "plain ask")]}
    out = app._extract_thread_id(body, header_value=None)
    assert out is not None
    assert out.startswith("fallback:")


def test_extract_fallback_hash_advances_on_new_user_turn(monkeypatch: pytest.MonkeyPatch):
    """Latest-user (not first-user) anchor — different ask → different thread."""
    _enable_threading(monkeypatch, fallback_hash=True)
    sys_msg = _system("sys")
    body_a = {"messages": [sys_msg, _user(None, "ask A")]}
    body_b = {"messages": [sys_msg, _user(None, "ask A"), _assistant("ok"),
                             _user(None, "ask B")]}
    a = app._extract_thread_id(body_a, header_value=None)
    b = app._extract_thread_id(body_b, header_value=None)
    assert a is not None and b is not None
    assert a != b


def test_extract_fallback_hash_stable_across_subcalls(monkeypatch: pytest.MonkeyPatch):
    """Same latest-user message + new tool result → same thread id."""
    _enable_threading(monkeypatch, fallback_hash=True)
    sys_msg = _system("sys")
    user = _user(None, "do the thing")
    body_a = {"messages": [sys_msg, user]}
    body_b = {"messages": [sys_msg, user, _assistant("calling tool"), _tool("r1")]}
    a = app._extract_thread_id(body_a, header_value=None)
    b = app._extract_thread_id(body_b, header_value=None)
    assert a == b


def test_extract_fallback_disabled_returns_none(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, fallback_hash=False, extractors=[])
    body = {"messages": [_user(None, "no extractor will match this")]}
    assert app._extract_thread_id(body, header_value=None) is None


# --- _apply_parent_tier_policy --------------------------------------------


def test_parent_policy_inform_does_not_change_tier(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, parent_tier_policy="inform")
    assert app._apply_parent_tier_policy("opus", "local") == "opus"


def test_parent_policy_ignore_does_not_change_tier(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, parent_tier_policy="ignore")
    assert app._apply_parent_tier_policy("opus", "local") == "opus"


def test_parent_policy_cap_caps_child_above_parent(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, parent_tier_policy="cap")
    # default tiers: local, local-thinking, haiku, sonnet, opus (insertion order)
    assert app._apply_parent_tier_policy("opus", "local") == "local"
    assert app._apply_parent_tier_policy("sonnet", "local-thinking") == "local-thinking"


def test_parent_policy_cap_does_not_upgrade(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, parent_tier_policy="cap")
    assert app._apply_parent_tier_policy("local", "opus") == "local"


def test_parent_policy_cap_handles_missing_parent(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, parent_tier_policy="cap")
    assert app._apply_parent_tier_policy("opus", None) == "opus"


# --- _apply_thread_sticky --------------------------------------------------


def test_thread_sticky_keeps_thinking_within_thread(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, adopt_sticky_pairs=True)
    parent = {"previous_tier": "local-thinking"}
    # default sticky_pairs keep local-thinking when current would flip to local
    assert app._apply_thread_sticky("local", parent, now=0.0) == "local-thinking"


def test_thread_sticky_no_change_when_no_parent(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch)
    assert app._apply_thread_sticky("local", {"previous_tier": None}, now=0.0) == "local"


# --- _build_classification_text -------------------------------------------


def test_build_classification_originating_uses_latest_user_text(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, classify_subcall_isolated=True)
    body = {"messages": [_system("sys"), _user("X", "real ask")]}
    out = app._build_classification_text(body, is_subcall=False)
    assert "real ask" in out
    assert "trailing message" not in out  # originating path has no role labels


def test_build_classification_subcall_includes_trailing_tool(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, classify_subcall_isolated=True)
    body = {
        "messages": [
            _system("sys"),
            _user("X", "user ask"),
            _assistant("calling tool"),
            _tool("a long tool result with details"),
        ],
    }
    out = app._build_classification_text(body, is_subcall=True)
    assert "Original user ask" in out
    assert "user ask" in out
    assert "trailing message" in out
    assert "tool result with details" in out
    assert "role=tool" in out


def test_build_classification_subcall_isolation_off(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, classify_subcall_isolated=False)
    body = {
        "messages": [
            _user("X", "user ask"),
            _tool("result"),
        ],
    }
    out = app._build_classification_text(body, is_subcall=True)
    # Falls back to plain latest-user text.
    assert out == "user ask"


# --- _claim_thread_state (admission-time race safety) ---------------------


@pytest.mark.asyncio
async def test_claim_first_request_is_originating(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch)
    monkeypatch.setattr(app, "_THREAD_STATE", {})
    state = await app._claim_thread_state("openclaw_message_id:fresh", now=1000.0)
    assert state["is_originating"] is True
    assert state["origin_request_id"] is None


@pytest.mark.asyncio
async def test_claim_after_record_is_subcall(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch)
    monkeypatch.setattr(app, "_THREAD_STATE", {})
    monkeypatch.setattr(app, "_lookup_thread_state_in_db", lambda *a, **k: None)
    tid = "openclaw_message_id:abc"
    first = await app._claim_thread_state(tid, now=1000.0)
    assert first["is_originating"]
    app._record_thread_state(
        tid, origin_request_id=42, previous_request_id=42,
        tier="sonnet", complexity=4, now=1000.5,
    )
    second = await app._claim_thread_state(tid, now=1001.0)
    assert second["is_originating"] is False
    assert second["origin_request_id"] == 42
    assert second["previous_tier"] == "sonnet"
    assert second["previous_complexity"] == 4


@pytest.mark.asyncio
async def test_claim_after_ttl_is_originating_again(monkeypatch: pytest.MonkeyPatch):
    _enable_threading(monkeypatch, ttl_s=10.0)
    monkeypatch.setattr(app, "_THREAD_STATE", {})
    monkeypatch.setattr(app, "_lookup_thread_state_in_db", lambda *a, **k: None)
    tid = "openclaw_message_id:abc"
    await app._claim_thread_state(tid, now=1000.0)
    app._record_thread_state(
        tid, origin_request_id=1, previous_request_id=1,
        tier="local", complexity=2, now=1000.5,
    )
    later = await app._claim_thread_state(tid, now=1100.0)
    assert later["is_originating"] is True
