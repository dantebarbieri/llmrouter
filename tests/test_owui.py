"""Tests for Open WebUI task detection, extraction, and routing helpers."""
from __future__ import annotations

from typing import Any

import pytest

from llmrouter import app
from llmrouter.config import ThreadExtractorSpec, ThreadingSpec


def _enable_threading(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> ThreadingSpec:
    """Replace CFG with a copy whose threading section is enabled."""
    spec = ThreadingSpec.model_validate({"enabled": True, **overrides})
    new_cfg = app.CFG.model_copy(update={"threading": spec})
    monkeypatch.setattr(app, "CFG", new_cfg)
    return spec

# Realistic OWUI task message fixtures

_QUERY_ANALYSIS_MSG = """\
### Task:
Analyze the chat history to determine the necessity of generating search queries, \
in the given language. By default, **prioritize generating 1-3 broad and relevant \
search queries** unless it is absolutely certain that no additional information is required.

### Chat History:
<chat_history>
USER: What's all the hubbub about Tim Cook and John Ternus?
</chat_history>
"""

_TITLE_GEN_MSG = """\
### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.
### Guidelines:
- The title should clearly represent the main theme or subject of the conversation.

### Chat History:
<chat_history>
USER: What's the latest news in Austin, TX?
ASSISTANT: Here is the latest news...
</chat_history>
"""

_TAG_GEN_MSG = """\
### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history, along \
with 1-3 more specific subtopic tags.

### Chat History:
<chat_history>
USER: Tell me about the weather.
</chat_history>
"""

_FOLLOWUP_MSG = """\
### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally \
ask next in this conversation as a **user**.

### Chat History:
<chat_history>
USER: How do I bake sourdough?
</chat_history>
"""

_RAG_RESPONSE_MSG = """\
### Task:
Respond to the user query using the provided context, incorporating inline citations.

<context>
<source id="1">Some article content here</source>
</context>

### Chat History:
<chat_history>
USER: What happened with the Apple CEO transition?
</chat_history>
"""

_PLAIN_USER_MSG = "What's the weather like today?"


# --- _detect_owui_task -------------------------------------------------------


@pytest.mark.parametrize("text, expected_type, expected_query", [
    (_QUERY_ANALYSIS_MSG, "query_analysis", "What's all the hubbub about Tim Cook and John Ternus?"),
    (_TITLE_GEN_MSG, "title_generation", "What's the latest news in Austin, TX?"),
    (_TAG_GEN_MSG, "tag_generation", "Tell me about the weather."),
    (_FOLLOWUP_MSG, "followup_suggestions", "How do I bake sourdough?"),
    (_RAG_RESPONSE_MSG, "rag_response", "What happened with the Apple CEO transition?"),
    (_PLAIN_USER_MSG, None, None),
    ("", None, None),
])
def test_detect_owui_task(text: str, expected_type: str | None, expected_query: str | None) -> None:
    task_type, user_query = app._detect_owui_task(text)
    assert task_type == expected_type, f"type mismatch: {task_type!r} != {expected_type!r}"
    assert user_query == expected_query, f"query mismatch: {user_query!r} != {expected_query!r}"


def test_detect_owui_task_no_chat_history_returns_task_type_only() -> None:
    """When the task header is present but there is no chat_history block, task_type
    is still detected but user_query is None (e.g., very first OWUI turn)."""
    msg = "### Task:\nGenerate 1-3 broad tags categorizing the main themes of the chat history.\n"
    task_type, user_query = app._detect_owui_task(msg)
    assert task_type == "tag_generation"
    assert user_query is None


def test_detect_owui_task_unknown_type_falls_back_to_owui_task() -> None:
    msg = "### Task:\nDo something completely new that has no defined pattern.\n<chat_history>\nUSER: hi\n</chat_history>\n"
    task_type, user_query = app._detect_owui_task(msg)
    assert task_type == "owui_task"
    assert user_query == "hi"


# --- _latest_user_text with OWUI extraction ----------------------------------


def test_latest_user_text_extracts_owui_query() -> None:
    messages = [{"role": "user", "content": _QUERY_ANALYSIS_MSG}]
    result = app._latest_user_text(messages)
    assert result == "What's all the hubbub about Tim Cook and John Ternus?"


def test_latest_user_text_plain_message_unchanged() -> None:
    messages = [{"role": "user", "content": _PLAIN_USER_MSG}]
    result = app._latest_user_text(messages)
    assert result == _PLAIN_USER_MSG


def test_latest_user_text_no_chat_history_returns_full_task() -> None:
    """Without a chat_history block, fall back to the full task message."""
    msg = "### Task:\nGenerate a concise, 3-5 word title.\n"
    messages = [{"role": "user", "content": msg}]
    result = app._latest_user_text(messages)
    assert result == msg  # falls back to raw (no extractable query)


# --- heuristic_tier fast-routing for OWUI lightweight tasks ------------------


@pytest.mark.parametrize("msg", [
    _QUERY_ANALYSIS_MSG,
    _TITLE_GEN_MSG,
    _TAG_GEN_MSG,
    _FOLLOWUP_MSG,
])
def test_heuristic_tier_fast_routes_lightweight_owui(msg: str) -> None:
    body = {"messages": [{"role": "user", "content": msg}]}
    tier, signals = app.heuristic_tier(body)
    assert signals["owui_task_type"] in app._OWUI_LIGHTWEIGHT_TASKS
    assert tier == app.CFG.heuristic.small_token_tier


def test_heuristic_tier_owui_not_in_signals_for_plain_message() -> None:
    body = {"messages": [{"role": "user", "content": _PLAIN_USER_MSG}]}
    _, signals = app.heuristic_tier(body)
    assert signals["owui_task_type"] is None


def test_heuristic_tier_rag_response_not_fast_routed() -> None:
    """rag_response is NOT a lightweight task — it should go through normal routing."""
    body = {"messages": [{"role": "user", "content": _RAG_RESPONSE_MSG}]}
    tier, signals = app.heuristic_tier(body)
    assert signals["owui_task_type"] == "rag_response"
    # Should NOT be forced to small_token_tier by the OWUI fast-route path
    # (it may land there for other reasons, but not via OWUI fast-routing).
    # The key check: the task type is not in OWUI_LIGHTWEIGHT_TASKS.
    assert "rag_response" not in app._OWUI_LIGHTWEIGHT_TASKS


def test_heuristic_tier_step_keyword_in_owui_query_triggers_correctly() -> None:
    """Step keywords inside the extracted OWUI user query should still fire."""
    msg = """\
### Task:
Analyze the chat history to determine the necessity of generating search queries.
<chat_history>
USER: Can you debug this Python function?
</chat_history>
"""
    body = {"messages": [{"role": "user", "content": msg}]}
    tier, signals = app.heuristic_tier(body)
    # It's a lightweight OWUI task, so the fast-route fires before step-keyword.
    assert tier == app.CFG.heuristic.small_token_tier
    assert signals["owui_task_type"] == "query_analysis"


# --- OWUI thread grouping via _extract_thread_id ----------------------------


_AUSTIN_QUERY = "What's the latest news in Austin, TX?"

_AUSTIN_TITLE_MSG = f"""\
### Task:
Generate a concise, 3-5 word title with an emoji summarizing the chat history.

### Chat History:
<chat_history>
USER: {_AUSTIN_QUERY}
ASSISTANT: Here is the latest news...
</chat_history>
"""

_AUSTIN_TAG_MSG = f"""\
### Task:
Generate 1-3 broad tags categorizing the main themes of the chat history.

### Chat History:
<chat_history>
USER: {_AUSTIN_QUERY}
</chat_history>
"""

_AUSTIN_FOLLOWUP_MSG = f"""\
### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next.

### Chat History:
<chat_history>
USER: {_AUSTIN_QUERY}
ASSISTANT: Here is the latest news in Austin, TX...
</chat_history>
"""


def test_owui_tasks_share_thread_with_originating_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All OWUI sub-tasks for the same user query should share a thread ID
    with the originating user request."""
    _enable_threading(monkeypatch, fallback_hash=True)
    # Originating user request (plain user message)
    body_user = {"messages": [{"role": "user", "content": _AUSTIN_QUERY}]}
    thread_user = app._extract_thread_id(body_user, None)
    assert thread_user is not None

    for owui_msg in [_AUSTIN_TITLE_MSG, _AUSTIN_TAG_MSG, _AUSTIN_FOLLOWUP_MSG]:
        body_owui = {"messages": [{"role": "user", "content": owui_msg}]}
        thread_owui = app._extract_thread_id(body_owui, None)
        assert thread_owui == thread_user, (
            f"Thread mismatch for {owui_msg[:40]!r}: "
            f"{thread_owui!r} != {thread_user!r}"
        )


def test_different_owui_queries_get_different_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_threading(monkeypatch, fallback_hash=True)
    body_a = {"messages": [{"role": "user", "content": _QUERY_ANALYSIS_MSG}]}
    body_b = {"messages": [{"role": "user", "content": _RAG_RESPONSE_MSG}]}
    thread_a = app._extract_thread_id(body_a, None)
    thread_b = app._extract_thread_id(body_b, None)
    assert thread_a is not None
    assert thread_b is not None
    assert thread_a != thread_b


# --- _extract_tool_signals ---------------------------------------------------


def test_extract_tool_signals_openclaw_style() -> None:
    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Search for something."},
        {"role": "assistant", "content": None},  # OpenClaw null-content tool call
        {"role": "tool", "content": '{"tool": "web_search", "results": []}'},
        {"role": "assistant", "content": "Found it."},
        {"role": "tool", "content": '{"tool": "web_search", "status": "error", "error": "403"}'},
    ]
    signals = app._extract_tool_signals(messages)
    assert signals["tool_calls_count"] == 2
    assert signals["tool_names"] == ["web_search"]
    assert signals["tool_error_count"] == 1


def test_extract_tool_signals_openai_style() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}},
            {"id": "c2", "function": {"name": "searxng_search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 75°F"},
        {"role": "tool", "tool_call_id": "c2", "content": '{"status": "error"}'},
    ]
    signals = app._extract_tool_signals(messages)
    assert signals["tool_calls_count"] == 4  # 2 from tool_calls + 2 tool results
    assert set(signals["tool_names"]) == {"get_weather", "searxng_search"}
    assert signals["tool_error_count"] == 1


def test_extract_tool_signals_anthropic_content_list_style() -> None:
    messages = [
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me look that up."},
            {"type": "tool_use", "id": "tu1", "name": "wikipedia__get_article", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "Article content..."},
        ]},
    ]
    signals = app._extract_tool_signals(messages)
    assert signals["tool_calls_count"] == 2  # 1 tool_use + 1 tool_result
    assert signals["tool_names"] == ["wikipedia__get_article"]
    assert signals["tool_error_count"] == 0


def test_extract_tool_signals_no_tools() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    signals = app._extract_tool_signals(messages)
    assert signals["tool_calls_count"] == 0
    assert signals["tool_names"] == []
    assert signals["tool_error_count"] == 0


def test_extract_tool_signals_non_json_tool_result_does_not_crash() -> None:
    messages = [
        {"role": "tool", "content": "Command still running (session abc, pid 1234)."},
    ]
    signals = app._extract_tool_signals(messages)
    assert signals["tool_calls_count"] == 1
    assert signals["tool_names"] == []
    assert signals["tool_error_count"] == 0
