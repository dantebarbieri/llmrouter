"""End-to-end replay tests that exercise chat_completions through a fake
upstream (no network) and assert thread grouping + parent-context
behavior.

Uses FastAPI's TestClient + httpx mocking pattern: monkeypatch
`httpx.AsyncClient` and `app.ollama_healthy` to keep things deterministic.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llmrouter import app
from llmrouter.config import ThreadExtractorSpec, ThreadingSpec

# --- fixtures --------------------------------------------------------------


@pytest.fixture
def isolated_db(monkeypatch: pytest.MonkeyPatch):
    """Point the app at a fresh sqlite file and re-init the schema.

    Uses TemporaryDirectory so both the file and its parent dir are
    cleaned up even if the test errors mid-flight.
    """
    with tempfile.TemporaryDirectory(prefix="llmrouter-e2e-") as tmpdir:
        db_path = os.path.join(tmpdir, f"e2e-{uuid.uuid4().hex}.db")
        new_env = app.ENV.model_copy(update={"db_path": db_path, "log_requests": True})
        monkeypatch.setattr(app, "ENV", new_env)
        monkeypatch.setattr(app, "_db_ready", False)
        monkeypatch.setattr(app, "_THREAD_STATE", {})
        yield db_path


@pytest.fixture
def threading_on(monkeypatch: pytest.MonkeyPatch):
    """Enable threading with default extractors (replaces frozen CFG)."""
    spec = ThreadingSpec.model_validate({
        "enabled": True,
        "parent_tier_policy": "inform",
        "classify_subcall_isolated": True,
        "fallback_hash": True,
        "extractors": [
            ThreadExtractorSpec.model_validate({
                "name": "openclaw_message_id",
                "source": "last_user_text",
                "pattern": r'"message_id"\s*:\s*"([^"]+)"',
            }),
        ],
    })
    new_cfg = app.CFG.model_copy(update={"threading": spec})
    monkeypatch.setattr(app, "CFG", new_cfg)


# --- fake upstream + fake classifier --------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any]):
        self.status_code = status_code
        self._body_obj = body
        self._body = json.dumps(body).encode()
        self.headers = {"content-type": "application/json"}
        self.text = self._body.decode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._body_obj

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        pass

    async def aiter_raw(self):  # pragma: no cover — non-stream path used here
        yield self._body


class _FakeClient:
    """Stand-in for httpx.AsyncClient that records and returns canned responses.

    Used both for upstream chat completions AND the local classifier call.
    The router opens a *new* AsyncClient per request, so we register a
    factory function on the module.
    """

    def __init__(self, classifier_complexity: int):
        self._classifier_complexity = classifier_complexity
        self.requests: list[dict[str, Any]] = []

    def _make_classifier_response(self, body_bytes: bytes) -> _FakeResponse:
        return _FakeResponse(200, {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "has_secret": False,
                        "secret_values": [],
                        "complexity": self._classifier_complexity,
                        "reason": "test",
                    }),
                },
            }],
        })

    def _make_upstream_response(self) -> _FakeResponse:
        return _FakeResponse(200, {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        })

    async def post(self, url: str, headers: dict[str, str] | None = None,
                   json: dict[str, Any] | None = None, **kw):
        self.requests.append({"url": url, "json": json})
        if json and json.get("max_tokens") == 120:
            return self._make_classifier_response(b"")
        return self._make_upstream_response()

    def build_request(self, method: str, url: str, headers: dict[str, str] | None = None,
                      json: dict[str, Any] | None = None, **kw):
        # Returned object's only contract: passed back to .send.
        return {"method": method, "url": url, "json": json}

    async def send(self, req: dict[str, Any], stream: bool = False):
        self.requests.append({"send": True, "url": req["url"]})
        return self._make_upstream_response()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        return None


@pytest.fixture
def fake_async_client(monkeypatch: pytest.MonkeyPatch):
    """Replace httpx.AsyncClient with a configurable fake. Returns a setter."""
    holder: dict[str, _FakeClient] = {"client": _FakeClient(classifier_complexity=4)}

    def factory(*a, **kw):
        return holder["client"]

    monkeypatch.setattr(app.httpx, "AsyncClient", factory)
    # Skip the ollama health probe — pretend it's healthy.

    async def healthy() -> bool:
        return True

    monkeypatch.setattr(app, "ollama_healthy", healthy)
    return holder


# --- helpers ---------------------------------------------------------------


def _openclaw_user(msg_id: str, text: str) -> dict[str, Any]:
    blob = (
        'Conversation info (untrusted metadata):\n'
        '```json\n'
        '{\n'
        f'  "message_id": "{msg_id}",\n'
        '  "chat_id": "room:!burst:host"\n'
        '}\n'
        '```\n\n'
    )
    return {"role": "user", "content": blob + text}


def _post(client: TestClient, body: dict[str, Any]):
    return client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {app.ENV.router_api_key}"},
    )


# --- tests -----------------------------------------------------------------


def test_burst_groups_under_one_thread(
    monkeypatch, isolated_db, threading_on, fake_async_client,
):
    """Replay an originating request + 5 sub-calls. Assert thread grouping."""
    client = TestClient(app.app)
    sys_msg = {"role": "system", "content": "you are a helpful agent"}
    orig = _openclaw_user("MSG-ORIG", "go plan a complex thing")

    # 1. Originating request.
    r1 = _post(client, {"model": "auto", "messages": [sys_msg, orig]})
    assert r1.status_code == 200
    assert r1.headers["x-llmrouter-thread-role"] == "originating"
    thread_hdr = r1.headers["x-llmrouter-thread-id"]
    assert thread_hdr.startswith("openclaw_message_id:MSG-ORIG")

    # 2-6. Sub-calls re-invoking with the same originating user message_id
    # plus a growing tool/assistant transcript.
    transcript = [sys_msg, orig]
    for i in range(5):
        transcript = transcript + [
            {"role": "assistant", "content": f"calling tool {i}"},
            {"role": "tool", "content": f"tool result {i}: ok"},
        ]
        r = _post(client, {"model": "auto", "messages": transcript})
        assert r.status_code == 200
        assert r.headers["x-llmrouter-thread-id"] == thread_hdr
        assert r.headers["x-llmrouter-thread-role"] == "subcall"

    # Inspect the DB: 6 rows, all sharing thread_id, all sharing
    # origin_request_id pointing at the originating row.
    conn = sqlite3.connect(isolated_db)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT id, thread_id, origin_request_id, thread_role, parent_complexity "
        "FROM requests ORDER BY id"
    ))
    conn.close()
    assert len(rows) == 6
    thread_ids = {r["thread_id"] for r in rows}
    assert thread_ids == {thread_hdr}
    origin_ids = {r["origin_request_id"] for r in rows}
    assert origin_ids == {rows[0]["id"]}, origin_ids
    assert rows[0]["thread_role"] == "originating"
    assert rows[0]["parent_complexity"] is None
    for sub in rows[1:]:
        assert sub["thread_role"] == "subcall"
        assert sub["parent_complexity"] is not None


def test_inform_policy_does_not_cap_subcall_tier(
    monkeypatch, isolated_db, threading_on, fake_async_client,
):
    """Parent classified as 4 → sonnet. Sub-call classifier returns 5 → opus.

    With parent_tier_policy=inform the child should NOT be capped at sonnet.
    """
    sys_msg = {"role": "system", "content": "agent"}
    orig = _openclaw_user("PARENT-A", "task")

    # Originating: classifier returns 4 → sonnet under default tier mapping.
    fake_async_client["client"] = _FakeClient(classifier_complexity=4)
    client = TestClient(app.app)
    r1 = _post(client, {"model": "auto", "messages": [sys_msg, orig]})
    assert r1.headers["x-llmrouter-tier"] == "sonnet"

    # Sub-call: classifier returns 5 → opus. Parent is sonnet (rank 3),
    # opus (rank 4). With `inform`, child should keep opus.
    fake_async_client["client"] = _FakeClient(classifier_complexity=5)
    r2 = _post(client, {
        "model": "auto",
        "messages": [
            sys_msg, orig,
            {"role": "assistant", "content": "thinking"},
            {"role": "tool", "content": "hard problem"},
        ],
    })
    assert r2.headers["x-llmrouter-tier"] == "opus"
    assert r2.headers["x-llmrouter-thread-role"] == "subcall"


def test_cap_policy_caps_subcall_tier(
    monkeypatch, isolated_db, fake_async_client,
):
    """parent_tier_policy=cap: child cannot exceed parent."""
    spec = ThreadingSpec.model_validate({
        "enabled": True,
        "parent_tier_policy": "cap",
        "classify_subcall_isolated": True,
        "fallback_hash": True,
        "extractors": [
            ThreadExtractorSpec.model_validate({
                "name": "openclaw_message_id",
                "source": "last_user_text",
                "pattern": r'"message_id"\s*:\s*"([^"]+)"',
            }),
        ],
    })
    new_cfg = app.CFG.model_copy(update={"threading": spec})
    monkeypatch.setattr(app, "CFG", new_cfg)

    sys_msg = {"role": "system", "content": "agent"}
    orig = _openclaw_user("PARENT-CAP", "task")

    # Parent: sonnet (complexity 4).
    fake_async_client["client"] = _FakeClient(classifier_complexity=4)
    client = TestClient(app.app)
    r1 = _post(client, {"model": "auto", "messages": [sys_msg, orig]})
    assert r1.headers["x-llmrouter-tier"] == "sonnet"

    # Sub-call: classifier wants opus (5). With cap, gets capped to sonnet.
    fake_async_client["client"] = _FakeClient(classifier_complexity=5)
    r2 = _post(client, {
        "model": "auto",
        "messages": [
            sys_msg, orig,
            {"role": "assistant", "content": "x"},
            {"role": "tool", "content": "y"},
        ],
    })
    assert r2.headers["x-llmrouter-tier"] == "sonnet"
    assert "parent-cap" in r2.headers["x-llmrouter-reason"]


def test_threading_disabled_no_thread_columns(
    monkeypatch, isolated_db, fake_async_client,
):
    """When threading.enabled=false, thread_id is NULL in the DB row."""
    spec = ThreadingSpec.model_validate({"enabled": False})
    new_cfg = app.CFG.model_copy(update={"threading": spec})
    monkeypatch.setattr(app, "CFG", new_cfg)

    sys_msg = {"role": "system", "content": "agent"}
    orig = _openclaw_user("X", "task")

    fake_async_client["client"] = _FakeClient(classifier_complexity=2)
    client = TestClient(app.app)
    r = _post(client, {"model": "auto", "messages": [sys_msg, orig]})
    assert r.status_code == 200
    assert "x-llmrouter-thread-id" not in r.headers

    conn = sqlite3.connect(isolated_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT thread_id, origin_request_id, thread_role FROM requests"
    ).fetchone()
    conn.close()
    assert row["thread_id"] is None
    assert row["origin_request_id"] is None
    assert row["thread_role"] is None
