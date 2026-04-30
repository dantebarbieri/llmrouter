"""Unit tests for the classifier + routing helpers in app.py.

No network. No database. Pure functions only.
"""
import pytest

from llmrouter import app

# --- _parse_classifier_json ---

def test_parse_valid_json():
    out = app._parse_classifier_json(
        '{"has_secret": false, "complexity": 3, "reason": "moderate reasoning"}'
    )
    assert out == {"has_secret": False, "complexity": 3, "reason": "moderate reasoning", "secret_values": []}


def test_parse_with_markdown_fence():
    out = app._parse_classifier_json(
        '```json\n{"has_secret": true, "complexity": 1, "reason": "key assignment"}\n```'
    )
    assert out["has_secret"] is True
    assert out["complexity"] == 1


def test_parse_with_think_tag():
    out = app._parse_classifier_json(
        '<think>hmm let me see</think>{"has_secret": false, "complexity": 2, "reason": "x"}'
    )
    assert out["has_secret"] is False
    assert out["complexity"] == 2


def test_parse_with_trailing_prose():
    out = app._parse_classifier_json(
        'Here is my answer: {"has_secret": false, "complexity": 4, "reason": "multi-step"}\n'
        'Hope that helps!'
    )
    assert out["complexity"] == 4


def test_parse_clamps_complexity_high():
    out = app._parse_classifier_json('{"has_secret": false, "complexity": 7, "reason": "x"}')
    assert out["complexity"] == 5


def test_parse_clamps_complexity_low():
    out = app._parse_classifier_json('{"has_secret": false, "complexity": 0, "reason": "x"}')
    assert out["complexity"] == 1


def test_parse_coerces_string_bool():
    out = app._parse_classifier_json('{"has_secret": "yes", "complexity": 3, "reason": "x"}')
    assert out["has_secret"] is True


def test_parse_coerces_int_bool():
    out = app._parse_classifier_json('{"has_secret": 1, "complexity": 3, "reason": "x"}')
    assert out["has_secret"] is True


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        app._parse_classifier_json("")


def test_parse_prose_only_raises():
    with pytest.raises(ValueError):
        app._parse_classifier_json("I cannot classify this request.")


def test_parse_missing_fields_raises():
    with pytest.raises(ValueError):
        app._parse_classifier_json('{"has_secret": false}')


def test_parse_non_object_raises():
    with pytest.raises(ValueError):
        app._parse_classifier_json('[1, 2, 3]')


# --- _map_tier ---

@pytest.mark.parametrize("has_secret,complexity,expected", [
    (False, 1, "local"),
    (False, 2, "local"),
    (False, 3, "local-thinking"),
    (False, 4, "sonnet"),
    (False, 5, "opus"),
    (True,  1, "local"),
    (True,  2, "local"),
    (True,  3, "local"),
    (True,  4, "local"),
    (True,  5, "local"),
])
def test_map_tier(has_secret, complexity, expected):
    assert app._map_tier(has_secret, complexity) == expected


def test_secret_never_routes_to_cloud():
    for c in range(1, 6):
        assert app._map_tier(True, c) == "local"


# --- hysteresis -------------------------------------------------------------

def _reset_hysteresis():
    app._LAST_TIER_BY_CHAT.clear()


def test_hysteresis_promotes_local_to_thinking_after_thinking_session():
    _reset_hysteresis()
    now = 1000.0
    app._record_tier("room:!abc", "local-thinking", now)
    assert app._apply_hysteresis("local", "room:!abc", now + 5) == "local-thinking"


def test_hysteresis_passes_through_when_previous_was_local():
    _reset_hysteresis()
    now = 1000.0
    app._record_tier("room:!abc", "local", now)
    # Prior local shouldn't promote — no reason to stick on Instruct
    assert app._apply_hysteresis("local", "room:!abc", now + 5) == "local"


def test_hysteresis_expires_after_ttl():
    _reset_hysteresis()
    now = 1000.0
    app._record_tier("room:!abc", "local-thinking", now)
    # Past the TTL → no promotion
    assert app._apply_hysteresis("local", "room:!abc", now + app.CFG.hysteresis.ttl_s + 1) == "local"


def test_hysteresis_no_chat_id_noop():
    _reset_hysteresis()
    now = 1000.0
    app._record_tier(None, "local-thinking", now)
    assert app._apply_hysteresis("local", None, now + 5) == "local"


def test_hysteresis_doesnt_demote_cloud():
    _reset_hysteresis()
    now = 1000.0
    app._record_tier("room:!abc", "local-thinking", now)
    # Classifier routed to sonnet — hysteresis should not touch cloud decisions
    assert app._apply_hysteresis("sonnet", "room:!abc", now + 5) == "sonnet"
    assert app._apply_hysteresis("opus", "room:!abc", now + 5) == "opus"


def test_hysteresis_only_records_local_tiers():
    _reset_hysteresis()
    app._record_tier("room:!abc", "sonnet", 1000.0)
    app._record_tier("room:!abc", "opus", 1000.0)
    assert "room:!abc" not in app._LAST_TIER_BY_CHAT


def test_extract_chat_id_from_metadata_block():
    body = {
        "messages": [
            {"role": "user", "content": (
                'Conversation info:\n```json\n'
                '{"chat_id": "room:!IKHsmcvoWUABvnvcmj:danteb.com"}\n'
                '```\nHello'
            )},
        ],
    }
    assert app._extract_chat_id(body) == "room:!IKHsmcvoWUABvnvcmj:danteb.com"


def test_extract_chat_id_missing_is_none():
    body = {"messages": [{"role": "user", "content": "just a plain message"}]}
    assert app._extract_chat_id(body) is None


# --- regex_secret_hit ---

def _body_with(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def test_regex_credential_assignment():
    assert app.regex_secret_hit(_body_with("password=hunter2hunter")) == "credential_assignment"


def test_regex_bearer_token():
    assert app.regex_secret_hit(_body_with("Bearer abc123def456ghi789jkl")) == "bearer_token"


def test_regex_ssh_public_key():
    assert app.regex_secret_hit(
        _body_with("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleKeyDataHere1234")
    ) == "ssh_public_key"


def test_regex_pem_private_key():
    assert app.regex_secret_hit(
        _body_with("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
    ) == "pem_private_key"


def test_regex_dotenv_reference():
    assert app.regex_secret_hit(_body_with("check my .env for the value")) == "dotenv_reference"


def test_regex_benign_mention_not_flagged():
    assert app.regex_secret_hit(_body_with("how do I rotate an API key?")) is None


def test_regex_only_scans_latest_user():
    body = {
        "messages": [
            {"role": "user",      "content": "password=hunter2hunter"},
            {"role": "assistant", "content": "ok"},
            {"role": "user",      "content": "what's the weather?"},
        ],
    }
    assert app.regex_secret_hit(body) is None


# --- union safety net (hand-computed truth table) ---

@pytest.mark.parametrize("regex_hit,llm_secret,expected_union", [
    ("credential_assignment", True,  True),
    ("credential_assignment", False, True),
    (None,                    True,  True),
    (None,                    False, False),
])
def test_secret_union(regex_hit, llm_secret, expected_union):
    union = bool(regex_hit) or bool(llm_secret)
    assert union is expected_union


# --- _redact_for_log ---

def test_no_redact_when_clean():
    preview, kws = app._redact_for_log("hello world foo bar baz qux", None)
    assert preview == "hello world foo bar baz qux"
    assert "hello" in kws or "world" in kws


def test_redact_preview_truncated():
    text = "x" * 10_000
    preview, _ = app._redact_for_log(text, None)
    assert len(preview) == app.CFG.limits.messages_preview_chars


def test_redact_llm_only_nulls_preview_when_no_values():
    preview, kws = app._redact_for_log("some sensitive content", "llm_classifier")
    assert preview is None
    assert kws == []


def test_redact_llm_only_scrubs_when_values_given():
    preview, kws = app._redact_for_log(
        "My password is V31m4JINKIES!!!! and I use it for the test machine.",
        "llm_classifier",
        ["V31m4JINKIES!!!!"],
    )
    assert preview is not None
    assert "V31m4JINKIES" not in preview
    assert "[REDACTED:llm_classifier]" in preview
    assert "test machine" in preview


def test_redact_llm_values_with_regex_hit_scrubs_both():
    # Regex catches `password=foo` style; LLM identifies a separate prose
    # secret. Both should be scrubbed.
    preview, _ = app._redact_for_log(
        "earlier: password=hunter2hunter; also my pin is 999111 thanks",
        "credential_assignment",
        ["999111"],
    )
    assert "hunter2hunter" not in preview
    assert "999111" not in preview
    assert "[REDACTED:credential_assignment]" in preview
    assert "[REDACTED:llm_classifier]" in preview


def test_redact_regex_hit_scrubs_and_keeps_context():
    preview, kws = app._redact_for_log(
        "here is my password=hunter2hunter and other context",
        "credential_assignment",
    )
    assert preview is not None
    assert "hunter2hunter" not in preview
    assert "[REDACTED:credential_assignment]" in preview
    assert "other context" in preview


def test_redact_regex_hit_scrubs_multiple_patterns():
    text = "Bearer abcdefghijklmnop1234 and password=topsecret123"
    preview, _ = app._redact_for_log(text, "bearer_token")
    assert "abcdefghijklmnop1234" not in preview
    assert "topsecret123" not in preview
    assert "[REDACTED:bearer_token]" in preview
    assert "[REDACTED:credential_assignment]" in preview


# --- _scrub_secrets ---

def test_scrub_credential_assignment():
    assert app._scrub_secrets("password=hunter2hunter trailing") == \
        "[REDACTED:credential_assignment] trailing"


def test_scrub_no_matches_returns_unchanged():
    text = "nothing sensitive here"
    assert app._scrub_secrets(text) == text


# --- _scrub_with_values ---

def test_scrub_with_values_basic():
    out = app._scrub_with_values("hello WORLD goodbye", ["WORLD"])
    assert out == "hello [REDACTED:llm_classifier] goodbye"


def test_scrub_with_values_skips_short():
    out = app._scrub_with_values("ab xy abc def", ["ab"])  # len < 3
    assert out == "ab xy abc def"


def test_scrub_with_values_longest_first():
    # Make sure 'foo' inside 'foobar' doesn't strip first and break 'foobar'.
    out = app._scrub_with_values("here is foobar and foo separately", ["foo", "foobar"])
    assert "foobar" not in out
    assert out.count("[REDACTED:llm_classifier]") == 2


def test_scrub_with_values_empty_input_safe():
    assert app._scrub_with_values("", ["x"]) == ""
    assert app._scrub_with_values("hi", []) == "hi"
    assert app._scrub_with_values("hi", None) == "hi"  # type: ignore[arg-type]


# --- parser secret_values extraction ---

def test_parse_secret_values_extracted():
    out = app._parse_classifier_json(
        '{"has_secret": true, "secret_values": ["abc12345"], "complexity": 2, "reason": "x"}'
    )
    assert out["secret_values"] == ["abc12345"]


def test_parse_secret_values_default_empty():
    out = app._parse_classifier_json(
        '{"has_secret": false, "complexity": 3, "reason": "x"}'
    )
    assert out["secret_values"] == []


def test_parse_secret_values_filters_short_and_nonstrings():
    out = app._parse_classifier_json(
        '{"has_secret": true, "secret_values": ["ab", "abc", null, 12345], "complexity": 1, "reason": "x"}'
    )
    # "ab" filtered (len<3), null filtered, 12345 coerced to "12345"
    assert out["secret_values"] == ["abc", "12345"]


# --- _clean_user_text ---

def test_clean_user_text_strips_metadata_blocks():
    raw = (
        "Conversation info (untrusted metadata):\n"
        "```json\n"
        '{"chat_id": "room:!foo", "sender": "Dante"}\n'
        "```\n\n"
        "Sender (untrusted metadata):\n"
        "```json\n"
        '{"id": "@danteb:danteb.com"}\n'
        "```\n\n"
        "Can you check again?"
    )
    cleaned = app._clean_user_text(raw)
    assert "untrusted metadata" not in cleaned
    assert "chat_id" not in cleaned
    assert "@danteb" not in cleaned
    assert cleaned.endswith("Can you check again?")


def test_clean_user_text_passthrough_no_metadata():
    raw = "what is the capital of France?"
    assert app._clean_user_text(raw) == "what is the capital of France?"


def test_clean_user_text_preserves_user_code_blocks():
    raw = "Can you debug this?\n```python\nprint('hi')\n```"
    out = app._clean_user_text(raw)
    assert "print('hi')" in out
    assert "```python" in out


def test_clean_user_text_strips_only_labelled_blocks():
    raw = (
        "Here's my config (untrusted metadata):\n"
        "```json\n"
        '{"foo": 1}\n'
        "```\n"
        "And my question: what does foo mean?"
    )
    out = app._clean_user_text(raw)
    assert "foo" not in out.split("what does foo mean?")[0]
    assert "what does foo mean?" in out


# --- _serialize_full_messages ---

def test_serialize_full_preserves_structure():
    import json as _json
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user",   "content": "hi"},
    ]
    out = app._serialize_full_messages(messages, None)
    parsed = _json.loads(out)
    assert parsed[0]["role"] == "system"
    assert parsed[1]["content"] == "hi"


def test_serialize_full_nulls_on_llm_classifier_without_values():
    messages = [{"role": "user", "content": "sensitive stuff"}]
    assert app._serialize_full_messages(messages, "llm_classifier") is None


def test_serialize_full_scrubs_on_llm_classifier_with_values():
    import json as _json
    messages = [{"role": "user", "content": "My password is V31m4JINKIES!!!! please review"}]
    out = app._serialize_full_messages(messages, "llm_classifier", ["V31m4JINKIES!!!!"])
    assert out is not None
    parsed = _json.loads(out)
    assert "V31m4JINKIES" not in parsed[0]["content"]
    assert "[REDACTED:llm_classifier]" in parsed[0]["content"]
    assert "please review" in parsed[0]["content"]


def test_serialize_full_scrubs_on_regex_hit():
    import json as _json
    messages = [{"role": "user", "content": "password=hunter2hunter please debug"}]
    out = app._serialize_full_messages(messages, "credential_assignment")
    parsed = _json.loads(out)
    assert "hunter2hunter" not in parsed[0]["content"]
    assert "[REDACTED:credential_assignment]" in parsed[0]["content"]
    assert "please debug" in parsed[0]["content"]


def test_serialize_full_caps_giant_payload():
    huge = "x" * 1_000_000
    messages = [{"role": "user", "content": huge}]
    out = app._serialize_full_messages(messages, None)
    assert len(out) <= app.CFG.limits.messages_full_max_chars + 20  # truncation marker slack
    assert out.endswith("…[truncated]")


# --- _messages_text include_system ---

def test_messages_text_excludes_system():
    messages = [
        {"role": "system", "content": "you are a tool-using assistant with these tools..."},
        {"role": "user",   "content": "hello"},
    ]
    assert "tools" not in app._messages_text(messages, include_system=False)
    assert "hello" in app._messages_text(messages, include_system=False)


def test_messages_text_includes_system_by_default():
    messages = [
        {"role": "system", "content": "system boilerplate"},
        {"role": "user",   "content": "hello"},
    ]
    out = app._messages_text(messages)
    assert "system boilerplate" in out
    assert "hello" in out
