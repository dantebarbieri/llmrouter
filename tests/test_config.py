"""Tests for the YAML config loader."""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from llmrouter.config import Config, RuntimeEnv, load_config

# --- defaults --------------------------------------------------------------


def test_load_defaults_when_no_path():
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert "local" in cfg.tiers
    assert "sonnet" in cfg.tiers
    assert cfg.fallback.cloud_tier == "sonnet"
    assert cfg.tiers["sonnet"].kind == "cloud"


def test_load_defaults_returns_compiled_secret_patterns():
    cfg = load_config(None)
    assert any(sp.name == "credential_assignment" for sp in cfg.secret_patterns)
    cred = next(sp for sp in cfg.secret_patterns if sp.name == "credential_assignment")
    assert cred.compiled.flags & re.IGNORECASE


def test_tier_to_model_view():
    cfg = load_config(None)
    assert cfg.tier_to_model["sonnet"] == "claude-sonnet"
    assert cfg.tier_to_model["local"] == "qwen-local"


def test_local_tiers_view():
    cfg = load_config(None)
    assert cfg.local_tiers == frozenset({"local", "local-thinking"})


def test_load_missing_file_returns_defaults(tmp_path: Path):
    # Non-existent path is silently treated as "no overrides".
    cfg = load_config(tmp_path / "does-not-exist.yaml")
    assert "local" in cfg.tiers


# --- overrides --------------------------------------------------------------


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_override_replaces_tiers_wholesale(tmp_path: Path):
    p = _write(tmp_path, """
        tiers:
          fast:  { model: gpt-4o-mini, kind: cloud }
          smart: { model: gpt-4o,      kind: cloud }
        fallback:
          cloud_tier: smart
        classifier:
          mode: hybrid
          local_model: ignored
          local_timeout_s: 5
          tiebreaker_model: gpt-4o-mini
          complexity_to_tier: { 1: fast, 2: fast, 3: smart, 4: smart, 5: smart }
          complexity_default_tier: smart
          secret_tier: fast
          tiebreaker_to_tier: { "1": fast, "2": fast, "3": smart, "4": smart, "5": smart }
          heuristic_default_tier: fast
        heuristic:
          small_token_threshold: 100
          small_token_tier: fast
          large_token_threshold: 4000
          large_token_tier: smart
          tools_tier: smart
          ambiguous_tier: null
          secret_hit_tier: fast
          step_keywords: ["plan "]
          stopwords: ["the"]
        secret_patterns: []
        hysteresis:
          enabled: false
          ttl_s: 60
          chat_id_pattern: 'chat:(.+)'
        limits:
          health_ttl_s: 1
          messages_preview_chars: 500
          messages_full_max_chars: 5000
    """)
    cfg = load_config(p)
    assert set(cfg.tiers) == {"fast", "smart"}
    assert cfg.fallback.cloud_tier == "smart"
    assert cfg.classifier.mode == "hybrid"
    assert cfg.heuristic.small_token_threshold == 100
    assert cfg.secret_patterns == ()


def test_override_subset_keeps_default_for_others(tmp_path: Path):
    # Only override `limits` — everything else keeps defaults.
    p = _write(tmp_path, """
        limits:
          health_ttl_s: 1.5
          messages_preview_chars: 1234
          messages_full_max_chars: 9999
    """)
    cfg = load_config(p)
    assert cfg.limits.messages_preview_chars == 1234
    assert "local" in cfg.tiers  # defaults preserved
    assert cfg.fallback.cloud_tier == "sonnet"


# --- validation -------------------------------------------------------------


def test_undefined_tier_reference_rejected(tmp_path: Path):
    p = _write(tmp_path, """
        fallback:
          cloud_tier: nonexistent
    """)
    with pytest.raises(Exception):  # pydantic ValidationError
        load_config(p)


def test_fallback_must_be_cloud_kind(tmp_path: Path):
    p = _write(tmp_path, """
        fallback:
          cloud_tier: local
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_invalid_regex_rejected(tmp_path: Path):
    p = _write(tmp_path, """
        secret_patterns:
          - { name: bad, pattern: '(unclosed', flags: [] }
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_unknown_flag_rejected(tmp_path: Path):
    p = _write(tmp_path, """
        secret_patterns:
          - { name: bad, pattern: 'x', flags: [NOT_A_REAL_FLAG] }
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_top_level_must_be_mapping(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(p)


# --- threading -------------------------------------------------------------


def test_threading_defaults_disabled():
    cfg = load_config(None)
    # Default loaded from llmrouter/config.default.yaml has enabled: false
    # so legacy behavior is preserved.
    assert cfg.threading.enabled is False
    assert cfg.threading.parent_tier_policy == "inform"


def test_threading_extractor_pattern_compiled(tmp_path: Path):
    p = _write(tmp_path, """
        threading:
          enabled: true
          extractors:
            - { name: foo, source: last_user_text, pattern: '"x":\\s*"([^"]+)"' }
    """)
    cfg = load_config(p)
    assert cfg.threading.enabled is True
    ex = cfg.threading.extractors[0]
    assert ex.compiled is not None
    assert ex.compiled.search('"x": "abc"').group(1) == "abc"


def test_threading_extractor_must_compile(tmp_path: Path):
    p = _write(tmp_path, """
        threading:
          enabled: true
          extractors:
            - { name: bad, source: last_user_text, pattern: '(unclosed' }
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_threading_extractor_must_have_one_capture_group(tmp_path: Path):
    p = _write(tmp_path, """
        threading:
          enabled: true
          extractors:
            - { name: bad, source: last_user_text, pattern: 'no-group-here' }
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_threading_extractor_rejects_multiple_capture_groups(tmp_path: Path):
    p = _write(tmp_path, """
        threading:
          enabled: true
          extractors:
            - { name: bad, source: last_user_text, pattern: '(a)(b)' }
    """)
    with pytest.raises(Exception):
        load_config(p)


def test_threading_parent_tier_policy_enum(tmp_path: Path):
    for policy in ("inform", "cap", "ignore"):
        p = _write(tmp_path, f"""
            threading:
              enabled: true
              parent_tier_policy: {policy}
        """)
        cfg = load_config(p)
        assert cfg.threading.parent_tier_policy == policy


def test_threading_parent_tier_policy_invalid_rejected(tmp_path: Path):
    p = _write(tmp_path, """
        threading:
          enabled: true
          parent_tier_policy: nuke
    """)
    with pytest.raises(Exception):
        load_config(p)


# --- runtime env -----------------------------------------------------------


def test_runtime_env_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm.example/v1/")
    monkeypatch.setenv("LITELLM_API_KEY", "lk")
    monkeypatch.setenv("ROUTER_API_KEY", "rk")
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.example/")
    monkeypatch.setenv("LLMROUTER_DB_PATH", "/tmp/x.db")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_REQUESTS", "0")
    env = RuntimeEnv.from_env()
    # Trailing slashes stripped from URLs.
    assert env.litellm_base_url == "https://litellm.example/v1"
    assert env.ollama_url == "http://ollama.example"
    assert env.log_level == "DEBUG"
    assert env.log_requests is False


def test_runtime_env_log_requests_truthy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "x")
    monkeypatch.setenv("LITELLM_API_KEY", "x")
    monkeypatch.setenv("ROUTER_API_KEY", "x")
    for v in ("1", "true", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("LOG_REQUESTS", v)
        assert RuntimeEnv.from_env().log_requests is True


def test_runtime_env_missing_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    with pytest.raises(KeyError):
        RuntimeEnv.from_env()
