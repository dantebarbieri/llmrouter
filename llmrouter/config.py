"""Configuration loader for llmrouter.

Loads YAML from $LLMROUTER_CONFIG (default /etc/llmrouter/config.yaml),
merges shallow over `config.default.yaml` shipped with the package,
validates with pydantic, and exposes a frozen `Config` instance.

Runtime-only settings (URLs, API keys, DB paths, log knobs) are read from
environment variables — never from the config file — so containers can be
configured without baking secrets into a YAML.
"""
from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- env-var (runtime) settings ---------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


class RuntimeEnv(BaseModel):
    """Settings sourced from environment variables, not YAML."""

    model_config = ConfigDict(frozen=True)

    litellm_base_url: str
    litellm_api_key: str
    router_api_key: str
    ollama_url: str
    ntfy_url: str
    ntfy_topic: str
    ntfy_token: str
    db_path: str
    log_level: str
    log_requests: bool
    config_path: str

    @classmethod
    def from_env(cls) -> RuntimeEnv:
        return cls(
            litellm_base_url=os.environ["LITELLM_BASE_URL"].rstrip("/"),
            litellm_api_key=os.environ["LITELLM_API_KEY"],
            router_api_key=os.environ["ROUTER_API_KEY"],
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ntfy_url=os.getenv("NTFY_URL", "").rstrip("/"),
            ntfy_topic=os.getenv("NTFY_TOPIC", "llmrouter"),
            ntfy_token=os.getenv("NTFY_TOKEN", ""),
            db_path=os.getenv("LLMROUTER_DB_PATH", "/data/llmrouter.db"),
            log_level=os.getenv("LOG_LEVEL", "info").upper(),
            log_requests=_env_bool("LOG_REQUESTS", True),
            config_path=os.getenv("LLMROUTER_CONFIG", "/etc/llmrouter/config.yaml"),
        )


# --- YAML schema ------------------------------------------------------------

_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


def _flags_to_int(flags: list[str]) -> int:
    out = 0
    for f in flags:
        try:
            out |= _FLAGS[f.upper()]
        except KeyError as e:
            raise ValueError(f"unknown regex flag: {f!r}") from e
    return out


class TierSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    kind: Literal["local", "cloud"] = "cloud"


class FallbackSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    cloud_tier: str
    ntfy_on_transition: bool = True


class ClassifierSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: str = "local"
    local_model: str = "qwen-local-classifier"
    local_timeout_s: float = 5.0
    tiebreaker_model: str = "claude-haiku"
    complexity_to_tier: dict[int, str]
    complexity_default_tier: str
    secret_tier: str
    tiebreaker_to_tier: dict[str, str]
    heuristic_default_tier: str

    @field_validator("complexity_to_tier", mode="before")
    @classmethod
    def _coerce_int_keys(cls, v: Any) -> dict[int, str]:
        if not isinstance(v, dict):
            raise ValueError("complexity_to_tier must be a mapping")
        return {int(k): str(val) for k, val in v.items()}


class HeuristicSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    small_token_threshold: int
    small_token_tier: str
    large_token_threshold: int
    large_token_tier: str
    tools_tier: str
    ambiguous_tier: str | None
    secret_hit_tier: str
    step_keywords: tuple[str, ...]
    stopwords: frozenset[str]

    @field_validator("step_keywords", mode="before")
    @classmethod
    def _to_tuple(cls, v: Any) -> tuple[str, ...]:
        return tuple(v or [])

    @field_validator("stopwords", mode="before")
    @classmethod
    def _to_frozenset(cls, v: Any) -> frozenset[str]:
        return frozenset(v or [])


class SecretPatternSpec(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    pattern: str
    flags: tuple[str, ...] = ()
    compiled: re.Pattern[str]

    @model_validator(mode="before")
    @classmethod
    def _compile(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        flags = v.get("flags") or []
        try:
            v["compiled"] = re.compile(v["pattern"], _flags_to_int(list(flags)))
        except re.error as e:
            raise ValueError(f"invalid regex for secret_pattern {v.get('name')!r}: {e}") from e
        v["flags"] = tuple(flags)
        return v


class MetadataStripSpec(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    pattern: str
    flags: tuple[str, ...] = ()
    compiled: re.Pattern[str]

    @model_validator(mode="before")
    @classmethod
    def _compile(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        flags = v.get("flags") or []
        try:
            v["compiled"] = re.compile(v["pattern"], _flags_to_int(list(flags)))
        except re.error as e:
            raise ValueError(f"invalid regex for metadata_strip_pattern: {e}") from e
        v["flags"] = tuple(flags)
        return v


class StickyPair(BaseModel):
    model_config = ConfigDict(frozen=True)
    from_prev: str
    when_now: str
    keep: str


class HysteresisSpec(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    enabled: bool = True
    ttl_s: float = 600.0
    chat_id_pattern: str
    chat_id_re: re.Pattern[str]
    sticky_pairs: tuple[StickyPair, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _compile(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        try:
            v["chat_id_re"] = re.compile(v["chat_id_pattern"])
        except re.error as e:
            raise ValueError(f"invalid hysteresis.chat_id_pattern: {e}") from e
        return v


class LimitsSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    health_ttl_s: float = 5.0
    messages_preview_chars: int = Field(default=4000, gt=0)
    messages_full_max_chars: int = Field(default=200_000, gt=0)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tiers: dict[str, TierSpec]
    fallback: FallbackSpec
    classifier: ClassifierSpec
    heuristic: HeuristicSpec
    secret_patterns: tuple[SecretPatternSpec, ...]
    metadata_strip_patterns: tuple[MetadataStripSpec, ...] = ()
    hysteresis: HysteresisSpec
    limits: LimitsSpec

    @model_validator(mode="after")
    def _validate_tier_refs(self) -> Config:
        names = set(self.tiers)

        def must_exist(field: str, value: str | None) -> None:
            if value is not None and value not in names:
                raise ValueError(f"{field} references undefined tier: {value!r}")

        must_exist("fallback.cloud_tier", self.fallback.cloud_tier)
        if self.tiers[self.fallback.cloud_tier].kind != "cloud":
            raise ValueError(
                f"fallback.cloud_tier {self.fallback.cloud_tier!r} must be a cloud-kind tier"
            )
        must_exist("classifier.complexity_default_tier", self.classifier.complexity_default_tier)
        must_exist("classifier.secret_tier", self.classifier.secret_tier)
        must_exist("classifier.heuristic_default_tier", self.classifier.heuristic_default_tier)
        for k, t in self.classifier.complexity_to_tier.items():
            must_exist(f"classifier.complexity_to_tier[{k}]", t)
        for k, t in self.classifier.tiebreaker_to_tier.items():
            must_exist(f"classifier.tiebreaker_to_tier[{k}]", t)
        for f in (
            "small_token_tier",
            "large_token_tier",
            "tools_tier",
            "ambiguous_tier",
            "secret_hit_tier",
        ):
            must_exist(f"heuristic.{f}", getattr(self.heuristic, f))
        for sp in self.hysteresis.sticky_pairs:
            must_exist("hysteresis.sticky_pairs.from_prev", sp.from_prev)
            must_exist("hysteresis.sticky_pairs.when_now", sp.when_now)
            must_exist("hysteresis.sticky_pairs.keep", sp.keep)
        return self

    # --- convenience derived views ---

    @property
    def tier_to_model(self) -> dict[str, str]:
        return {name: t.model for name, t in self.tiers.items()}

    @property
    def local_tiers(self) -> frozenset[str]:
        return frozenset(name for name, t in self.tiers.items() if t.kind == "local")


# --- loading ---------------------------------------------------------------


def _read_default_yaml() -> dict[str, Any]:
    text = resources.files("llmrouter").joinpath("config.default.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("config.default.yaml must be a mapping at the top level")
    return data


def _read_user_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping")
    return data


def _shallow_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Per-top-level-section override.

    `tiers`, `secret_patterns`, etc. are replaced wholesale when the user
    sets them — no deep merge. This keeps semantics predictable: if you
    define `tiers:` in your config, those are the only tiers.
    """
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate the config. Falls back to package defaults."""
    base = _read_default_yaml()
    if path:
        user = _read_user_yaml(path)
        merged = _shallow_merge(base, user)
    else:
        merged = base
    return Config.model_validate(merged)
