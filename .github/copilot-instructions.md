# Copilot instructions for llmrouter

OpenAI-compatible HTTP router (FastAPI) that sits in front of LiteLLM. For each
`/v1/chat/completions` request it classifies the prompt, picks a configured
**tier**, rewrites `model`, forwards to LiteLLM, and logs everything to a local
SQLite + FTS5 DB exposed at `/ui/`.

## Build / test / lint

```bash
pip install -e ".[dev]"
pytest -q                                 # full suite
pytest tests/test_classifier.py -q        # one file
pytest tests/test_classifier.py::test_parse_valid_json   # one test
ruff check .
```

CI runs ruff + pytest on Python 3.11/3.12/3.13 (`.github/workflows/ci.yml`).
Target `py311` and ruff line length 110; ruleset `E,F,W,I,B,UP,SIM` with
`E501,B008,B017` ignored (see `pyproject.toml`).

`pytest-asyncio` is in `auto` mode — don't decorate async tests.

## Architecture

Two files do almost all the work:

- **`llmrouter/config.py`** — pydantic v2 schema for the YAML config. All
  models are `frozen=True`. Loading is: package-shipped `config.default.yaml`
  shallow-merged with `$LLMROUTER_CONFIG` (top-level keys are *replaced
  wholesale*, never deep-merged), then validated. Cross-references between
  sections (tier names referenced from `fallback`, `classifier`, `heuristic`,
  `hysteresis.sticky_pairs`) are checked in a `model_validator`.
- **`llmrouter/app.py`** — single FastAPI module containing the router, the
  classifier pipeline, the SQLite logger, and the `/ui/` endpoints + websocket
  hub. **Reads required env vars at import time** (`ENV = RuntimeEnv.from_env()`
  and `CFG = load_config(...)` run as module-level statements), so any test
  that imports `llmrouter.app` must seed `LITELLM_BASE_URL`,
  `LITELLM_API_KEY`, `ROUTER_API_KEY` first — see `tests/conftest.py` for the
  pattern (also sets `LLMROUTER_CONFIG=""` to force package defaults).

### Two-layer configuration (do not blur)

- **YAML** = policy (tiers, classifier mode, secret regexes, hysteresis,
  threading, limits). Loaded via `load_config()`.
- **Env vars** = runtime wiring (URLs, API keys, DB path, log level). Loaded
  via `RuntimeEnv.from_env()`. **Never** put secrets/URLs into YAML, and
  never read env vars outside `RuntimeEnv`.

### Request flow (`chat_completions` in `app.py`)

1. Auth check against `ROUTER_API_KEY`.
2. If `model` is a known tier name or anything other than `auto`, skip
   classification.
3. Otherwise: classify (mode set in `classifier.mode` — `local` calls a local
   model via LiteLLM and parses `{has_secret, secret_values, complexity}` JSON;
   `hybrid`/`haiku` use regex+heuristic with a Haiku tiebreaker). Regex
   `secret_patterns` always run as a safety-net union regardless of mode.
4. Map to tier via `_map_tier` / hysteresis / threading rules.
5. If selected tier is `kind: local` and Ollama is unhealthy, promote to
   `fallback.cloud_tier` and (on transition) fire ntfy.
6. Forward to LiteLLM with `model` rewritten to the tier's upstream name.
7. Log to SQLite (preview is `_redact_for_log`-scrubbed) and publish to the
   `/ui/ws` hub.

### Threading (`config.threading`)

Groups originating user turns with their tool-call / reasoning sub-calls into
one logical thread. Thread id comes from `threading.extractors` (regexes with
**exactly one capture group**, scanned against `last_user_text`), an explicit
`x-llmrouter-thread-id` header, or a stable hash fallback. Sub-calls store
`parent_request_id`. When `classify_subcall_isolated=True` the classifier sees
only the trailing message text, not full history — this is intentional and
fixes a real bug; don't "simplify" it back. `parent_tier_policy` controls
whether the parent's tier informs/caps the child.

### SQLite logging

DB path = `LLMROUTER_DB_PATH`. Schema is created and migrated in
`_init_db` / `_migrate_requests_schema` at startup; FTS5 virtual table mirrors
keywords + redacted preview. When changing logged columns, update both the
`CREATE TABLE` block and the migration helper.

## Conventions

- Pydantic models for config are `frozen=True` and use `model_validator(mode="before")`
  to compile regexes at load time; mirror that pattern when adding new
  regex-bearing config sections.
- Regex flags in YAML are uppercase strings (`IGNORECASE`, `MULTILINE`,
  `DOTALL`) translated by `_flags_to_int`.
- Keep secret values out of logs and UI: route any logged user text through
  `_scrub_secrets` / `_scrub_with_values` / `_redact_for_log`.
- Health, hysteresis, and thread-state caches are process-local dicts guarded
  by `asyncio.Lock`s. Don't reach into them from tests; prefer the pure
  helpers (`_parse_classifier_json`, `_map_tier`, `heuristic_tier`, etc.) which
  are what existing tests exercise.
