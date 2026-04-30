# llmrouter

A small, OpenAI-compatible HTTP router that sits in front of [LiteLLM](https://github.com/BerriAI/litellm) and picks the right model **tier** for each `/v1/chat/completions` request — based on prompt complexity, tool-use, prompt length, and a regex+LLM **secret detector** that keeps credentials off cloud APIs. Every request is logged to a local SQLite database and rendered in a built-in web UI with full-text search.

> Status: **alpha**. Extracted from a personal homeserver setup. APIs may shift between minor versions until 1.0.

## What it does

For each request, the router:

1. **Classifies** the latest user message — by default with one call to a local model (Qwen, etc.) returning `{has_secret, secret_values, complexity}` as JSON. Falls back to a regex+heuristic pipeline (and an optional Haiku tiebreaker) on any error or timeout.
2. **Routes** to a configured tier: secrets always go to a local model; complexity 1-2 → cheap, 3 → medium, 4 → strong, 5 → strongest. The full mapping is YAML-configurable.
3. **Health-gates** local tiers: when Ollama is unreachable, transparently promotes to a configured cloud fallback tier and fires an [ntfy](https://ntfy.sh) alert on transitions.
4. **Logs** the request to SQLite (FTS5-indexed) with the secret value scrubbed from the preview, and pushes a live update to any connected `/ui/ws` client.
5. **Forwards** the request to LiteLLM with the `model` field rewritten to the picked tier's upstream model name.

Designed for self-hosted setups that proxy multiple agents (chat UIs, IDE assistants, Matrix bots…) through a single OpenAI-compatible endpoint and want sane cost control + secret hygiene without paying the per-request overhead of a heavyweight gateway.

## How it compares

| | llmrouter | LiteLLM (alone) | RouteLLM | OpenRouter |
|---|---|---|---|---|
| OpenAI-compatible | ✅ | ✅ | ✅ | ✅ |
| Self-hosted | ✅ | ✅ | ✅ | ❌ |
| Secret detection + redaction | ✅ | ❌ | ❌ | ❌ |
| Built-in audit-log UI | ✅ | partial | ❌ | dashboard (hosted) |
| Local-model fallback | ✅ | ❌ | ❌ | ❌ |
| Multi-classifier modes | ✅ | ❌ | ✅ (paper) | ❌ |
| YAML-driven tier policy | ✅ | partial | ❌ | ❌ |

llmrouter is *complementary* to LiteLLM — it sits in front of it and uses LiteLLM as the actual provider router.

## Quickstart (Docker)

```bash
docker run --rm -p 8000:8000 \
  -e LITELLM_BASE_URL=http://host.docker.internal:4000/v1 \
  -e LITELLM_API_KEY=sk-litellm \
  -e ROUTER_API_KEY=sk-router-anything \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  -v "$PWD/data:/data" \
  ghcr.io/dantebarbieri/llmrouter:latest
```

Then point any OpenAI client at `http://localhost:8000/v1` with `Authorization: Bearer sk-router-anything` and request the special model name `auto` to enable classification:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-router-anything" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

The web UI is at <http://localhost:8000/ui/>.

## Configuration

Two layers:

1. **YAML config** — tier definitions, classifier behavior, secret patterns, heuristic thresholds, hysteresis, log limits. Loaded from `$LLMROUTER_CONFIG` (default `/etc/llmrouter/config.yaml`). See [`llmrouter/config.default.yaml`](llmrouter/config.default.yaml) for the full schema with comments.
2. **Environment variables** — runtime URLs, API keys, paths, log knobs:

   | Variable | Required | Default | Purpose |
   |---|---|---|---|
   | `LITELLM_BASE_URL` | yes | — | LiteLLM `/v1` endpoint |
   | `LITELLM_API_KEY` | yes | — | LiteLLM auth |
   | `ROUTER_API_KEY` | yes | — | Bearer token clients send to the router |
   | `OLLAMA_URL` | no | `http://localhost:11434` | Ollama health-probe URL |
   | `NTFY_URL` | no | `""` (disabled) | ntfy server for health alerts |
   | `NTFY_TOPIC` | no | `llmrouter` | ntfy topic |
   | `NTFY_TOKEN` | no | `""` | ntfy auth |
   | `LLMROUTER_CONFIG` | no | `/etc/llmrouter/config.yaml` | Path to YAML config (missing file → defaults) |
   | `LLMROUTER_DB_PATH` | no | `/data/llmrouter.db` | SQLite log DB |
   | `LOG_LEVEL` | no | `info` | uvicorn + app log level |
   | `LOG_REQUESTS` | no | `1` | Disable request logging |

A request can also explicitly request a tier (e.g. `"model": "sonnet"`) to skip classification, or any other model name to pass through unchanged.

### Thread-aware routing

When `threading.enabled: true`, the router groups tool/sub-call requests under their originating user ask so:

- the UI can render a tree (collapsed sub-calls under the parent row), and
- the classifier can use parent context as a *signal* — a 4/5 parent doesn't cap a genuinely-5/5 sub-step.

Thread keys are extracted in priority order:

1. `X-LLMRouter-Thread-Id` request header (escape hatch).
2. Configurable regex extractors against the **last user message** only (so sub-calls in a burst share their originating ask's id, not the conversation's first id). Defaults match OpenClaw `message_id` and `[cron:<uuid> ...]` task tags.
3. Stable hash of `(system_prompt[:512], latest_user_text[:2048])` when extractors miss (`fallback_hash: true`).

`parent_tier_policy` controls how parent context affects routing: `inform` (default; log only), `cap` (child cannot exceed parent tier), `ignore` (skip parent lookup). With `classify_subcall_isolated: true`, sub-calls are classified against an isolated text view (original ask + role-labeled trailing message) so tool-result follow-ups are scored on their own complexity. Backward compat: omit the `threading:` section and behavior is unchanged from earlier versions.

## Develop

```bash
git clone https://github.com/dantebarbieri/llmrouter.git
cd llmrouter
python -m venv .venv && . .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
ruff check .
```

## Roadmap (known pain points to iterate on)

- Pluggable classifier backends (HTTP webhook, custom Python entry point).
- Pluggable storage backends (Postgres for multi-replica deploys).
- Per-tenant API keys + per-key tier policy.

## License

[MIT](LICENSE).
