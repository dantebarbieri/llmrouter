"""pytest fixtures for llmrouter tests.

`llmrouter.app` reads required env vars at import time. Seed placeholders so
`import llmrouter.app` succeeds in unit tests that don't actually hit any
network or DB.
"""
import os
import tempfile

os.environ.setdefault("LITELLM_BASE_URL", "http://unused/v1")
os.environ.setdefault("LITELLM_API_KEY", "test-key")
os.environ.setdefault("ROUTER_API_KEY", "test-key")
os.environ.setdefault(
    "LLMROUTER_DB_PATH",
    os.path.join(tempfile.gettempdir(), "llmrouter-test.db"),
)
os.environ.setdefault("LOG_REQUESTS", "0")
# Force the loader to fall back entirely to package defaults rather than
# attempt to read /etc/llmrouter/config.yaml (which doesn't exist in CI).
os.environ.setdefault("LLMROUTER_CONFIG", "")
