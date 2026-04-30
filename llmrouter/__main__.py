"""`python -m llmrouter` and `llmrouter` entry point — runs uvicorn."""
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    host = os.getenv("LLMROUTER_HOST", "0.0.0.0")
    port = int(os.getenv("LLMROUTER_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run("llmrouter.app:app", host=host, port=port, log_level=log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
