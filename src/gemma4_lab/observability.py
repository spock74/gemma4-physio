"""Logfire bootstrap. Call `setup()` once near process start; idempotent.

Reads `LOGFIRE_TOKEN` from the config module's exported constant — never from
direct `os.getenv` calls scattered across the code.
"""

from __future__ import annotations

from .config import logfire_token_or_none

_initialized: bool = False


def setup() -> None:
    """Configure Logfire. Safe to call multiple times — only configures once."""
    global _initialized
    if _initialized:
        return

    import logfire

    logfire.configure(
        service_name="gemma4-lab",
        service_version="0.1.0",
        token=logfire_token_or_none(),
        send_to_logfire="if-token-present",
        console=False,
    )

    # Instruments — fail-soft; missing extras shouldn't crash the app.
    try:
        logfire.instrument_pydantic_ai()
    except Exception:
        pass
    try:
        logfire.instrument_httpx(capture_all=False)
    except Exception:
        pass
    try:
        logfire.instrument_system_metrics()
    except Exception:
        pass

    _initialized = True


def is_initialized() -> bool:
    return _initialized
