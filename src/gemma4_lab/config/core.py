"""Application config — single source of truth for paths, model choice, and secrets.

Secrets policy (project rule): API keys are NEVER hardcoded. They are read from
the OS environment via `os.getenv`, exported as module-level constants, and
imported where needed. A `.env` file at the project root is loaded into
`os.environ` at import time as a development convenience — but the canonical
source is the OS environment, which always wins over `.env`.

Usage:
    from gemma4_lab.config import GEMINI_API_KEY, require_gemini_key
    from gemma4_lab.config import LOGFIRE_TOKEN, HF_TOKEN
    from gemma4_lab.config import Settings    # paths, model variant — NO secrets
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ProjectRoot = Path(__file__).resolve().parents[2]


def _load_dotenv_into_environ() -> None:
    """Tiny in-process dotenv loader. Only fills variables not already in env.

    Kept inline to avoid pulling python-dotenv as a hard runtime dep just for
    development convenience. The OS environment is the canonical source.
    """
    env_path = ProjectRoot / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv_into_environ()

# ---------------------------------------------------------------------------
# Secret constants — read once at import time from the OS environment.
# Empty string means absent. Use the `require_*` helpers when a key is required.
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
LOGFIRE_TOKEN: str = os.getenv("LOGFIRE_TOKEN", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
NEURONPEDIA_API_KEY: str = os.getenv("NEURONPEDIA_API_KEY", "")


def require_gemini_key() -> str:
    """Return GEMINI_API_KEY or raise with a clear message."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it in your shell "
            "(e.g., `export GEMINI_API_KEY=…`) or add it to .env at the project root."
        )
    return GEMINI_API_KEY


def hf_token_or_none() -> str | None:
    """Return HF_TOKEN or None if absent (HF Hub allows unauthenticated reads
    for public weights, just with lower rate limits)."""
    return HF_TOKEN or None


def logfire_token_or_none() -> str | None:
    """Return LOGFIRE_TOKEN or None if absent (Logfire runs local-only without)."""
    return LOGFIRE_TOKEN or None


def neuronpedia_key_or_none() -> str | None:
    """Return NEURONPEDIA_API_KEY or None (Neuronpedia read endpoints are largely
    public; the key only lifts rate limits / unlocks write+inference)."""
    return NEURONPEDIA_API_KEY or None


# ---------------------------------------------------------------------------
# Typed non-secret config.
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Non-secret configuration (paths, model variant, device, dtype).

    Secrets do NOT live here — see the module-level constants above.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    model_variant: Literal["e2b", "e4b"] = "e2b"
    model_dtype: str = "bfloat16"
    device: str = "mps"

    hf_cache_dir: Path = ProjectRoot / "models" / "hf-cache"
    data_dir: Path = ProjectRoot / "data"
    models_dir: Path = ProjectRoot / "models"

    @property
    def model_id(self) -> str:
        """HF Hub repo for the configured Gemma 4 variant."""
        return f"google/gemma-4-{self.model_variant.upper()}-it"
