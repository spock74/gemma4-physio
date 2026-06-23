"""Smoke tests — no model load, no network. Verifies imports + config + observability."""

from __future__ import annotations

import importlib

import pytest


def test_imports() -> None:
    """All public modules import cleanly."""
    import gemma4_lab  # noqa: F401
    import gemma4_lab.config  # noqa: F401
    import gemma4_lab.observability  # noqa: F401
    from gemma4_lab import agents, finetuning, synthetic_data, tools  # noqa: F401
    from gemma4_lab.inference import (  # noqa: F401
        GemmaLocal,
        GenerationResult,
        hf_local,
        llama_cpp,
        mlx_local,
    )
    from gemma4_lab.synthetic_data.gemini_client import GeminiClient  # noqa: F401

    assert gemma4_lab.__version__


def test_secret_constants_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys are read from os.environ at module import time, never hardcoded."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("LOGFIRE_TOKEN", "fake-logfire")
    monkeypatch.setenv("HF_TOKEN", "fake-hf")

    # Re-import so the module-level constants pick up the patched env.
    import gemma4_lab.config as cfg

    cfg = importlib.reload(cfg)

    assert cfg.GEMINI_API_KEY == "fake-gemini-key"
    assert cfg.LOGFIRE_TOKEN == "fake-logfire"
    assert cfg.HF_TOKEN == "fake-hf"

    # Helpers reflect the constants.
    assert cfg.require_gemini_key() == "fake-gemini-key"
    assert cfg.logfire_token_or_none() == "fake-logfire"
    assert cfg.hf_token_or_none() == "fake-hf"


def test_require_gemini_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import gemma4_lab.config as cfg

    cfg = importlib.reload(cfg)
    assert cfg.GEMINI_API_KEY == ""
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        cfg.require_gemini_key()


def test_settings_has_no_secret_fields() -> None:
    """Settings must NOT carry secrets (those live in module constants)."""
    from gemma4_lab.config import Settings

    field_names = set(Settings.model_fields.keys())
    forbidden = {"gemini_api_key", "logfire_token", "hf_token"}
    leaked = field_names & forbidden
    assert not leaked, f"Settings leaked secret fields: {leaked}"


def test_model_id_property() -> None:
    from gemma4_lab.config import Settings

    s_e2b = Settings(model_variant="e2b")  # type: ignore[arg-type]
    s_e4b = Settings(model_variant="e4b")  # type: ignore[arg-type]
    assert s_e2b.model_id == "google/gemma-4-E2B-it"
    assert s_e4b.model_id == "google/gemma-4-E4B-it"


def test_observability_setup_idempotent() -> None:
    """Calling setup() twice must not raise and must remain initialized."""
    from gemma4_lab import observability

    observability.setup()
    assert observability.is_initialized()
    observability.setup()  # second call — must not raise
    assert observability.is_initialized()


def test_phase2_backends_raise() -> None:
    from gemma4_lab.inference.llama_cpp import LlamaCppLocal
    from gemma4_lab.inference.mlx_local import MLXLocal

    with pytest.raises(NotImplementedError):
        MLXLocal()
    with pytest.raises(NotImplementedError):
        LlamaCppLocal()


def test_thought_split_helper() -> None:
    """Internal parser handles thinking-on and thinking-off outputs."""
    from gemma4_lab.inference.hf_local import _split_thought_and_answer

    # No thought channel — typical E2B/E4B no-thinking output.
    thought, answer = _split_thought_and_answer("Hello there.")
    assert thought is None
    assert answer == "Hello there."

    # With thought channel.
    decoded = "<|channel|>thought\nstep 1: foo\nstep 2: bar<|channel|>The answer is 42."
    thought, answer = _split_thought_and_answer(decoded)
    assert thought is not None and "step 1" in thought
    assert answer == "The answer is 42."
