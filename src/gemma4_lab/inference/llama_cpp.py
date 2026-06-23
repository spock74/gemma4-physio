"""Phase 2 placeholder: llama.cpp / GGUF backend (llama-cpp-python) for Gemma 4.

# TODO Phase 2: implement LlamaCppLocal mirroring GemmaLocal's interface.
# - Use unsloth's GGUF release (`unsloth/gemma-4-E2B-it-GGUF`) as the default.
# - Same `GenerationResult` contract.
# - Wrap each call in a `logfire.span("llama_cpp.generate", ...)`.
"""

from __future__ import annotations

from typing import Any


class LlamaCppLocal:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Phase 2: llama.cpp backend not yet implemented.")
