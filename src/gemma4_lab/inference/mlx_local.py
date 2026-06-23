"""Phase 2 placeholder: Apple MLX backend (mlx-lm) for Gemma 4.

# TODO Phase 2: implement MLXLocal mirroring GemmaLocal's interface.
# - Same `GenerationResult` contract.
# - Same `generate(messages, thinking=False, **kw)` signature.
# - Wrap each call in a `logfire.span("mlx_local.generate", ...)`.
# - Read MLX checkpoint path from a new `Settings.mlx_checkpoint_path` field.
"""

from __future__ import annotations

from typing import Any


class MLXLocal:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Phase 2: MLX backend not yet implemented.")
