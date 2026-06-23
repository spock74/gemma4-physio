"""Local inference backends.

Phase 1 ships `hf_local.GemmaLocal` (HF Transformers, bf16, MPS).
Phase 2 will add `mlx_local.MLXLocal` and `llama_cpp.LlamaCppLocal`.

All backends share the `GenerationResult` contract from `hf_local`.
"""

from .hf_local import GemmaLocal, GenerationResult

__all__ = ["GemmaLocal", "GenerationResult"]
