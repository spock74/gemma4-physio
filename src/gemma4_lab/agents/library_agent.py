"""Phase 5 placeholder: the library agent.

# TODO Phase 5:
# - Build a pydantic-ai Agent backed by local Gemma 4 (via a custom Model adapter
#   that wraps `inference.hf_local.GemmaLocal`).
# - Register the tools from `tools/library.py`.
# - System prompt: "You are a librarian assistant for a small fictional book corpus.
#   Use the available tools rather than guessing. Cite book IDs in answers."
# - Logfire instrumentation comes for free via `logfire.instrument_pydantic_ai()`.
"""

from __future__ import annotations
