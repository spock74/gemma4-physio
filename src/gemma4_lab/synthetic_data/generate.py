"""Phase 4 placeholder: synthesize library-task Q&A data via Gemini.

# TODO Phase 4:
# - Define a small fictional book corpus (~50 books) — JSON under data/raw/.
# - Generate ~5k Q&A pairs spanning: metadata lookup, plot summaries, theme matching,
#   recommendation chains, multi-tool reasoning.
# - Output JSONL under data/synthetic/ with stable schema:
#   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "tools_used": [...]}
# - Use `GeminiClient` for the calls; chunk + cache to keep cost predictable.
# - Wrap the batch in Logfire spans; record cost via response usage_metadata.
"""

from __future__ import annotations
