"""Phase 3 placeholder: synthetic-data → HF Datasets pipeline.

# TODO Phase 3:
# - Read JSONL produced by `synthetic_data.generate` from data/synthetic/.
# - Build a `datasets.Dataset` formatted for TRL `SFTTrainer` (chat template applied).
# - Train/val split, deduplication, basic length filtering.
# - Tokenize with the same tokenizer used by the target Gemma 4 variant.
"""

from __future__ import annotations
