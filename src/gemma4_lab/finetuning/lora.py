"""Phase 3 placeholder: LoRA fine-tuning entry point.

# TODO Phase 3:
# - PEFT LoRA config (rank ~16, target_modules from Gemma 4 attention/MLP linear layers).
# - TRL `SFTTrainer` configured for bf16 on MPS — eval every N steps.
# - Logfire span wrapping the train loop; emit checkpoint paths and eval metrics.
# - Save LoRA adapters under models/adapters/<run-id>/.
"""

from __future__ import annotations
