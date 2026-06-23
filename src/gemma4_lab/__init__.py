"""gemma4-lab — local Gemma 4 inference, fine-tuning, and agent lab."""

import os as _os

__version__ = "0.1.0"

# Gemma 4 E2B/E4B both ship as ~9.5 GB safetensors (MatFormer: E2B is a runtime
# sub-extraction of the E4B weight set). On 16 GB Apple Silicon, MPS's default
# memory watermark rejects single buffers > ~7 GB with "Invalid buffer size".
# Lifting the watermark lets the OS swap if needed; the unified-memory + NVMe
# combination handles it without crashing.
# Must be set before `import torch` anywhere in the process — keep this here.
_os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# Let unsupported ops silently fall back to CPU rather than crash.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
