from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
import logfire
from typing import Any

from ..config.pipeline_schema import ModelConfig
from ..config import hf_token_or_none

def load_model_and_tokenizer(config: ModelConfig) -> tuple[torch.nn.Module, Any]:
    """
    Factory to load a HuggingFace model and tokenizer according to declarative config.
    Safely handles MPS + bfloat16 requirements.
    """
    with logfire.span("factory.load", model_id=config.model_id, device=config.device):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            token=hf_token_or_none(),
            cache_dir=str(config.cache_dir),
        )
        
        kwargs: dict[str, Any] = dict(
            dtype=config.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=config.attn_implementation,
            token=hf_token_or_none(),
            cache_dir=str(config.cache_dir),
        )
        
        if config.device == "mps":
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {"mps": "6GiB", "cpu": "14GiB"}
        else:
            kwargs["device_map"] = config.device
            
        try:
            # Gemma 3/4 defaults to multimodal AutoModelForImageTextToText 
            # or causal LM depending on the exact variant. We try conditional generation first.
            if "gemma-4" in config.model_id.lower():
                model = AutoModelForImageTextToText.from_pretrained(config.model_id, **kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
        except Exception:
            # Fallback for standard architectures
            model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
            
        model.eval()
        return model, tokenizer
