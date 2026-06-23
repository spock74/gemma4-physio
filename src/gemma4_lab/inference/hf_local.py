"""Local Gemma 4 inference via HuggingFace Transformers (bf16, MPS).

Multi-turn rule (per Gemma 4 docs): when re-prompting in a multi-turn conversation,
the historical assistant turns must contain ONLY the visible answer (`result.text`),
never the thought block (`result.thought`). Strip thoughts before re-feeding history.

Secrets: HF_TOKEN is imported from `gemma4_lab.config` (where it is read once
from the OS environment). It is never read directly here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..config import Settings, hf_token_or_none


@dataclass
class GenerationResult:
    text: str
    """The visible answer (no special tokens, no thinking trace)."""

    thought: str | None
    """Internal reasoning, if `thinking=True` and the model emitted a thought channel."""

    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str


def _split_thought_and_answer(decoded: str) -> tuple[str | None, str]:
    """Parse Gemma 4's thinking-mode output.

    When `enable_thinking=True`, the decoded text contains:
        <|channel|>thought\n<reasoning><|channel|><answer>

    For E2B/E4B with `enable_thinking=False`, the channel tags are NOT emitted,
    so the entire decoded string is the answer.
    """
    open_tag = "<|channel|>thought"
    close_tag = "<|channel|>"
    if open_tag not in decoded:
        return None, decoded.strip()

    after_open = decoded.split(open_tag, 1)[1].lstrip("\n")
    if close_tag not in after_open:
        return None, decoded.strip()
    thought, answer = after_open.split(close_tag, 1)
    return thought.strip() or None, answer.strip()


class GemmaLocal:
    """Single-process Gemma 4 runner. Loads lazily on first `generate()` call."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._dtype: Any = None
        self._device: str = settings.device

    def _load(self) -> None:
        if self._model is not None:
            return
        import logfire
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        with logfire.span("gemma_local.load", model_id=self.model_id, device=self._device):
            self._dtype = getattr(torch, self.settings.model_dtype)

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=hf_token_or_none(),
                cache_dir=str(self.settings.hf_cache_dir),
            )
            # MatFormer note: gemma-4-E2B-it ships ~9.5 GB of weights (the full
            # E4B set; E2B is a runtime sub-extraction). Apple Silicon's per-
            # buffer Metal cap (~7 GiB on M2 base) rejects a single 9.5 GB
            # allocation, so we offload across MPS+CPU via accelerate.
            # `device_map="auto"` + `max_memory` is required on 16 GB Macs.
            # Caveat: most layers end up on CPU, so inference is slow. Phase 2
            # GGUF/MLX backends will fix this with quantization.
            # transformers v5 renamed `torch_dtype` -> `dtype`. Old name causes
            # a silent fp32 cast which makes the buffer issue twice as bad.
            from_pretrained_kwargs: dict[str, Any] = dict(
                dtype=self._dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                token=hf_token_or_none(),
                cache_dir=str(self.settings.hf_cache_dir),
            )
            if self._device == "mps":
                from_pretrained_kwargs["device_map"] = "auto"
                from_pretrained_kwargs["max_memory"] = {"mps": "6GiB", "cpu": "14GiB"}
            else:
                from_pretrained_kwargs["device_map"] = self._device

            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **from_pretrained_kwargs,
            )
            self._model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        thinking: bool = False,
        max_new_tokens: int = 512,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Run a single generation.

        Args:
            messages: Chat messages in the standard `[{"role": "user", "content": "..."}]` shape.
            thinking: If True, enable Gemma 4's thinking-mode channel.
            max_new_tokens: Hard cap on generated tokens.
            **gen_kwargs: Forwarded to `model.generate()` (e.g., `temperature`, `top_p`).
        """
        import logfire

        self._load()
        assert self._model is not None and self._tokenizer is not None

        with logfire.span(
            "gemma_local.generate",
            model_id=self.model_id,
            thinking=thinking,
            max_new_tokens=max_new_tokens,
        ) as span:
            prompt: str = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            input_token_count = int(inputs["input_ids"].shape[1])

            t0 = time.perf_counter()
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=gen_kwargs.pop("do_sample", False),
                **gen_kwargs,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            new_tokens = output_ids[0][input_token_count:]
            output_token_count = int(new_tokens.shape[0])

            decoded_with_specials = self._tokenizer.decode(new_tokens, skip_special_tokens=False)
            thought, _answer_with_specials = _split_thought_and_answer(decoded_with_specials)
            # Re-decode without specials for clean visible text. The channel
            # markers are special tokens and are removed by skip_special_tokens.
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            span.set_attribute("input_tokens", input_token_count)
            span.set_attribute("output_tokens", output_token_count)
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("has_thought", thought is not None)

            logfire.info(
                "generation_complete",
                model_id=self.model_id,
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                latency_ms=latency_ms,
                has_thought=thought is not None,
            )

            return GenerationResult(
                text=text,
                thought=thought,
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                latency_ms=latency_ms,
                model_id=self.model_id,
            )
