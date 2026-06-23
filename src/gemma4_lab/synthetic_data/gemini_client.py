"""Minimal Gemini wrapper over the new google-genai SDK.

Secrets: GEMINI_API_KEY is imported from `gemma4_lab.config` (where it is read
once from the OS environment). The key is never read here directly, never
accepted as a constructor argument, and never logged.
"""

from __future__ import annotations

from ..config import require_gemini_key


class GeminiClient:
    def __init__(self, default_model: str = "gemini-2.5-pro") -> None:
        # `google` is a PEP 420 namespace package shared with google-auth,
        # google-cloud, etc. Use the absolute submodule import so pyright
        # resolves it correctly.
        import google.genai as genai

        self.default_model = default_model
        self._client = genai.Client(api_key=require_gemini_key())

    def generate(self, prompt: str, model: str | None = None) -> str:
        """One-shot text generation. Returns the response text."""
        import logfire

        model_name = model or self.default_model
        with logfire.span("gemini.generate", model=model_name) as span:
            resp = self._client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = resp.text or ""
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                for attr in ("prompt_token_count", "candidates_token_count", "total_token_count"):
                    val = getattr(usage, attr, None)
                    if val is not None:
                        span.set_attribute(attr, val)
            span.set_attribute("output_chars", len(text))
            return text
