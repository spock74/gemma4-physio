"""Activation capture and logit readout for Gemma 4, built on `GemmaLocal`.

Reuses the loaded model/tokenizer from `inference.hf_local.GemmaLocal` so the
interp track shares one warm model with the rest of the lab. Two readouts:

    last_token_residuals(text, layers)  -> {layer: Tensor[d_model]}  (cpu, float32)
    next_token_logits(text)             -> Tensor[vocab]             (cpu, float32)

Device note: on M2 the model is split across MPS+CPU by accelerate, so different
decoder layers live on different devices. Capture hooks therefore detach+move to
CPU; intervention hooks (directions.py) must move their direction to each layer's
own `hidden.device`. Inputs go to `GemmaLocal._device` (the embedding device);
accelerate dispatches the rest.

Numerical health (bf16 Gemma 4 can emit NaN/inf activations): every capture hook
checks `torch.isfinite` on the FULL hidden tensor and every logits readout checks
the returned logits. Contamination is DETECTED here and exposed loudly
(`last_capture_health` / `last_logits_health` + Logfire warning + console) — the
RECOVERY decision (exclude layer / withhold verdict) belongs to the caller and is
never silently applied in the hook. See interp/numerical_health.py for the policy.

Every capture is wrapped in a Logfire span (project rule: observability first).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..inference.hf_local import GemmaLocal


def tensor_health(t: torch.Tensor) -> dict | None:
    """Health record for a tensor: None if fully finite, else counts + fraction.
    Pure helper (unit-tested); used by capture hooks, readouts, and the audit."""
    n_bad = int((~torch.isfinite(t)).sum())
    if n_bad == 0:
        return None
    return {"n_nonfinite": n_bad, "n_elements": t.numel(), "frac": n_bad / t.numel()}


def contaminated_layers_from_residuals(
    per_prompt: list[dict[int, torch.Tensor]],
) -> dict[int, int]:
    """{layer: n_prompts_affected} over captured residuals (vectors [d] or all-position
    [seq, d]) that contain any non-finite value. The runners use this to EXCLUDE layers
    from sweeps and to WITHHOLD verdicts — explicit recovery, never silent."""
    out: dict[int, int] = {}
    for res in per_prompt:
        for layer, t in res.items():
            if not torch.isfinite(t).all():
                out[layer] = out.get(layer, 0) + 1
    return dict(sorted(out.items()))


def resolve_text_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder-layer ModuleList regardless of wrapper class.

    Gemma 4 loads as `AutoModelForImageTextToText`
    (`Gemma4ForConditionalGeneration`); its text layers sit at
    `model.model.language_model.layers`. Other layouts are probed as fallbacks;
    on miss, the error prints the top-level child names so adding a path is a
    one-line edit.
    """
    candidate_paths = [
        ("model", "language_model", "layers"),   # Gemma 4 multimodal wrapper
        ("model", "layers"),                     # text-only causal LM
        ("language_model", "model", "layers"),   # older multimodal wrappers
        ("language_model", "layers"),
        ("transformer", "h"),                    # GPT-2 family
    ]
    for path in candidate_paths:
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    top = [name for name, _ in model.named_children()]
    raise RuntimeError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        f"top-level children = {top}. Extend candidate_paths in "
        "interp/recorder.py:resolve_text_layers."
    )


class ActivationRecorder:
    """Captures residual-stream activations and next-token logits via forward hooks."""

    def __init__(self, gemma: GemmaLocal) -> None:
        self.gemma = gemma
        self._layers: nn.ModuleList | None = None
        # Numerical health of the MOST RECENT capture / logits readout. Detection
        # only — callers decide recovery (exclude layers, withhold verdicts).
        self.last_capture_health: dict[int, dict] = {}
        self.last_logits_health: dict | None = None
        self._warned_layers: set[int] = set()

    # -- model access -------------------------------------------------------

    def _ensure_loaded(self) -> None:
        self.gemma._load()
        if self._layers is None:
            assert self.gemma._model is not None
            self._layers = resolve_text_layers(self.gemma._model)

    @property
    def layers(self) -> nn.ModuleList:
        self._ensure_loaded()
        assert self._layers is not None
        return self._layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def tokenizer(self) -> Any:
        self._ensure_loaded()
        return self.gemma._tokenizer

    @property
    def model(self) -> Any:
        self._ensure_loaded()
        return self.gemma._model

    # -- encoding -----------------------------------------------------------

    def encode(
        self, text: str, templated: bool = False, assistant_prefix: str | None = None
    ) -> dict[str, Any]:
        """Tokenize `text`. If `templated`, wrap as a single user turn with the
        chat template (thinking disabled); else feed the raw string.

        `assistant_prefix` PRE-FILLS the model turn (requires `templated`): the
        next-token prediction is then the immediate continuation of the prefix.
        This is the readout that elicits factual recall from the -it model — fed a
        raw cloze it echoes the context, but instructed in the user turn with the
        cloze stem prefilling its own turn it completes the fact as token 0.
        Inputs land on the model's input device; accelerate moves them across the split.
        """
        tok = self.tokenizer
        if templated:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if assistant_prefix:
                prompt = prompt + assistant_prefix
        else:
            prompt = text + (assistant_prefix or "")
        return tok(prompt, return_tensors="pt").to(self.gemma._device)

    # -- readouts -----------------------------------------------------------

    def _capture_hook(self, layer_idx: int, sink: dict[int, torch.Tensor], health: dict[int, dict]):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            bad = tensor_health(hidden)  # FULL tensor, not just the captured slice
            if bad is not None:
                health[layer_idx] = bad
            sink[layer_idx] = hidden[:, -1, :].detach().float().squeeze(0).cpu()
        return hook

    def _finish_capture(self, health: dict[int, dict]) -> None:
        """Expose + loudly log capture contamination. Detection only — recovery
        (exclude layer / withhold verdict) is the caller's decision; nothing is
        zeroed or masked here."""
        import logfire

        self.last_capture_health = health
        if not health:
            return
        layers = sorted(health)
        logfire.warning(
            "capture_nonfinite_activations",
            layers=layers,
            max_frac=max(v["frac"] for v in health.values()),
        )
        if set(layers) != self._warned_layers:  # console once per distinct layer set
            print(f"  [numerical-health] NON-FINITE activations at layers {layers} "
                  f"(max frac {max(v['frac'] for v in health.values()):.2e}) — "
                  "callers must exclude/withhold (see interp.numerical_health).")
            self._warned_layers = set(layers)

    def last_token_residuals(
        self,
        text: str,
        layers: list[int],
        templated: bool = False,
        assistant_prefix: str | None = None,
    ) -> dict[int, torch.Tensor]:
        """Residual stream at the last input token for each requested layer,
        captured in a SINGLE forward pass. Returns {layer: Tensor[d_model]} on CPU.
        With `assistant_prefix` the last token is the end of the prefix (where
        entity-knowledge gating would act, just before the answer is emitted)."""
        import logfire

        self._ensure_loaded()
        sink: dict[int, torch.Tensor] = {}
        health: dict[int, dict] = {}
        handles = [
            self.layers[i].register_forward_hook(self._capture_hook(i, sink, health))
            for i in layers
        ]
        try:
            with logfire.span(
                "interp.capture", n_layers=len(layers), templated=templated
            ):
                inputs = self.encode(text, templated, assistant_prefix)
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            for h in handles:
                h.remove()
        self._finish_capture(health)
        return sink

    def all_token_residuals(
        self,
        text: str,
        layers: list[int],
        templated: bool = False,
        assistant_prefix: str | None = None,
    ) -> dict[int, torch.Tensor]:
        """Residual stream at EVERY token position for each requested layer (one
        forward pass). Returns {layer: Tensor[seq, d_model]} on CPU. Used by E2 to
        reach n_repr >> d for CKA; since E2B/E4B share the tokenizer, rows align by
        (prompt, position) across variants."""
        import logfire

        self._ensure_loaded()
        sink: dict[int, torch.Tensor] = {}
        health: dict[int, dict] = {}

        def make_hook(idx: int):
            def hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                bad = tensor_health(hidden)
                if bad is not None:
                    health[idx] = bad
                sink[idx] = hidden[0].detach().float().cpu()  # [seq, d_model]
            return hook

        handles = [self.layers[i].register_forward_hook(make_hook(i)) for i in layers]
        try:
            with logfire.span("interp.capture_all", n_layers=len(layers), templated=templated):
                inputs = self.encode(text, templated, assistant_prefix)
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            for h in handles:
                h.remove()
        self._finish_capture(health)
        return sink

    def next_token_logits(
        self, text: str, templated: bool = False, assistant_prefix: str | None = None
    ) -> torch.Tensor:
        """Logits over the vocabulary at the final position. Forward pass only —
        no generation. Returns Tensor[vocab] on CPU (float32).

        Call inside `ablating(...)` / `steering(...)` to read the causal effect of
        an intervention without paying the ~0.4 tok/s generation cost.
        """
        import logfire

        self._ensure_loaded()
        with logfire.span("interp.logits", templated=templated):
            inputs = self.encode(text, templated, assistant_prefix)
            with torch.no_grad():
                out = self.model(**inputs)
            logits = out.logits[0, -1, :].detach().float().cpu()
        self.last_logits_health = tensor_health(logits)
        if self.last_logits_health is not None:  # readout itself is contaminated — always loud
            logfire.warning("readout_nonfinite_logits", **self.last_logits_health)
            print(f"  [numerical-health] NON-FINITE next-token logits "
                  f"({self.last_logits_health['n_nonfinite']}/{self.last_logits_health['n_elements']}) — "
                  "any verdict on this readout must be withheld.")
        return logits
