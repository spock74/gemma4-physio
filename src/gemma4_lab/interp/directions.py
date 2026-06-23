"""Causal-direction toolkit: difference-of-means extraction, directional ablation
and steering, plus the two small statistics used to report results (rank AUC and
Cohen's d). Torch + stdlib only — no sklearn (locked-stack discipline).

Numerical-health convention (repo-wide, see interp/numerical_health.py): every
PURE-MATH helper here REQUIRES finite inputs and raises ValueError otherwise —
NaN/inf from bf16 activations must never silently become a plausible-looking
number. Callers exclude contaminated layers BEFORE calling (the recorder exposes
`last_capture_health`). The intervention HOOKS (`ablating`/`steering`) deliberately
do not guard: they rewrite the model's own live stream mid-forward, and detection
belongs to capture/readout, not to a raise inside generation.

The interventions are PyTorch forward hooks that REWRITE each decoder layer's
output, so they take effect during the forward pass that reads the logits:

    ablating(layers, d):  h <- h - (h . d_hat) d_hat   (remove the component along d)
    steering(layers, d):  h <- h + coeff * d_hat       (push along d)

Device note: under accelerate's MPS+CPU split each layer's `hidden` may be on a
different device, so the hooks move `direction` to `hidden.device`/dtype per call.
The returned tensor stays on the original device so accelerate's own dispatch
hooks remain consistent. Verify an intervention actually bites by checking that
the logits move (a no-op hook ordering would silently leave them unchanged).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from statistics import mean, stdev
from typing import Any

import torch
import torch.nn as nn

# -- numerical-health guard ---------------------------------------------------

def require_finite(name: str, *tensors: torch.Tensor) -> None:
    """Fail-loud guard for pure-math helpers: raise ValueError on any non-finite
    input instead of letting NaN/inf propagate into a plausible-looking result.
    Callers exclude contaminated layers first (see interp.numerical_health)."""
    for t in tensors:
        n_bad = int((~torch.isfinite(t)).sum())
        if n_bad:
            raise ValueError(
                f"{name}: non-finite input ({n_bad}/{t.numel()} elements). "
                "Exclude contaminated layers first — see interp.numerical_health."
            )


def _require_finite_floats(name: str, xs: list[float]) -> None:
    import math
    if any(not math.isfinite(x) for x in xs):
        n_bad = sum(1 for x in xs if not math.isfinite(x))
        raise ValueError(
            f"{name}: non-finite input ({n_bad}/{len(xs)} values). "
            "Exclude contaminated layers/readouts first — see interp.numerical_health."
        )


# -- direction extraction ---------------------------------------------------

def diff_of_means_direction(
    positive: list[torch.Tensor], negative: list[torch.Tensor]
) -> torch.Tensor:
    """Unit-normalized (mean(positive) - mean(negative)). Inputs are [d_model]
    vectors (e.g., last-token residuals at one layer), on CPU. Raises on
    non-finite input (see module docstring)."""
    pos = torch.stack(positive)
    neg = torch.stack(negative)
    require_finite("diff_of_means_direction", pos, neg)
    d = pos.mean(0) - neg.mean(0)
    return d / (d.norm() + 1e-8)


def projection(vec: torch.Tensor, direction: torch.Tensor) -> float:
    """Scalar projection <vec, d_hat>. Both [d_model]. Raises on non-finite input."""
    require_finite("projection", vec, direction)
    return float(vec @ direction)


def _final_norm_weight(model: nn.Module) -> torch.Tensor | None:
    """Locate the final RMSNorm weight (gamma) the readout applies before the LM
    head. Gemma 4: model.model.language_model.norm.weight."""
    for path in (("model", "language_model", "norm"), ("model", "norm"),
                 ("language_model", "norm"), ("norm",)):
        obj: Any = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        w = getattr(obj, "weight", None)
        if isinstance(w, torch.Tensor):
            return w
    return None


def unembedding_direction(model: nn.Module, token_id: int) -> torch.Tensor:
    """Residual-space direction the readout maps onto `token_id`'s logit:
    normalize(gamma ⊙ W_U[token]), where W_U is the (text) LM-head weight row and
    gamma is the final-RMSNorm weight (Gemma uses plain `weight`, multiplied — not
    1+weight). Falls back to normalize(W_U[token]) if gamma is not locatable.
    Gemma ties embeddings and soft-caps logits (monotone — direction unaffected).
    Unit [d_model] on CPU. This is the apparatus-positive control for ablation."""
    head = model.get_output_embeddings()
    w_u = head.weight if head is not None else model.get_input_embeddings().weight
    w = w_u[token_id].detach().float().cpu()
    gamma = _final_norm_weight(model)
    if gamma is not None:
        w = gamma.detach().float().cpu() * w
    else:
        warnings.warn("unembedding_direction: final-norm gamma not found; using raw W_U row",
                      stacklevel=2)
    require_finite("unembedding_direction", w)
    return w / (w.norm() + 1e-8)


# -- interventions ----------------------------------------------------------

def _project_out(hidden: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    coeff = (hidden @ d).unsqueeze(-1)   # [..., seq, 1]
    return hidden - coeff * d


def _apply_at(hidden: torch.Tensor, fn: Any, positions: str | list[int] | None) -> torch.Tensor:
    """Apply `fn` to the residual at `positions` only (sequence dim = -2).
    positions=None → every position (the default global behavior); "last" → only
    the final token; list[int] → those indices (negatives allowed). Non-selected
    positions pass through unchanged."""
    if positions is None:
        return fn(hidden)
    seq = hidden.shape[-2]
    idx = [seq - 1] if positions == "last" else [p if p >= 0 else seq + p for p in positions]
    sel = [slice(None)] * hidden.dim()
    sel[-2] = idx
    sel = tuple(sel)
    out = hidden.clone()
    out[sel] = fn(hidden[sel])
    return out


@contextmanager
def ablating(
    layers: nn.ModuleList, direction: torch.Tensor, positions: str | list[int] | None = None
):
    """Directional ablation across the given layers (simplified single-vector Arditi
    reimpl): removes the component along `direction` from each layer's output for the
    duration of the context. `positions` localizes WHERE in the sequence it acts
    (None=all, "last"=final token, list=indices); slice `layers` to localize the
    depth (e.g. `rec.layers[L:L+1]`)."""
    def make_hook():
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                h = output[0]
                d = direction.to(h.device, h.dtype)
                return (_apply_at(h, lambda x: _project_out(x, d), positions), *output[1:])
            d = direction.to(output.device, output.dtype)
            return _apply_at(output, lambda x: _project_out(x, d), positions)
        return hook

    handles = [layer.register_forward_hook(make_hook()) for layer in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steering(
    layers: nn.ModuleList, direction: torch.Tensor, coeff: float,
    positions: str | list[int] | None = None,
):
    """Add `coeff * direction` to each given layer's output (sufficiency test).
    `positions` localizes where it acts (see `ablating`)."""
    def make_hook():
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                h = output[0]
                d = direction.to(h.device, h.dtype)
                return (_apply_at(h, lambda x: x + coeff * d, positions), *output[1:])
            d = direction.to(output.device, output.dtype)
            return _apply_at(output, lambda x: x + coeff * d, positions)
        return hook

    handles = [layer.register_forward_hook(make_hook()) for layer in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# -- reporting statistics ---------------------------------------------------

def rank_auc(
    scores: list[float], labels: list[str], positive: str = "yes", negative: str = "no"
) -> float | None:
    """P(a random positive scores higher than a random negative). Direction-blind
    callers should use max(auc, 1 - auc). Raises on non-finite scores."""
    _require_finite_floats("rank_auc", scores)
    pos = [s for s, lab in zip(scores, labels, strict=True) if lab == positive]
    neg = [s for s, lab in zip(scores, labels, strict=True) if lab == negative]
    if not pos or not neg:
        return None
    concordant = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (concordant + 0.5 * ties) / (len(pos) * len(neg))


def cohens_d(xs: list[float], ys: list[float]) -> float | None:
    _require_finite_floats("cohens_d", xs + ys)
    if len(xs) < 2 or len(ys) < 2:
        return None
    sx, sy = stdev(xs), stdev(ys)
    pooled = (((len(xs) - 1) * sx**2 + (len(ys) - 1) * sy**2)
              / (len(xs) + len(ys) - 2)) ** 0.5
    if pooled == 0:
        return None
    return (mean(xs) - mean(ys)) / pooled
