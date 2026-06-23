"""Per-layer numerical-health audit for bf16 Gemma 4 activations.

Why this exists: Gemma 4 in bf16 is reported (community projects) to emit NON-FINITE
activations (NaN/inf) in many layers (20+/42 reported on E4B). Our pipeline captures
bf16 residuals from an accelerate MPS+CPU split and feeds them to diff-of-means /
CKA / probes per layer — NaN propagates silently and surfaces as a plausible-looking
number. This module measures the contamination directly, on the EXACT readout the
experiments use (assistant-prefill recall), so it decides whether E1/E1c/E2 numbers
are trustworthy at all.

Policy (mandatory, repo-wide): NO silent fallback. DETECT explicitly -> LOG loudly
(Logfire span attrs + console, with counts and WHICH layers) -> RECOVER transparently
(exclude/withhold, reported) -> RECORD a `numerical_health` block in every result
JSON. Never mask; always surface.

`winsorize_residuals` is provided as a DOCUMENTED ALTERNATIVE recovery path and is
OFF by default everywhere: winsorizing changes the geometry being measured, so it may
only be used behind an explicit flag and must be recorded in the output.

Run:  python -m gemma4_lab.interp.numerical_health [e2b|e4b]
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import torch

from ..config import Settings
from .entity_knowledge import RECALL_INSTRUCTION
from .recorder import ActivationRecorder, tensor_health

__all__ = ["tensor_health", "winsorize_residuals", "audit"]


def winsorize_residuals(x: torch.Tensor, q: float = 0.999) -> torch.Tensor:
    """ALTERNATIVE recovery path, OFF by default: replace non-finite entries with 0
    and clamp magnitudes to the q-quantile of the finite values. WARNING: this
    CHANGES the geometry being measured — use only behind an explicit flag, and
    record its use in the result JSON. Provided so the recovery option is documented
    rather than improvised."""
    finite = torch.isfinite(x)
    if finite.all():
        return x
    safe = torch.where(finite, x, torch.zeros_like(x))
    bound = torch.quantile(safe.abs().flatten().float(), q).item()
    return safe.clamp(-bound, bound)


def audit(
    variant: str = "e2b",
    corpus_path: str | Path | None = None,
    e1_layer: int = 26,
    max_prompts: int | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Run the E1 corpus through `variant` with the assistant-prefill readout,
    capture the residual at EVERY layer and EVERY position, and report per layer:
    fraction of non-finite elements, finite-norm range, and usability. Loud verdict
    on the E1 extraction layer."""
    import gc

    import logfire

    from .. import observability
    from ..inference.hf_local import GemmaLocal

    observability.setup()
    settings = Settings(model_variant=variant)  # type: ignore[arg-type]
    rec = ActivationRecorder(GemmaLocal(settings))
    corpus_path = Path(corpus_path or (settings.data_dir / "eval" / "entity_knowledge_contrast.json"))
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    prompts = [it["prompt"] for it in corpus["known"]] + [it["prompt"] for it in corpus["unknown"]]
    # E4B (16 GB) does not fit 16 GB RAM -> disk-offload; without per-prompt cache
    # release, MPS allocations accumulate and OOM-kill the process (~prompt 10). Cap
    # the prompt count for big variants (a structural per-layer NaN shows on any
    # prompt) and free the MPS cache every prompt.
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    n_layers = rec.n_layers
    layers = list(range(n_layers))
    stats = {
        L: {"n_nonfinite": 0, "n_elements": 0, "n_prompts_affected": 0,
            "n_last_pos_affected": 0, "max_finite_norm": 0.0, "min_finite_norm": float("inf")}
        for L in layers
    }

    with logfire.span("interp.numerical_health.audit", variant=variant, n_prompts=len(prompts)):
        for i, p in enumerate(prompts):
            if i % 10 == 0:
                print(f"  [audit] prompt {i}/{len(prompts)}", flush=True)
            res = rec.all_token_residuals(RECALL_INSTRUCTION, layers, True, p)
            for L in layers:
                h = res[L]                       # [seq, d_model] float32 (NaN/inf preserved)
                s = stats[L]
                bad = ~torch.isfinite(h)
                nb = int(bad.sum())
                s["n_nonfinite"] += nb
                s["n_elements"] += h.numel()
                if nb:
                    s["n_prompts_affected"] += 1
                if not torch.isfinite(h[-1]).all():
                    s["n_last_pos_affected"] += 1   # the position E1/E1c/E2 decodability read
                norms = torch.linalg.vector_norm(h, dim=-1)
                fin = norms[torch.isfinite(norms)]
                if fin.numel():
                    s["max_finite_norm"] = max(s["max_finite_norm"], float(fin.max()))
                    s["min_finite_norm"] = min(s["min_finite_norm"], float(fin.min()))
            del res, h            # release per-prompt residuals before the next forward
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        per_layer = []
        contaminated = []
        for L in layers:
            s = stats[L]
            frac = s["n_nonfinite"] / s["n_elements"] if s["n_elements"] else 0.0
            usable = s["n_nonfinite"] == 0
            if not usable:
                contaminated.append(L)
            per_layer.append({
                "layer": L,
                "frac_nonfinite": round(frac, 8),
                "n_nonfinite": s["n_nonfinite"],
                "n_prompts_affected": s["n_prompts_affected"],
                "n_last_pos_affected": s["n_last_pos_affected"],
                "max_finite_norm": round(s["max_finite_norm"], 2),
                "min_finite_norm": round(s["min_finite_norm"], 2) if s["min_finite_norm"] != float("inf") else None,
                "usable": usable,
            })
        logfire.info(
            "numerical_health_verdict",
            variant=variant,
            n_layers_contaminated=len(contaminated),
            contaminated_layers=contaminated,
        )
        e1_layer_contaminated = e1_layer in contaminated
        result = {
            "experiment": "numerical_health",
            "model_variant": variant,
            "model_id": settings.model_id,
            "readout": "assistant_prefill",
            "n_prompts": len(prompts),
            "prompts_capped": max_prompts is not None,
            "n_layers": n_layers,
            "n_layers_contaminated": len(contaminated),
            "contaminated_layers": contaminated,
            "e1_extraction_layer": e1_layer,
            "e1_layer_contaminated": e1_layer_contaminated,
            "per_layer": per_layer,
        }
        out_dir = Path(out_dir or (settings.data_dir / "eval" / "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"numerical_health_{variant}_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 76)
    print(f"NUMERICAL HEALTH — {settings.model_id} (bf16, MPS+CPU split), "
          f"{len(prompts)} prompts, all positions")
    print("=" * 76)
    print(f"{'layer':>5} {'frac_nonfinite':>15} {'prompts_hit':>12} {'lastpos_hit':>12} "
          f"{'max|h|':>10} {'min|h|':>10}  status")
    for r in per_layer:
        print(f"{r['layer']:>5} {r['frac_nonfinite']:>15.2e} {r['n_prompts_affected']:>12} "
              f"{r['n_last_pos_affected']:>12} {r['max_finite_norm']:>10.1f} "
              f"{str(r['min_finite_norm']):>10}  {'OK' if r['usable'] else '** CONTAMINATED **'}")
    print("-" * 76)
    if contaminated:
        print(f"  !! {len(contaminated)}/{n_layers} layers contaminated: {contaminated}")
    else:
        print(f"  ALL {n_layers} layers fully finite — capture pipeline is numerically clean.")
    if e1_layer_contaminated:
        print(f"  !! E1 EXTRACTION LAYER {e1_layer} IS CONTAMINATED — the phase6-report E1/E1c")
        print("     numbers built on this layer are INVALID and must be re-run with the guard.")
    else:
        print(f"  E1 extraction layer {e1_layer}: clean.")
    print(f"  wrote {out_path}")
    print("=" * 76)
    return result


if __name__ == "__main__":
    _variant = sys.argv[1] if len(sys.argv) > 1 else "e2b"
    # E4B (16 GB) OOMs this 16 GB host under disk-offload; cap prompts (a structural
    # per-layer NaN shows on any prompt) unless an explicit count is given as argv[2].
    _cap = int(sys.argv[2]) if len(sys.argv) > 2 else (16 if _variant == "e4b" else None)
    audit(variant=_variant, max_prompts=_cap)
