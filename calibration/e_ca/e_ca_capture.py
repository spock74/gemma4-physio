"""E-CA / capture (CORE conda env, torch 2.11.0) — run gemma-3-270m-it on each
entity-knowledge prompt and capture resid-post at ALL 18 layers in one forward
pass.

The entity-knowledge corpus has "known" (factual cloze completions) and
"unknown" (fictional / impossible referents) prompts.  We feed the raw prompt
text — no chat template — because the Gemma Scope 2 SAE was trained on Pile
(raw text is on-distribution for the SAE encoder; chat markup would shift the
activation distribution).

For every prompt we register a forward hook on each of the 18 decoder layers
and save the full [n_layers, seq_len, d_model] residual tensor as a float32
.npz for the encode stage.

Run (CORE env):
    python calibration/e_ca/e_ca_capture.py [--n-prompts N]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent  # calibration/e_ca -> calibration -> project root
CORPUS = PROJECT_ROOT / "data" / "eval" / "entity_knowledge_contrast.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="E-CA capture: residuals at all layers")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="number of prompts per label to process (default: all)")
    parser.add_argument("--model-size", type=str, choices=["270m", "4b"], default="270m",
                        help="model size to run (270m or 4b)")
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN absent — Gemma 3 is gated")

    # ── set model variables dynamically ──────────────────────────────
    if args.model_size == "270m":
        model_id = "google/gemma-3-270m-it"
        expect_n_layers = 18
        expect_d_model = 640
    else:
        model_id = "google/gemma-3-4b-it"
        expect_n_layers = 34
        expect_d_model = 2560

    cap_dir = HERE / "captures" / args.model_size

    # ── load corpus ──────────────────────────────────────────────────────
    if not CORPUS.exists():
        raise FileNotFoundError(f"corpus not found: {CORPUS}")
    corpus = json.loads(CORPUS.read_text())
    known = corpus["known"]
    unknown = corpus["unknown"]
    if args.n_prompts is not None:
        known = known[: args.n_prompts]
        unknown = unknown[: args.n_prompts]
    prompts: list[dict] = []
    for i, item in enumerate(known):
        prompts.append({"prompt": item["prompt"], "label": "known", "idx": i,
                         "prompt_id": f"known_{i:03d}"})
    for i, item in enumerate(unknown):
        prompts.append({"prompt": item["prompt"], "label": "unknown", "idx": i,
                         "prompt_id": f"unknown_{i:03d}"})
    print(f"corpus: {len(known)} known + {len(unknown)} unknown = {len(prompts)} prompts",
          flush=True)

    # ── load model ───────────────────────────────────────────────────────
    dtype = torch.float32 if args.model_size == "270m" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(
        model_id, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
        dtype=dtype, attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    elif hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise AttributeError("Cannot find language model layers in model")

    n_layers = len(layers)
    assert n_layers == expect_n_layers, f"expected {expect_n_layers} layers, got {n_layers}"

    # sanity-check d_model from the config
    if hasattr(model.config, "text_config"):
        d_model = model.config.text_config.hidden_size
    else:
        d_model = model.config.hidden_size
    assert d_model == expect_d_model, f"expected d_model {expect_d_model}, got {d_model}"
    print(f"loaded {model_id}: {n_layers} layers, d_model={d_model}, device={DEVICE}",
          flush=True)

    # ── capture loop ─────────────────────────────────────────────────────
    cap_dir.mkdir(parents=True, exist_ok=True)
    total_bad = 0
    for pi, pinfo in enumerate(prompts):
        prompt_text = pinfo["prompt"]
        label = pinfo["label"]
        idx = pinfo["idx"]
        prompt_id = pinfo["prompt_id"]

        # tokenize raw prompt (no chat template)
        enc = tok(prompt_text, return_tensors="pt").to(DEVICE)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        seq_len = int(input_ids.shape[1])

        # register hooks on all layers
        captured: dict[int, torch.Tensor] = {}
        handles = []

        for layer_i in range(n_layers):
            def hook(_m, _i, output, _layer=layer_i, _cap=captured):
                h = output[0] if isinstance(output, tuple) else output
                _cap[_layer] = h[0].detach().float().cpu()  # [seq, d_model]
            handles.append(layers[layer_i].register_forward_hook(hook))

        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        # stack into [n_layers, seq_len, d_model]
        assert len(captured) == n_layers, \
            f"captured {len(captured)} layers, expected {n_layers}"
        residuals = torch.stack([captured[i] for i in range(n_layers)], dim=0)
        assert residuals.shape == (n_layers, seq_len, d_model), \
            f"shape {tuple(residuals.shape)} != ({n_layers}, {seq_len}, {d_model})"

        # numerical health check
        n_bad = int((~torch.isfinite(residuals)).sum())
        if n_bad:
            print(f"  [{pi+1}/{len(prompts)}] {prompt_id}: "
                  f"WARNING {n_bad} non-finite entries", flush=True)
            total_bad += n_bad

        # save
        out = cap_dir / f"{label}_{idx:03d}.npz"
        np.savez(
            out,
            residuals=residuals.numpy().astype(np.float32),
            prompt_id=str(prompt_id),
            label=str(label),
            n_layers=np.int64(n_layers),
            seq_len=np.int64(seq_len),
        )
        print(f"  [{pi+1}/{len(prompts)}] {prompt_id}: "
              f"seq_len={seq_len} -> {out.name}", flush=True)

    # ── summary ──────────────────────────────────────────────────────────
    n_files = len(list(cap_dir.glob("*.npz")))
    print(f"\ncapture done: {n_files} .npz files in {cap_dir}")
    if total_bad:
        print(f"  WARNING: {total_bad} total non-finite entries across all prompts")
    else:
        print("  all residuals finite ✓")
    print(f"\nnext step (SAE venv):")
    print(f"  calibration/.venv-sae/bin/python calibration/e_ca/e_ca_encode.py")
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
