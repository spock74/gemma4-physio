"""O2 / capture (CORE conda env, torch 2.11.0) — run gemma-3-270m-it on each
frozen Neuronpedia snippet and capture resid-post at layer 12.

Neuronpedia/Gemma Scope protocol for THIS source: the frozen snippet tokens
ALREADY start with `<bos>` (the prepended BOS is shown) and are chat-formatted
corpus text, so we feed the converted ids DIRECTLY — no extra BOS, and positions
align 1:1 with Neuronpedia's `values` (no row dropping at encode time).

Token strings need a fix-up: Neuronpedia renders the sentencepiece marker `'▁'`
as a plain space `' '`, so e.g. `' an'` must map to vocab token `'▁an'` before
convert_tokens_to_ids — otherwise ~half the tokens fall to UNK.

Captures the forward-hook OUTPUT of model.model.layers[12] == resid-post L12 ==
Gemma Scope hook `blocks.12.hook_resid_post`. Saves [seq, 640] residuals to
captures/ for the sae-venv encode stage.

Run (CORE env): python calibration/gemmascope2_fidelity/o2_capture.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-270m-it"
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
LAYER = 12
EXPECT_D = 640
HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
CAP = HERE / "captures"


def main() -> int:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN absent — gemma-3-270m-it is gated")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"))
    UNK = tok.unk_token_id

    def to_id(t: str) -> int:
        """Map a Neuronpedia display token to a vocab id. Neuronpedia renders the
        sentencepiece '▁' as a plain space, so try the raw string, then '▁'+rest
        for a leading space, then a global ' '->'▁' swap."""
        cands = [t]
        if t.startswith(" "):
            cands.append("▁" + t[1:])
        cands.append(t.replace(" ", "▁"))
        for c in cands:
            i = tok.convert_tokens_to_ids(c)
            if i is not None and i != UNK:
                return i
        return UNK

    # float32: the SAE/Neuronpedia pipeline encodes float32 activations, and L12
    # resid norms are ~5e3 — bf16's coarse resolution there scrambles the small
    # (non-peak) activations and tanks Pearson. 270m in fp32 is ~1 GB, trivial.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
        dtype=torch.float32, attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    layers = model.model.layers
    assert len(layers) == 18, f"expected 18 layers, got {len(layers)}"
    print(f"loaded {MODEL_ID}; capturing resid-post of layers[{LAYER}] "
          f"(= blocks.{LAYER}.hook_resid_post)", flush=True)

    CAP.mkdir(exist_ok=True)
    for ref_path in sorted(REF.glob("np_feature_*.json")):
        ref = json.loads(ref_path.read_text())
        idx = int(ref["feature_index"])
        snip = ref["activations"][0]
        tok_strs = snip["tokens"]
        np_values = np.asarray(snip["values"], dtype=np.float32)
        max_idx = int(snip["maxValueTokenIndex"])

        ids = [to_id(t) for t in tok_strs]
        n_unk = sum(1 for i in ids if i == UNK)
        if n_unk:
            print(f"  feature {idx}: {n_unk} UNK after robust token->id — SKIP (tokenization mismatch)")
            continue
        # snippet already starts with <bos>; feed ids directly (no extra BOS)
        assert ids[0] == tok.bos_token_id, f"feature {idx}: tokens[0] is not <bos>"
        input_ids = torch.tensor([ids], device=DEVICE)

        captured: dict[str, torch.Tensor] = {}

        def hook(_m, _i, output, _c=captured):
            h = output[0] if isinstance(output, tuple) else output
            _c["resid"] = h[0].detach().float().cpu()  # [seq, d]

        handle = layers[LAYER].register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        finally:
            handle.remove()

        resid = captured["resid"]
        assert resid.shape[-1] == EXPECT_D, f"d_model {resid.shape[-1]} != {EXPECT_D}"
        n_bad = int((~torch.isfinite(resid)).sum())
        if n_bad:
            print(f"  feature {idx}: {n_bad} non-finite resid entries — flagged")

        assert resid.shape[0] == len(tok_strs), \
            f"feature {idx}: seq {resid.shape[0]} != n_values {len(tok_strs)}"
        out = CAP / f"feature_{idx}.npz"
        np.savez(out,
                 resid=resid.numpy().astype(np.float32),  # aligned 1:1 with np_values (bos at row 0)
                 np_values=np_values, max_idx=np.int64(max_idx),
                 feature_index=np.int64(idx), n_content_tokens=np.int64(len(tok_strs)))
        print(f"  feature {idx}: resid {tuple(resid.shape)} (aligned, bos@0), "
              f"NP maxValue {snip['maxValue']:.3f} @ {max_idx} -> {out.name}", flush=True)

    print("\ncapture done — encode next in the sae venv:")
    print("  calibration/.venv-sae/bin/python calibration/gemmascope2_fidelity/o2_encode.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
