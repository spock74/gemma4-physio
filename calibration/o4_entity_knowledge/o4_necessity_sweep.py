"""O4.3b — PER-LAYER necessity sweep (addresses the layer-selection / forking-paths
critique). For EVERY layer L: fit d_know on TRAIN at L (held-out), then localized
ablation at L (recall position) on VAL known items, vs an orthogonal matched control.

Reports the per-layer profile of d_know Δlog-prob(gold) and the specificity ratio
d_know/orth, so "L11 is special" (4b) becomes a measured claim, not a post-hoc pick.
Run on BOTH models so model-specificity (4b yes / 270m no) is tested at ANALOGOUS
positions, not the late-vs-mid mismatch of the first pass.

Run (CORE env):
  python calibration/o4_entity_knowledge/o4_necessity_sweep.py 270m
  python calibration/o4_entity_knowledge/o4_necessity_sweep.py 4b
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from gemma4_lab.interp.directions import ablating, diff_of_means_direction
from gemma4_lab.interp.entity_knowledge import RECALL_INSTRUCTION

CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
HERE = Path(__file__).resolve().parent
CORPUS = Path("data/eval/entity_knowledge_contrast.json")
MODELS = {"270m": "google/gemma-3-270m-it", "4b": "google/gemma-3-4b-it"}


def locate_layers(model):
    for path in (("model", "layers"), ("model", "language_model", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("decoder layers not found")


def load(tag):
    mid = MODELS[tag]
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(f"HF_TOKEN absent — {mid} is gated")
    tok = AutoTokenizer.from_pretrained(mid, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"))
    cls = AutoModelForCausalLM if tag == "270m" else AutoModelForImageTextToText
    dtype = torch.float32 if tag == "270m" else torch.bfloat16
    model = cls.from_pretrained(mid, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
                                dtype=dtype, attn_implementation="eager").to(DEVICE)
    model.eval()
    return tok, model, mid


def recall_inputs(tok, stem):
    p = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                tokenize=False, add_generation_prompt=True) + stem
    return tok(p, return_tensors="pt").to(DEVICE)


def all_layer_resid(model, layers, inputs):
    sink = {}

    def mk(i):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            sink[i] = h[0, -1, :].detach().float().cpu()
        return hook
    hs = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hs:
            h.remove()
    return sink


def logits_of(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def split(n, seed, vf=0.5):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    nv = max(1, round(n * vf))
    val = set(perm[:nv])
    return [i for i in range(n) if i not in val], sorted(val)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "4b"
    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    tok, model, mid = load(tag)
    layers = locate_layers(model)
    nL = len(layers)
    print(f"loaded {mid}: {nL} layers; per-layer necessity sweep (held-out d_know per layer)\n", flush=True)

    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)

    # fit d_know at every layer from TRAIN (one capture pass per train item)
    k_tr_res = [all_layer_resid(model, layers, recall_inputs(tok, known[i]["prompt"])) for i in k_tr]
    u_tr_res = [all_layer_resid(model, layers, recall_inputs(tok, unknown[i]["prompt"])) for i in u_tr]
    d_know = {L: diff_of_means_direction([r[L] for r in k_tr_res], [r[L] for r in u_tr_res]) for L in range(nL)}

    g = torch.Generator().manual_seed(100)
    orth = {}
    for L in range(nL):
        r = torch.randn(d_know[L].shape[0], generator=g)
        o = r - (r @ d_know[L]) * d_know[L]
        orth[L] = o / o.norm()

    # necessity on VAL known: per item compute clean once, then ablate at each layer
    def dlp_and_demote(inputs, gold, clean, direction, L):
        with ablating(layers[L:L + 1], direction, positions="last"):
            abl = logits_of(model, inputs)
        clp = float(torch.log_softmax(clean, -1)[gold])
        alp = float(torch.log_softmax(abl, -1)[gold])
        demoted = int((abl > abl[gold]).sum()) > int((clean > clean[gold]).sum())
        return clp - alp, demoted

    per_layer = {L: {"dk_dlp": [], "dk_demote": [], "orth_dlp": []} for L in range(nL)}
    for i in k_va:
        it = known[i]
        inputs = recall_inputs(tok, it["prompt"])
        gold = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        clean = logits_of(model, inputs)
        for L in range(nL):
            dlp, dem = dlp_and_demote(inputs, gold, clean, d_know[L], L)
            olp, _ = dlp_and_demote(inputs, gold, clean, orth[L], L)
            per_layer[L]["dk_dlp"].append(dlp)
            per_layer[L]["dk_demote"].append(dem)
            per_layer[L]["orth_dlp"].append(olp)

    n = len(k_va)
    profile = []
    for L in range(nL):
        dk = sum(per_layer[L]["dk_dlp"]) / n
        ot = sum(per_layer[L]["orth_dlp"]) / n
        dem = sum(per_layer[L]["dk_demote"]) / n
        ratio = dk / ot if abs(ot) > 1e-6 else float("inf") if dk > 1e-6 else 0.0
        profile.append({"layer": L, "dknow_dlogprob": round(dk, 4), "orth_dlogprob": round(ot, 4),
                        "ratio": round(ratio, 2) if ratio != float("inf") else "inf",
                        "frac_demoted": round(dem, 3)})
    best = max(profile, key=lambda r: r["dknow_dlogprob"])

    result = {"objective": "O4.3b_sweep", "model_id": mid, "n_layers": nL, "n_val_known": n,
              "control": "orthogonal (matched)", "best_necessity_layer": best, "profile": profile}
    HERE.mkdir(exist_ok=True)
    out = HERE / f"o4_necessity_sweep_{tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"O4.3b PER-LAYER NECESSITY SWEEP — {mid}")
    print("=" * 64)
    print(f"  {'L':>3} {'d_know Δlp':>11} {'orth Δlp':>9} {'ratio':>8} {'demoted':>8}")
    for r in profile:
        star = "  <-- peak" if r["layer"] == best["layer"] else ""
        print(f"  {r['layer']:>3} {r['dknow_dlogprob']:>11.3f} {r['orth_dlogprob']:>9.3f} "
              f"{str(r['ratio']):>8} {r['frac_demoted']:>7.0%}{star}")
    print("-" * 64)
    print(f"  peak necessity at L{best['layer']}: d_know Δlog-prob {best['dknow_dlogprob']:+.3f}, "
          f"ratio {best['ratio']}, demoted {best['frac_demoted']:.0%}")
    print(f"  wrote {out}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
