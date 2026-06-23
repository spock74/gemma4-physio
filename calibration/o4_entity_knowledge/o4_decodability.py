"""O4.1 + O4.2 — known/unknown decodability across ALL layers, and readout
validity, on a Gemma 3 -it model. Non-causal, safe (no interventions).

O4.1: per-layer held-out AUC of a diff-of-means known/unknown direction (d_know),
fit on TRAIN, evaluated on a disjoint VAL split — the all-layers decodability
profile the old gemma-4 track could not produce (layer-17-only).

O4.2: under the assistant-prefill recall readout, the clean gold-token RANK for
each known item — does the -it model actually recall (low rank), or echo (high)?
Necessity claims (O4.3, held for review) only apply where recall is real.

Reuses src/gemma4_lab/interp/directions.py and the RECALL_INSTRUCTION readout.

Run (CORE env):
  python calibration/o4_entity_knowledge/o4_decodability.py 270m
  python calibration/o4_entity_knowledge/o4_decodability.py 4b
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from gemma4_lab.interp.directions import diff_of_means_direction, projection, rank_auc
from gemma4_lab.interp.entity_knowledge import RECALL_INSTRUCTION

CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
HERE = Path(__file__).resolve().parent
CORPUS = Path("data/eval/entity_knowledge_contrast.json")
MODELS = {"270m": "google/gemma-3-270m-it", "4b": "google/gemma-3-4b-it"}


def locate_layers(model: nn.Module) -> nn.ModuleList:
    for path in (("model", "layers"), ("model", "language_model", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("decoder layers not found")


def load(tag: str):
    model_id = MODELS[tag]
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(f"HF_TOKEN absent — {model_id} is gated")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"))
    cls = AutoModelForCausalLM if tag == "270m" else AutoModelForImageTextToText
    # 4b in fp32 = ~17 GB > 16 GB host RAM -> bf16 (adequate for diff-of-means/AUC;
    # the fp32 requirement was specific to O2 SAE-vs-Neuronpedia magnitude matching).
    dtype = torch.float32 if tag == "270m" else torch.bfloat16
    model = cls.from_pretrained(
        model_id, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
        dtype=dtype, attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    return tok, model, model_id


def recall_inputs(tok, stem: str):
    """Assistant-prefill recall readout: RECALL_INSTRUCTION as the user turn, the
    cloze stem prefilling the model's own turn so the fact is the next token."""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": RECALL_INSTRUCTION}],
        tokenize=False, add_generation_prompt=True) + stem
    return tok(prompt, return_tensors="pt").to(DEVICE)


def capture_all_layers(model, layers, inputs) -> dict[int, torch.Tensor]:
    sink: dict[int, torch.Tensor] = {}

    def mk(i):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, tuple) else out
            sink[i] = h[0, -1, :].detach().float().cpu()
        return hook

    handles = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return sink


def last_logits(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def split(n: int, seed: int, val_frac=0.5):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    nv = max(1, round(n * val_frac))
    val = set(perm[:nv])
    return [i for i in range(n) if i not in val], sorted(val)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "270m"
    if tag not in MODELS:
        raise RuntimeError(f"unknown model tag {tag!r}; choose from {list(MODELS)}")
    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]

    tok, model, model_id = load(tag)
    layers = locate_layers(model)
    n_layers = len(layers)
    print(f"loaded {model_id}: {n_layers} layers; readout=assistant-prefill recall\n", flush=True)

    # O4.1 capture: last-token resid at all layers, one forward per prompt
    k_res = [capture_all_layers(model, layers, recall_inputs(tok, it["prompt"])) for it in known]
    u_res = [capture_all_layers(model, layers, recall_inputs(tok, it["prompt"])) for it in unknown]

    # health
    bad = [(c, i) for c, res in (("known", k_res), ("unknown", u_res)) for i, r in enumerate(res)
           for L, t in r.items() if not torch.isfinite(t).all()]
    if bad:
        raise RuntimeError(f"non-finite residuals captured: {bad[:10]} ... — STOP (no silent recovery)")

    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)

    def dir_at(L, ktr, utr):
        return diff_of_means_direction([k_res[i][L] for i in ktr], [u_res[i][L] for i in utr])

    def auc_at(L, d, kidx, uidx):
        scores = [projection(k_res[i][L], d) for i in kidx] + [projection(u_res[i][L], d) for i in uidx]
        labels = ["yes"] * len(kidx) + ["no"] * len(uidx)
        a = rank_auc(scores, labels)
        return max(a, 1 - a) if a is not None else 0.5

    profile = []
    for L in range(n_layers):
        d = dir_at(L, k_tr, u_tr)
        profile.append({"layer": L,
                        "train_auc": round(auc_at(L, d, k_tr, u_tr), 4),
                        "val_auc": round(auc_at(L, d, k_va, u_va), 4)})
    best = max(profile, key=lambda r: r["val_auc"])

    # O4.2 readout validity: clean gold-token rank for known items
    def gold_id(ans):
        ids = tok(" " + ans.strip(), add_special_tokens=False)["input_ids"]
        return int(ids[0])

    ranks = []
    for it in known:
        lg = last_logits(model, recall_inputs(tok, it["prompt"]))
        gid = gold_id(it["answer"])
        rank = int((lg > lg[gid]).sum())
        ranks.append({"prompt": it["prompt"], "answer": it["answer"], "gold_rank": rank,
                      "top1": tok.decode(int(lg.argmax()))})
    ranks_sorted = sorted(r["gold_rank"] for r in ranks)
    med_rank = ranks_sorted[len(ranks_sorted) // 2]
    frac_top1 = sum(1 for r in ranks if r["gold_rank"] == 0) / len(ranks)
    frac_top5 = sum(1 for r in ranks if r["gold_rank"] < 5) / len(ranks)

    result = {
        "objective": "O4.1+O4.2", "model_id": model_id, "n_layers": n_layers,
        "readout": "assistant_prefill_recall", "instruction": RECALL_INSTRUCTION,
        "n_known": len(known), "n_unknown": len(unknown),
        "split": {"n_train_known": len(k_tr), "n_val_known": len(k_va),
                  "n_train_unknown": len(u_tr), "n_val_unknown": len(u_va)},
        "best_layer_by_val_auc": best,
        "layer_profile": profile,
        "readout_validity": {"median_gold_rank": med_rank, "frac_recalled_top1": round(frac_top1, 3),
                             "frac_recalled_top5": round(frac_top5, 3),
                             "per_item": ranks},
    }
    HERE.mkdir(exist_ok=True)
    out = HERE / f"o4_decodability_{tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"O4.1 DECODABILITY — known/unknown held-out AUC across all layers ({model_id})")
    print("=" * 70)
    print(f"  {'layer':>5} {'train':>6} {'val':>6}")
    for r in profile:
        star = "  <- best val" if r["layer"] == best["layer"] else ""
        print(f"  {r['layer']:>5} {r['train_auc']:>6.3f} {r['val_auc']:>6.3f}{star}")
    print(f"\n  best decodable layer = {best['layer']}  (val AUC {best['val_auc']:.3f}, train {best['train_auc']:.3f})")
    print("-" * 70)
    print(f"O4.2 READOUT VALIDITY — does the -it model actually recall known facts?")
    print(f"  median gold rank = {med_rank}   top-1 {frac_top1:.0%}   top-5 {frac_top5:.0%}   (n={len(known)})")
    print(f"  -> necessity (O4.3) is only meaningful on items with low clean rank")
    print(f"  wrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
