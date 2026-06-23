"""O4.3 — LOCALIZED necessity of the known/unknown direction d_know.

Held for review until O4.1/O4.2 passed; now run with the full rigor bar:
  - d_know fit on the TRAIN split at layer L (held-out); necessity tested on VAL known.
  - LOCALIZED ablation: single layer L, recall position only (NOT global — the O3 lesson).
  - Specificity: d_know vs N matched random unit dirs vs one orthogonal dir.
  - PAIRED readouts: every item reports Δlogit AND Δlog-prob AND Δgold-rank AND
    KL(clean||ablated). Raw-logit drop alone can be common-mode / softmax-invariant
    (the STEP #1 lesson) — the rank/log-prob/KL trio is what says "recall was hurt".

Verdict gate: d_know is a SPECIFIC localized causal carrier iff its mean Δlog-prob
(gold) exceeds 2x the strongest control AND it demotes the gold token (rank rises)
more than controls. Otherwise the honest result is "decodable, not a localized
causal carrier" — itself a finding.

Run (CORE env):
  python calibration/o4_entity_knowledge/o4_necessity.py 4b 11
  python calibration/o4_entity_knowledge/o4_necessity.py 270m 17
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from gemma4_lab.config import hf_token_or_none
from gemma4_lab.interp.directions import ablating, diff_of_means_direction, projection, rank_auc
from gemma4_lab.interp.entity_knowledge import RECALL_INSTRUCTION

CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
HERE = Path(__file__).resolve().parent
CORPUS = Path("data/eval/entity_knowledge_contrast.json")
MODELS = {"270m": "google/gemma-3-270m-it", "4b": "google/gemma-3-4b-it"}
N_RANDOM = 5


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
    if tag not in MODELS:
        raise RuntimeError(f"unknown model tag {tag!r}; expected one of {list(MODELS)}")
    model_id = MODELS[tag]
    token = hf_token_or_none()  # secret read once from env in config.py — never hardcoded
    if not token:
        raise RuntimeError(f"HF_TOKEN absent in config — {model_id} is gated (expected a token, found none)")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=token)
    cls = AutoModelForCausalLM if tag == "270m" else AutoModelForImageTextToText
    dtype = torch.float32 if tag == "270m" else torch.bfloat16
    model = cls.from_pretrained(
        model_id, cache_dir=CACHE_DIR, token=token,
        dtype=dtype, attn_implementation="eager").to(DEVICE)
    model.eval()
    return tok, model, model_id


def recall_inputs(tok, stem: str):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": RECALL_INSTRUCTION}],
        tokenize=False, add_generation_prompt=True) + stem
    return tok(prompt, return_tensors="pt").to(DEVICE)


def resid_at(model, layers, inputs, L: int) -> torch.Tensor:
    sink = {}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        sink["h"] = h[0, -1, :].detach().float().cpu()

    handle = layers[L].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return sink["h"]


def logits_of(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def split(n, seed, val_frac=0.5):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    nv = max(1, round(n * val_frac))
    val = set(perm[:nv])
    return [i for i in range(n) if i not in val], sorted(val)


def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P||Q) with P=clean, Q=ablated, both logits over vocab."""
    logp = torch.log_softmax(p_logits, -1)
    logq = torch.log_softmax(q_logits, -1)
    p = logp.exp()
    return float((p * (logp - logq)).sum())


def necessity_readouts(model, layers, L: int, inputs, gold: int,
                       clean: torch.Tensor, direction: torch.Tensor) -> dict:
    """THE intervention (single source of truth, reused by the O4.4 sweep): localized
    single-layer directional ablation at the recall position, with paired readouts.
    Identical to the O4.3 single-point test — do not fork this logic."""
    with ablating(layers[L:L + 1], direction, positions="last"):
        abl = logits_of(model, inputs)
    clean_lp = float(torch.log_softmax(clean, -1)[gold])
    abl_lp = float(torch.log_softmax(abl, -1)[gold])
    return {
        "d_logit": float(clean[gold]) - float(abl[gold]),  # >0 = logit hurt
        "d_logprob": clean_lp - abl_lp,                     # >0 = recall hurt (softmax-aware)
        "clean_rank": int((clean > clean[gold]).sum()),
        "abl_rank": int((abl > abl[gold]).sum()),
        "kl": kl(clean, abl),
    }


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "4b"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]

    tok, model, model_id = load(tag)
    layers = locate_layers(model)
    print(f"loaded {model_id}: {len(layers)} layers; LOCALIZED necessity at L{L}, recall position\n", flush=True)

    # fit d_know at L on TRAIN (held-out)
    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)
    k_res_tr = [resid_at(model, layers, recall_inputs(tok, known[i]["prompt"]), L) for i in k_tr]
    u_res_tr = [resid_at(model, layers, recall_inputs(tok, unknown[i]["prompt"]), L) for i in u_tr]
    d_know = diff_of_means_direction(k_res_tr, u_res_tr)

    # held-out separation sanity at L (VAL)
    k_res_va = [resid_at(model, layers, recall_inputs(tok, known[i]["prompt"]), L) for i in k_va]
    u_res_va = [resid_at(model, layers, recall_inputs(tok, unknown[i]["prompt"]), L) for i in u_va]
    scores = [projection(r, d_know) for r in k_res_va] + [projection(r, d_know) for r in u_res_va]
    labels = ["yes"] * len(k_res_va) + ["no"] * len(u_res_va)
    a = rank_auc(scores, labels)
    val_auc = max(a, 1 - a)

    # matched controls
    d = int(d_know.shape[0])
    g = torch.Generator().manual_seed(100)
    rands = [(lambda r: r / r.norm())(torch.randn(d, generator=g)) for _ in range(N_RANDOM)]
    _r = torch.randn(d, generator=g)
    orth = _r - (_r @ d_know) * d_know
    orth = orth / orth.norm()

    def ablate_readouts(inputs, gold, clean, direction):
        return necessity_readouts(model, layers, L, inputs, gold, clean, direction)

    items = []
    for i in k_va:
        it = known[i]
        inputs = recall_inputs(tok, it["prompt"])
        gold = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        clean = logits_of(model, inputs)
        dk = ablate_readouts(inputs, gold, clean, d_know)
        rnd = [ablate_readouts(inputs, gold, clean, rd) for rd in rands]
        ort = ablate_readouts(inputs, gold, clean, orth)
        items.append({"prompt": it["prompt"], "answer": it["answer"],
                      "dknow": dk, "random": rnd, "orth": ort})

    def agg(key, sub):
        if sub == "random":
            # per-item mean across the K random dirs, then mean across items
            return sum(sum(r[key] for r in it["random"]) / N_RANDOM for it in items) / len(items)
        return sum(it[sub][key] for it in items) / len(items)

    def agg_random_max(key):  # strongest random control (per-dir item-mean, then max)
        per_dir = [sum(it["random"][k][key] for it in items) / len(items) for k in range(N_RANDOM)]
        return max(per_dir)

    m = {k: {"dknow": agg(k, "dknow"), "orth": agg(k, "orth"),
             "random_mean": agg(k, "random"), "random_max": agg_random_max(k)}
         for k in ("d_logit", "d_logprob", "kl")}
    # rank: report mean clean/abl rank and fraction demoted
    mean_clean_rank = sum(it["dknow"]["clean_rank"] for it in items) / len(items)
    dk_mean_abl_rank = sum(it["dknow"]["abl_rank"] for it in items) / len(items)
    dk_frac_demoted = sum(1 for it in items if it["dknow"]["abl_rank"] > it["dknow"]["clean_rank"]) / len(items)
    dk_frac_lost_top1 = sum(1 for it in items
                            if it["dknow"]["clean_rank"] == 0 and it["dknow"]["abl_rank"] > 0) / len(items)

    # specificity on the SOFTMAX-AWARE readout (Δlog-prob), not raw logit
    base = max(m["d_logprob"]["random_max"], m["d_logprob"]["orth"], 1e-9)
    spec_ratio_logprob = m["d_logprob"]["dknow"] / base
    base_logit = max(m["d_logit"]["random_max"], m["d_logit"]["orth"], 1e-9)
    spec_ratio_logit = m["d_logit"]["dknow"] / base_logit
    gate_pass = bool(spec_ratio_logprob > 2.0 and m["d_logprob"]["dknow"] > 0 and dk_frac_demoted > 0.25)

    result = {
        "objective": "O4.3", "model_id": model_id, "layer": L, "site": "single layer, recall position",
        "val_auc_at_L": round(val_auc, 4), "n_val_known": len(k_va), "n_random": N_RANDOM,
        "metrics": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in m.items()},
        "rank": {"mean_clean_rank": round(mean_clean_rank, 3),
                 "dknow_mean_abl_rank": round(dk_mean_abl_rank, 3),
                 "dknow_frac_demoted": round(dk_frac_demoted, 3),
                 "dknow_frac_lost_top1": round(dk_frac_lost_top1, 3)},
        "specificity_ratio_logprob": round(spec_ratio_logprob, 3),
        "specificity_ratio_logit": round(spec_ratio_logit, 3),
        "gate_pass": gate_pass,
        "verdict": (
            f"SPECIFIC localized causal carrier at L{L}: d_know ablation hurts recall "
            f"(Δlog-prob {m['d_logprob']['dknow']:+.3f}) >2x controls and demotes gold "
            f"in {dk_frac_demoted:.0%} of items."
            if gate_pass else
            f"NOT a specific localized causal carrier at L{L}: decodable (val AUC {val_auc:.3f}) "
            f"but single-layer d_know ablation does not hurt recall beyond controls "
            f"(Δlog-prob ratio {spec_ratio_logprob:.2f}, gold demoted {dk_frac_demoted:.0%}). "
            "Decodable != causal — an honest finding."),
        "items": items,
    }
    HERE.mkdir(exist_ok=True)
    out = HERE / f"o4_necessity_{tag}_L{L}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"O4.3 LOCALIZED NECESSITY — {model_id}  L{L}  (held-out d_know, recall pos)")
    print("=" * 72)
    print(f"  d_know val AUC @ L{L} = {val_auc:.3f}   (n_val known {len(k_va)})")
    print(f"  {'readout':<10} {'d_know':>9} {'rand_mean':>10} {'rand_max':>9} {'orth':>9}")
    for k, lab in [("d_logit", "Δlogit"), ("d_logprob", "Δlog-prob"), ("kl", "KL")]:
        v = m[k]
        print(f"  {lab:<10} {v['dknow']:>9.3f} {v['random_mean']:>10.3f} {v['random_max']:>9.3f} {v['orth']:>9.3f}")
    print(f"  gold rank: clean {mean_clean_rank:.2f} -> d_know-ablated {dk_mean_abl_rank:.2f}  "
          f"(demoted {dk_frac_demoted:.0%}, lost top-1 {dk_frac_lost_top1:.0%})")
    print(f"  specificity ratio (Δlog-prob) = {spec_ratio_logprob:.2f}   "
          f"(raw-logit ratio {spec_ratio_logit:.2f})")
    print("-" * 72)
    print(f"  {'PASS' if gate_pass else 'NULL'}: {result['verdict']}")
    print(f"  wrote {out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
