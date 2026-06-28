"""O3 — standard interp toolkit, each method behind a POSITIVE CONTROL, on
gemma-3-270m-it (charter O3). A positive control is a setup rigged so the method
MUST show its effect if the plumbing is correct; passing validates the mechanism
(not a science claim). No fallback — a failed control is reported as the finding.

Methods × controls:
  1. linear probe / diff-of-means  -> two distinct semantic classes separate
                                       held-out at AUC ~1.0
  2. logit lens                    -> head(final_norm(resid_last_layer)) reproduces
                                       the model's own top-1 EXACTLY (it IS the head)
  3. activation patching           -> patch clean(France) last-pos resid into
                                       corrupt(Japan); prediction flips toward France
  4. ablation                      -> ablate d_unembed(answer) drops its logit hard,
                                       a random direction barely moves it
  5. steering                      -> +c*d_unembed(answer) raises its logit monotonically,
                                       -c lowers it

Reuses src/gemma4_lab/interp/directions.py (model-agnostic). Run (CORE env):
  python calibration/o3_toolkit/o3_controls.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_lab.interp.directions import (
    ablating,
    diff_of_means_direction,
    rank_auc,
    steering,
    unembedding_direction,
)

MODEL_ID = "google/gemma-3-270m-it"
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# model + small helpers
# ---------------------------------------------------------------------------
def load():
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN absent — gemma-3-270m-it is gated")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
        dtype=torch.float32, attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    return tok, model


def layers_of(model) -> torch.nn.ModuleList:
    return model.model.layers


def encode_chat(tok, user: str):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt").to(DEVICE)


def logits_last(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def first_token_id(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return int(ids[0])


def capture_resid(model, layers, inputs, layer_idx: int, all_pos: bool = False):
    sink = {}

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        sink["h"] = (h[0] if all_pos else h[0, -1, :]).detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return sink["h"]


# ---------------------------------------------------------------------------
# Control 1 — linear probe / diff-of-means
# ---------------------------------------------------------------------------
def control_probe(tok, model) -> dict:
    sea = ["The deep ocean current carried the warm water north.",
           "Sailors watched the tide rise along the rocky coast.",
           "A coral reef teemed with fish beneath the waves.",
           "The river flowed into a wide salty estuary.",
           "Whales migrate across the cold open sea each winter.",
           "The harbor filled with boats as the storm surge grew.",
           "Seaweed drifted on the surface of the calm lagoon.",
           "The lighthouse warned ships away from the shallow reef.",
           "Rain fed the stream that emptied into the bay.",
           "Divers explored the shipwreck on the sandy seabed.",
           "The monsoon swelled the delta with muddy floodwater.",
           "Gulls circled above the foaming surf at dawn."]
    code = ["The function returns a list after the loop finishes.",
            "She refactored the class to remove the global variable.",
            "A null pointer crashed the compiler during the build.",
            "The recursive call overflowed the program's stack.",
            "He pushed a commit that fixed the failing unit test.",
            "The API endpoint returns JSON parsed by the client.",
            "An off-by-one error broke the array index in the loop.",
            "The thread acquired a lock before writing to the buffer.",
            "They cached the query result to speed up the request.",
            "The regex matched every token in the input string.",
            "A merge conflict appeared after rebasing the branch.",
            "The garbage collector freed the unused heap objects."]
    LAYER = 9
    pos = [capture_resid(model, layers_of(model), encode_chat(tok, s), LAYER) for s in sea]
    neg = [capture_resid(model, layers_of(model), encode_chat(tok, s), LAYER) for s in code]
    tr_p, te_p = pos[:8], pos[8:]
    tr_n, te_n = neg[:8], neg[8:]
    d = diff_of_means_direction(tr_p, tr_n)
    scores = [float(v @ d) for v in te_p] + [float(v @ d) for v in te_n]
    labels = ["yes"] * len(te_p) + ["no"] * len(te_n)
    auc = rank_auc(scores, labels)
    auc = max(auc, 1 - auc)
    return {"method": "linear_probe_diff_of_means", "layer": LAYER, "held_out_auc": round(auc, 4),
            "n_train_per_class": 8, "n_test_per_class": 4, "pass": bool(auc >= 0.9),
            "control": "two distinct semantic classes (sea vs code) must separate held-out"}


# ---------------------------------------------------------------------------
# Control 2 — logit lens
# ---------------------------------------------------------------------------
def control_logit_lens(tok, model) -> dict:
    layers = layers_of(model)
    norm = model.model.norm
    head = model.get_output_embeddings()
    prompts = ["Name the capital of France.", "Name the capital of Japan.",
               "What is two plus two?"]
    rows = []
    ok = True
    for u in prompts:
        inputs = encode_chat(tok, u)
        model_logits = logits_last(model, inputs)
        resid_last = capture_resid(model, layers, inputs, len(layers) - 1)  # last layer output
        with torch.no_grad():
            lens_logits = head(norm(resid_last.to(DEVICE))).detach().float().cpu()
        same_top1 = int(model_logits.argmax()) == int(lens_logits.argmax())
        max_abs_diff = float((model_logits - lens_logits).abs().max())
        ok = ok and same_top1
        rows.append({"prompt": u, "model_top1": tok.decode(int(model_logits.argmax())),
                     "lens_top1": tok.decode(int(lens_logits.argmax())),
                     "top1_match": same_top1, "max_abs_logit_diff": round(max_abs_diff, 4)})
    return {"method": "logit_lens", "control": "head(final_norm(last-layer resid)) == model head",
            "per_prompt": rows, "pass": bool(ok)}


# ---------------------------------------------------------------------------
# Control 3 — activation patching
# ---------------------------------------------------------------------------
def control_patching(tok, model) -> dict:
    layers = layers_of(model)
    clean = encode_chat(tok, "Name the capital of France.")   # -> Paris
    corrupt = encode_chat(tok, "Name the capital of Japan.")  # -> Tokyo
    if clean["input_ids"].shape != corrupt["input_ids"].shape:
        raise RuntimeError("clean/corrupt prompts tokenize to different lengths — fix stimuli")
    a = int(logits_last(model, clean).argmax())   # clean answer token (Paris)
    b = int(logits_last(model, corrupt).argmax())  # corrupt answer token (Tokyo)
    if a == b:
        raise RuntimeError(f"clean and corrupt predict the same token {tok.decode(a)!r} — uninformative control")

    base = logits_last(model, corrupt)
    base_gap = float(base[a] - base[b])  # logit(Paris) - logit(Tokyo) on corrupt: should be negative

    results = []
    flipped_any = False
    for L in range(0, len(layers)):
        clean_last = capture_resid(model, layers, clean, L)  # [d] clean last-pos resid at L

        def patch_hook(_m, _i, output, _cl=clean_last):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, -1, :] = _cl.to(h.device, h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h

        handle = layers[L].register_forward_hook(patch_hook)
        try:
            patched = logits_last(model, corrupt)
        finally:
            handle.remove()
        gap = float(patched[a] - patched[b])
        flipped = gap > 0  # Paris now beats Tokyo on the corrupt prompt
        flipped_any = flipped_any or flipped
        results.append({"layer": L, "gap_logit_a_minus_b": round(gap, 3), "flipped": flipped})

    best = max(results, key=lambda r: r["gap_logit_a_minus_b"])
    return {"method": "activation_patching", "clean_answer": tok.decode(a), "corrupt_answer": tok.decode(b),
            "baseline_gap": round(base_gap, 3), "best_layer": best["layer"],
            "best_gap": best["gap_logit_a_minus_b"],
            "control": "patch clean last-pos resid into corrupt; some layer flips A>B",
            "pass": bool(flipped_any), "per_layer": results}


# ---------------------------------------------------------------------------
# Control 4 — ablation
# ---------------------------------------------------------------------------
def control_ablation(tok, model) -> dict:
    # LOCALIZED (last layer, last position): global all-layer ablation is so
    # destructive that a random direction tanks the logit too (the repo's own E1
    # lesson — global necessity is an artifact). A single-point ablation of the
    # readout direction must drop the target logit while a matched random dir does not.
    layers = layers_of(model)
    inputs = encode_chat(tok, "Name the capital of France.")
    a = int(logits_last(model, inputs).argmax())  # Paris
    base = float(logits_last(model, inputs)[a])

    d_unemb = unembedding_direction(model, a)             # causally-relevant readout direction
    g = torch.Generator().manual_seed(0)
    d_rand = torch.randn(d_unemb.shape, generator=g)
    d_rand = d_rand / d_rand.norm()                        # matched random control

    def logit_under_ablation(direction):
        with ablating(layers[-1:], direction, positions="last"):  # one layer, final token
            return float(logits_last(model, inputs)[a])

    drop_unemb = base - logit_under_ablation(d_unemb)
    drop_rand = base - logit_under_ablation(d_rand)
    return {"method": "ablation", "answer": tok.decode(a), "base_logit": round(base, 3),
            "site": "last layer, last position (localized)",
            "drop_unembedding_dir": round(drop_unemb, 3), "drop_random_dir": round(drop_rand, 3),
            "control": "localized ablation of d_unembed(answer) >> matched random dir",
            "pass": bool(drop_unemb > 2.0 and drop_unemb > 5 * abs(drop_rand))}


# ---------------------------------------------------------------------------
# Control 5 — steering
# ---------------------------------------------------------------------------
def control_steering(tok, model) -> dict:
    # LOCALIZED (last layer, last position), coeff scaled to that residual's norm:
    # steering all layers/positions destabilizes, and a fixed small coeff is
    # negligible against a ~1e4-norm residual that RMSNorm then rescales. Adding
    # k*||resid||*d_unembed at the readout site must move the answer logit monotonically.
    layers = layers_of(model)
    inputs = encode_chat(tok, "Name the capital of France.")
    a = int(logits_last(model, inputs).argmax())  # Paris
    d_unemb = unembedding_direction(model, a)
    resid_norm = float(capture_resid(model, layers, inputs, len(layers) - 1).norm())

    ks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    curve = []
    for k in ks:
        c = k * resid_norm
        if k == 0.0:
            val = float(logits_last(model, inputs)[a])
        else:
            with steering(layers[-1:], d_unemb, coeff=c, positions="last"):
                val = float(logits_last(model, inputs)[a])
        curve.append((round(k, 2), round(val, 3)))
    vals = [v for _, v in curve]
    monotone = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    return {"method": "steering", "answer": tok.decode(a), "site": "last layer, last position",
            "resid_norm": round(resid_norm, 1), "curve_k_logit": curve,
            "control": "+k*||resid||*d_unembed raises the answer logit monotonically, -k lowers it",
            "pass": bool(monotone)}


def main() -> int:
    tok, model = load()
    print(f"loaded {MODEL_ID}; running 5 positive controls\n", flush=True)
    controls = [
        control_probe(tok, model),
        control_logit_lens(tok, model),
        control_patching(tok, model),
        control_ablation(tok, model),
        control_steering(tok, model),
    ]
    all_pass = all(c["pass"] for c in controls)
    print("=" * 70)
    print("O3 — toolkit positive controls (gemma-3-270m-it)")
    print("=" * 70)
    for c in controls:
        tag = "PASS" if c["pass"] else "FAIL"
        extra = {
            "linear_probe_diff_of_means": lambda c: f"held-out AUC {c['held_out_auc']}",
            "logit_lens": lambda c: f"top1 match all={c['pass']} (max|Δlogit| "
                                    f"{max(r['max_abs_logit_diff'] for r in c['per_prompt'])})",
            "activation_patching": lambda c: f"{c['clean_answer']!r} vs {c['corrupt_answer']!r}; "
                                             f"baseline gap {c['baseline_gap']} -> best {c['best_gap']} @L{c['best_layer']}",
            "ablation": lambda c: f"drop unembed {c['drop_unembedding_dir']} vs random {c['drop_random_dir']}",
            "steering": lambda c: f"curve {[v for _, v in c['curve_k_logit']]}",
        }[c["method"]](c)
        print(f"  [{tag}] {c['method']:28} — {extra}")
    print("=" * 70)
    print(f"  {'ALL CONTROLS PASS' if all_pass else 'SOME CONTROLS FAILED'}")
    (HERE / "o3_results.json").write_text(
        json.dumps({"all_pass": all_pass, "controls": controls}, indent=2), encoding="utf-8")
    print(f"  wrote {HERE / 'o3_results.json'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
