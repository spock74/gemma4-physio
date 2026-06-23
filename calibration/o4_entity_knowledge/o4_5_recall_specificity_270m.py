"""O4.5 — recall-specificity control for d_know (google/gemma-3-270m-it).
Evaluates whether the necessity of d_know is specific to factual recall or generic.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import torch

from o4_necessity import (
    load,
    locate_layers,
    logits_of,
    necessity_readouts,
    recall_inputs,
    split,
)
from gemma4_lab import observability
from gemma4_lab.interp.directions import diff_of_means_direction

TAG = "270m"
NONRECALL_INSTRUCTION = "Continue the sentence naturally."
N_RANDOM = 20
RNG_SEED = 100
PRIMARY_THRESHOLD = 3.0
FLUENCY_SMALL_FRAC = 0.33
CONTROLS_INERT_RATIO = 3.0

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "o4_necessity_sweep_270m.json"
OUT = HERE / "results" / "o4_5_recall_specificity_270m.json"
NONRECALL_JSON = Path("data/eval/nonrecall_knownfixed.json")
FLUENCY_JSON = Path("data/eval/fluency_neutral.json")
CORPUS = Path("data/eval/entity_knowledge_contrast.json")


def generic_inputs(tok, stem: str):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": NONRECALL_INSTRUCTION}],
        tokenize=False, add_generation_prompt=True) + stem
    return tok(prompt, return_tensors="pt").to("mps")


def all_layer_resid(model, layers, inputs) -> dict:
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


def perplexity(model, inputs, layers=None, L=None, direction=None) -> float:
    from gemma4_lab.interp.directions import ablating
    import contextlib
    ids = inputs["input_ids"]
    ctx = ablating(layers[L:L + 1], direction, positions=None) if L is not None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, :-1, :].float()
    tgt = ids[0, 1:]
    logp = torch.log_softmax(logits, -1)
    nll = -logp[range(tgt.shape[0]), tgt].mean()
    return float(torch.exp(nll))


def main() -> int:
    observability.setup()

    if not SWEEP.exists():
        raise RuntimeError(f"O4.3b sweep result missing at {SWEEP} — run necessity sweep first.")
        
    sweep = json.loads(SWEEP.read_text())
    
    # Identify the band where dknow_dlogprob >= 20.0
    band = sorted(r["layer"] for r in sweep["profile"] if r["dknow_dlogprob"] >= 20.0)
    if not band:
        # Fallback to top-2 layers by necessity if no layer is >= 20.0
        sorted_layers = sorted(sweep["profile"], key=lambda r: r["dknow_dlogprob"], reverse=True)
        band = sorted(r["layer"] for r in sorted_layers[:2])
        
    peak = max(sweep["profile"], key=lambda r: r["dknow_dlogprob"])["layer"]
    control_layers = band  # for 270m, band is small, so controls run on all band layers
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    prereg = {
        "model": f"google/gemma-3-{TAG}-it",
        "intervention": "single-layer directional ablation at the recall position",
        "d_know": "O4 known/unknown diff-of-means, refit per band-layer on Train; eval on Val",
        "band": band, "peak_layer": peak, "control_layers": control_layers,
        "conditions": {"recall": "RECALL_INSTRUCTION + known stem, gold next",
                       "nonrecall_knownfixed": "fact stated, generic next",
                       "fluency": "neutral, no entity"},
        "common_metric": "KL(clean||ablated) over full next-token dist at readout position",
        "controls": f"N={N_RANDOM} matched-random + 1 orthogonal per condition at control_layers",
        "primary": "in-band median KL_recall(d_know) / in-band median KL_nonrecall(d_know)",
        "thresholds": {"primary_gt": PRIMARY_THRESHOLD, "fluency_small_frac": FLUENCY_SMALL_FRAC,
                       "controls_inert_ratio": CONTROLS_INERT_RATIO},
        "verdict_rule": "RECALL_SPECIFIC (knowledge gate) iff primary>3 AND fluency_frac<0.33 AND controls_inert>3; else GENERIC_CHANNEL",
    }
    
    OUT.write_text(json.dumps({"objective": "O4.5", "status": "prereg_written", "prereg": prereg}, indent=2),
                   encoding="utf-8")

    tok, model, model_id = load(TAG)
    layers = locate_layers(model)
    print(f"loaded {model_id}; band={band} peak L{peak} control_layers={control_layers}\n", flush=True)

    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    nonrec = json.loads(NONRECALL_JSON.read_text())["items"]
    fluency = json.loads(FLUENCY_JSON.read_text())["items"]
    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)

    # fit d_know per band-layer on TRAIN (recall readout), held-out
    k_tr_res = [all_layer_resid(model, layers, recall_inputs(tok, known[i]["prompt"])) for i in k_tr]
    u_tr_res = [all_layer_resid(model, layers, recall_inputs(tok, unknown[i]["prompt"])) for i in u_tr]
    d_know = {L: diff_of_means_direction([r[L] for r in k_tr_res], [r[L] for r in u_tr_res]) for L in band}

    g = torch.Generator().manual_seed(RNG_SEED)
    d_model = int(d_know[band[0]].shape[0])
    rands = [(lambda r: r / r.norm())(torch.randn(d_model, generator=g)) for _ in range(N_RANDOM)]
    orth = {}
    for L in control_layers:
        rr = torch.randn(d_model, generator=g)
        o = rr - (rr @ d_know[L]) * d_know[L]
        orth[L] = o / o.norm()

    conditions = {
        "recall": [(recall_inputs(tok, known[i]["prompt"]),) for i in k_va],
        "nonrecall_knownfixed": [(generic_inputs(tok, nonrec[i]["stem"]),) for i in k_va],
        "fluency": [(generic_inputs(tok, it["stem"]),) for it in fluency],
    }

    def kl_dknow_per_layer(items, cleans):
        out = {}
        for L in band:
            kls = [necessity_readouts(model, layers, L, inp, 0, clean, d_know[L])["kl"]
                   for (inp,), clean in zip(items, cleans)]
            out[L] = median(kls)
        return out

    def kl_controls_per_layer(items, cleans):
        out = {}
        for L in control_layers:
            per_dir = [[] for _ in range(N_RANDOM)]
            orth_kls = []
            for (inp,), clean in zip(items, cleans):
                for j, rd in enumerate(rands):
                    per_dir[j].append(necessity_readouts(model, layers, L, inp, 0, clean, rd)["kl"])
                orth_kls.append(necessity_readouts(model, layers, L, inp, 0, clean, orth[L])["kl"])
            rand_dir_med = [median(c) for c in per_dir]
            out[L] = {"random_max": max(rand_dir_med), "random_mean": sum(rand_dir_med) / N_RANDOM,
                      "orth": median(orth_kls)}
        return out

    results = {}
    for name, items in conditions.items():
        print(f"  condition {name}: {len(items)} items — d_know KL across band ...", flush=True)
        cleans = [logits_of(model, inp) for (inp,) in items]
        dk = kl_dknow_per_layer(items, cleans)
        ctrl = kl_controls_per_layer(items, cleans)
        results[name] = {"dknow_kl_per_layer": {str(k): round(v, 4) for k, v in dk.items()},
                         "control_kl_per_layer": {str(k): {kk: round(vv, 4) for kk, vv in v.items()}
                                                  for k, v in ctrl.items()},
                         "inband_median_dknow_kl": round(median(list(dk.values())), 4)}
        print(f"    in-band median d_know KL = {results[name]['inband_median_dknow_kl']}", flush=True)

    flu_items = conditions["fluency"]
    clean_ppls = [perplexity(model, inp) for (inp,) in flu_items]
    abl_ppls = [perplexity(model, inp, layers, peak, d_know[peak]) for (inp,) in flu_items]
    fluency_ppl = {"clean_median": round(median(clean_ppls), 3),
                   "dknow_ablated_at_peak_median": round(median(abl_ppls), 3),
                   "ppl_ratio_ablated_over_clean": round(median(abl_ppls) / median(clean_ppls), 3)}

    recall_kl = results["recall"]["inband_median_dknow_kl"]
    nonrecall_kl = results["nonrecall_knownfixed"]["inband_median_dknow_kl"]
    fluency_kl = results["fluency"]["inband_median_dknow_kl"]
    primary_ratio = recall_kl / nonrecall_kl if nonrecall_kl > 1e-9 else float("inf")
    fluency_frac = fluency_kl / recall_kl if recall_kl > 1e-9 else float("inf")
    
    ctrl_recall = results["recall"]["control_kl_per_layer"]
    dk_recall_at_ctrl = median([results["recall"]["dknow_kl_per_layer"][str(L)] for L in control_layers])
    worst_ctrl = max(max(v["random_max"], v["orth"]) for v in ctrl_recall.values())
    controls_inert_ratio = dk_recall_at_ctrl / worst_ctrl if worst_ctrl > 1e-9 else float("inf")

    specific = bool(primary_ratio > PRIMARY_THRESHOLD and fluency_frac < FLUENCY_SMALL_FRAC
                    and controls_inert_ratio > CONTROLS_INERT_RATIO)
    verdict = {
        "inband_median_KL": {"recall": recall_kl, "nonrecall_knownfixed": nonrecall_kl, "fluency": fluency_kl},
        "primary_ratio_recall_over_nonrecall": round(primary_ratio, 3),
        "fluency_fraction_of_recall": round(fluency_frac, 3),
        "controls_inert_ratio": round(controls_inert_ratio, 3),
        "fluency_perplexity": fluency_ppl,
        "verdict": "RECALL_SPECIFIC" if specific else "GENERIC_CHANNEL",
        "interpretation": (
            "RECALL-SPECIFIC (knowledge gate): ablating d_know disrupts factual recall far more than "
            "known-ness-matched non-recall or neutral fluency; the band carries a recall/knowledge signal."
            if specific else
            "GENERIC CHANNEL: ablating d_know perturbs non-recall / fluency comparably to recall — "
            "d_know is a load-bearing mid-network direction, NOT specifically a recall gate. "
            "Downgrade 'erases recall' to 'load-bearing direction'. (Valid null-type result.)"),
        "criteria": {"primary>3": primary_ratio > PRIMARY_THRESHOLD,
                     "fluency_small(<0.33)": fluency_frac < FLUENCY_SMALL_FRAC,
                     "controls_inert(>3)": controls_inert_ratio > CONTROLS_INERT_RATIO},
    }
    
    output_meta = {
        "objective": "O4.5", "status": "done", "model_id": model_id,
        "prereg": prereg, "conditions": results, "verdict": verdict
    }
    
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)

    print("\n" + "=" * 72)
    print(f"O4.5 RECALL-SPECIFICITY — {model_id}")
    print("=" * 72)
    print(f"  in-band median KL  recall={recall_kl}  nonrecall={nonrecall_kl}  fluency={fluency_kl}")
    print(f"  primary KL_recall/KL_nonrecall = {primary_ratio:.2f}  (need >{PRIMARY_THRESHOLD})")
    print(f"  fluency fraction of recall     = {fluency_frac:.2f}  (need <{FLUENCY_SMALL_FRAC})")
    print(f"  controls inert ratio           = {controls_inert_ratio:.2f}  (need >{CONTROLS_INERT_RATIO})")
    print(f"  fluency perplexity clean {fluency_ppl['clean_median']} -> ablated {fluency_ppl['dknow_ablated_at_peak_median']} "
          f"(x{fluency_ppl['ppl_ratio_ablated_over_clean']})")
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  {verdict['interpretation']}")
    print(f"  wrote {OUT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
