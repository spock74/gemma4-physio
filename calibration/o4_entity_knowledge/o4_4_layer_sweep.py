"""O4.4 — per-layer necessity sweep on gemma-3-4b-it, WITH FULL CONTROLS.

Wraps the O4.3 intervention (`o4_necessity.necessity_readouts`) in a layer loop —
the intervention itself is NOT rewritten. Everything is held identical to O4.3
except the layer: same seed-0 TRAIN/VAL split, same recall-position single-layer
ablation, same paired readouts. d_know is refit per layer on TRAIN; ablation runs
on VAL known only.

Per layer L:
  - fit d_know@L (diff-of-means, TRAIN, unit-norm as in O4.3)
  - ablate d_know@L at recall pos on VAL: Δlogit, Δlog-prob(gold), gold-rank
    (clean→ablated), KL(clean‖ablated)
  - controls@L: N_RANDOM matched-random unit dirs + 1 orthogonal; record
    Δlog-prob max AND mean for random, Δlog-prob for orth
  - specificity_ratio@L = Δlogprob(d_know) / max(Δlogprob(random_max), Δlogprob(orth))

PRE-REGISTERED (written to the JSON before any result row; constants are frozen and
NOT tuned after seeing results):
  PASS@L iff specificity_ratio > RATIO_THRESHOLD AND |control Δlogprob| < CONTROL_SMALL.
  Headline (O4.3 "L11 is a causal carrier") SURVIVES iff L11 PASS AND it sits in a
  contiguous active band (>=2 adjacent PASS incl. L11) OR is a clear global max with
  trending PASS neighbours; else DOWNGRADE to "layer-specific, possibly selected".

Schedule: full range(34), spiral order outward from L11 (so L11 and its neighbours —
the headline-relevant layers — are computed first and survive an interruption).
Incremental save after every layer. Logfire span per layer (project rule).

Run (CORE env):  python calibration/o4_entity_knowledge/o4_4_layer_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from o4_necessity import (  # reuse — do NOT rewrite the intervention
    CORPUS,
    load,
    locate_layers,
    logits_of,
    necessity_readouts,
    recall_inputs,
    split,
)
from gemma4_lab.interp.directions import diff_of_means_direction

# ---- PRE-REGISTERED constants (frozen before results; never tuned post-hoc) ----
TAG = "4b"
N_RANDOM = 20                 # >= 20 matched-random controls per layer
RATIO_THRESHOLD = 2.0
CONTROL_SMALL = 1.0           # nats; the strongest control Δlog-prob must be below this
DENOM_FLOOR = 0.5             # nats; floor on the ratio denominator so ~0 controls don't explode it
RNG_SEED = 100                # reproducible random controls
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "o4_4_layer_sweep_4b.json"

PREREG = {
    "model": "google/gemma-3-4b-it (-it only)",
    "intervention": "reused verbatim from o4_necessity.necessity_readouts (O4.3); "
                    "single-layer directional ablation at the recall position",
    "split": "seed-0 TRAIN/VAL (split(known,0)/split(unknown,1)) — identical to o4_decodability/o4_necessity; "
             "d_know refit per layer on TRAIN; ablate on VAL known only; zero re-split",
    "n_random": N_RANDOM, "ratio_threshold": RATIO_THRESHOLD, "control_small_nats": CONTROL_SMALL,
    "denom_floor_nats": DENOM_FLOOR, "rng_seed": RNG_SEED,
    "pass_rule": "PASS@L iff specificity_ratio > 2 AND max(random_max_dlp, |orth_dlp|) < 1.0 nat",
    "specificity_ratio": "d_know Δlogprob / max(random_max Δlogprob, orth Δlogprob, DENOM_FLOOR)",
    "headline_survival_rule": "O4.3 headline SURVIVES iff L11 PASS AND (L10 PASS OR L12 PASS) "
                              "[contiguous active band >=2]; else DOWNGRADE to 'layer-specific, possibly selected'",
    "schedule": "full range(34), spiral order from L11; incremental save per layer",
}


def all_layer_resid(model, layers, inputs) -> dict:
    """Capture last-token resid at every layer in one forward pass (Logfire-spanned)."""
    import logfire
    sink = {}

    def mk(i):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            sink[i] = h[0, -1, :].detach().float().cpu()
        return hook
    hs = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    try:
        with logfire.span("o4_4.capture_all_layers"):
            with torch.no_grad():
                model(**inputs)
    finally:
        for h in hs:
            h.remove()
    return sink


def main() -> int:
    import logfire
    from gemma4_lab import observability
    observability.setup()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]

    tok, model, model_id = load(TAG)
    layers = locate_layers(model)
    nL = len(layers)
    print(f"loaded {model_id}: {nL} layers; O4.4 full-control necessity sweep "
          f"(N_RANDOM={N_RANDOM} + orth per layer)\n", flush=True)

    # seed-0 split (identical to O4.1/O4.3)
    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)

    # capture all-layer TRAIN resid once (fit d_know per layer)
    k_tr_res = [all_layer_resid(model, layers, recall_inputs(tok, known[i]["prompt"])) for i in k_tr]
    u_tr_res = [all_layer_resid(model, layers, recall_inputs(tok, unknown[i]["prompt"])) for i in u_tr]
    d_know = {L: diff_of_means_direction([r[L] for r in k_tr_res], [r[L] for r in u_tr_res]) for L in range(nL)}

    # matched-random controls (shared across layers; unit-norm like d_know) + per-layer orth
    g = torch.Generator().manual_seed(RNG_SEED)
    d_model = int(d_know[0].shape[0])
    rands = [(lambda r: r / r.norm())(torch.randn(d_model, generator=g)) for _ in range(N_RANDOM)]
    orth = {}
    for L in range(nL):
        rr = torch.randn(d_model, generator=g)
        o = rr - (rr @ d_know[L]) * d_know[L]
        orth[L] = o / o.norm()

    # VAL known: clean logits + gold id, computed ONCE per item (layer-independent)
    val = []
    for i in k_va:
        it = known[i]
        inputs = recall_inputs(tok, it["prompt"])
        gold = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        clean = logits_of(model, inputs)
        val.append({"prompt": it["prompt"], "answer": it["answer"], "inputs": inputs,
                    "gold": gold, "clean": clean})

    schedule = sorted(range(nL), key=lambda L: (abs(L - 11), L))  # spiral from L11
    print(f"schedule (spiral from L11): {schedule}\n", flush=True)

    rows_by_L: dict[int, dict] = {}

    def write(status: str, verdict=None):
        out = {"objective": "O4.4", "model_id": model_id, "n_layers": nL,
               "n_val_known": len(k_va), "status": status,
               "prereg": PREREG,
               "rows": [rows_by_L[L] for L in range(nL) if L in rows_by_L]}
        if verdict is not None:
            out["verdict"] = verdict
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    write("in_progress")
    for L in schedule:
        with logfire.span("o4_4.layer", layer=L):
            dk_dlp, dk_dlogit, dk_kl = [], [], []
            clean_ranks, abl_ranks_dk = [], []
            rand_dlp_per_dir = [[] for _ in range(N_RANDOM)]
            orth_dlp = []
            for v in val:
                dk = necessity_readouts(model, layers, L, v["inputs"], v["gold"], v["clean"], d_know[L])
                dk_dlp.append(dk["d_logprob"]); dk_dlogit.append(dk["d_logit"]); dk_kl.append(dk["kl"])
                clean_ranks.append(dk["clean_rank"]); abl_ranks_dk.append(dk["abl_rank"])
                for j, rd in enumerate(rands):
                    rand_dlp_per_dir[j].append(
                        necessity_readouts(model, layers, L, v["inputs"], v["gold"], v["clean"], rd)["d_logprob"])
                orth_dlp.append(
                    necessity_readouts(model, layers, L, v["inputs"], v["gold"], v["clean"], orth[L])["d_logprob"])

            n = len(val)
            dknow_dlp = sum(dk_dlp) / n
            rand_dir_means = [sum(c) / n for c in rand_dlp_per_dir]
            rand_max = max(rand_dir_means)
            rand_mean = sum(rand_dir_means) / len(rand_dir_means)
            orth_mean = sum(orth_dlp) / n
            denom = max(rand_max, orth_mean, DENOM_FLOOR)
            ratio = dknow_dlp / denom
            control_max_abs = max(abs(rand_max), abs(rand_mean), abs(orth_mean))
            passed = bool(ratio > RATIO_THRESHOLD and control_max_abs < CONTROL_SMALL)
            rows_by_L[L] = {
                "layer": L,
                "dknow_dlogprob": round(dknow_dlp, 4),
                "dknow_dlogit": round(sum(dk_dlogit) / n, 4),
                "dknow_kl": round(sum(dk_kl) / n, 4),
                "mean_clean_rank": round(sum(clean_ranks) / n, 3),
                "mean_abl_rank_dknow": round(sum(abl_ranks_dk) / n, 3),
                "frac_demoted_dknow": round(sum(1 for c, a in zip(clean_ranks, abl_ranks_dk) if a > c) / n, 3),
                "random_max_dlogprob": round(rand_max, 4),
                "random_mean_dlogprob": round(rand_mean, 4),
                "orth_dlogprob": round(orth_mean, 4),
                "specificity_ratio": round(ratio, 3),
                "control_max_abs_dlogprob": round(control_max_abs, 4),
                "PASS": passed,
            }
            print(f"  L{L:>2}: dk Δlp {dknow_dlp:+7.3f} | rand_max {rand_max:+.3f} orth {orth_mean:+.3f} "
                  f"| ratio {ratio:7.2f} | ctrl {control_max_abs:.3f} | {'PASS' if passed else 'fail'} "
                  f"| demoted {rows_by_L[L]['frac_demoted_dknow']:.0%}", flush=True)
            write("in_progress")

    # ---- pre-registered verdict (rule frozen above; just apply it) ----
    def passed_at(L):
        return L in rows_by_L and rows_by_L[L]["PASS"]
    band = sorted(L for L in rows_by_L if rows_by_L[L]["PASS"])
    l11_pass = passed_at(11)
    l11_contiguous = l11_pass and (passed_at(10) or passed_at(12))
    survives = bool(l11_contiguous)
    # contiguous run containing 11
    contig = []
    if l11_pass:
        lo = 11
        while lo - 1 in band:
            lo -= 1
        hi = 11
        while hi + 1 in band:
            hi += 1
        contig = list(range(lo, hi + 1))
    peak = max(rows_by_L.values(), key=lambda r: r["dknow_dlogprob"])
    verdict = {
        "n_pass_layers": len(band), "pass_layers": band,
        "L11_PASS": l11_pass, "L11_in_contiguous_band_ge2": l11_contiguous,
        "L11_contiguous_run": contig,
        "peak_layer": peak["layer"], "peak_dknow_dlogprob": peak["dknow_dlogprob"],
        "headline": ("SURVIVES — L11 is causal and sits in a contiguous active band "
                     f"{contig} (peak L{peak['layer']} {peak['dknow_dlogprob']:+.1f} nats). "
                     "O4.3's necessity claim is not a single-layer selection artifact."
                     if survives else
                     "DOWNGRADE — L11 is not in a contiguous active band; O4.3 should be "
                     "restated as 'layer-specific, possibly selected'."),
        "headline_survives": survives,
    }
    write("done", verdict)

    print("\n" + "=" * 72)
    print(f"O4.4 VERDICT — {model_id}")
    print("=" * 72)
    print(f"  PASS layers ({len(band)}): {band}")
    print(f"  L11 PASS: {l11_pass} | contiguous band incl. L11: {contig}")
    print(f"  peak necessity: L{peak['layer']} (Δlog-prob {peak['dknow_dlogprob']:+.1f} nats)")
    print(f"  HEADLINE: {'SURVIVES' if survives else 'DOWNGRADE'}")
    print(f"  {verdict['headline']}")
    print(f"  wrote {OUT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
