"""O4.7 — sharpness control: rank-based necessity on BASE gemma-3-4b-pt.

O4.6's necessity gap (base recall KL peak 19 vs -it 55) is partly confoundable by
SHARPNESS: -it's output distributions are more peaked, so the same ablation moves
them more -> larger KL, independent of how causal d_know is. Gold-RANK demotion is
scale/sharpness-invariant: does ablating d_know push the gold token DOWN the ranking?

Same base model, raw-cloze readout, intervention reused VERBATIM
(o4_necessity.necessity_readouts returns clean_rank / abl_rank). d_know refit per layer
on the seed-0 TRAIN split; eval on VAL known. Matched orthogonal control per layer.

Reference (-it, O4.4): ablating d_know demotes the gold token in ~100% of items across
the strong band (L11 65%, L9/L16/L18/L20 100%).

PRE-REGISTERED (frozen; written before results):
  per layer: frac_demoted (abl_rank > clean_rank) and frac_lost_top1 (clean rank 0 -> >0),
  d_know vs orthogonal control.
  Verdict on max-over-candidate-layers frac_demoted (d_know):
    SHARPNESS_CONFOUND        iff >= 0.8  (necessity IS present on base; the KL gap was
                                           sharpness -> revise O4.6 toward "present on both")
    NECESSITY_GENUINELY_WEAKER iff < 0.4  (gold barely demoted on base -> O4.6 stands)
    PARTIAL                    otherwise
  Orthogonal control must stay low (frac_demoted_orth < 0.2) for any positive read.

Run (CORE env):  python calibration/o4_entity_knowledge/o4_7_base_rank_necessity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from o4_6_base_control import CANDIDATE_LAYERS, load_base, raw_inputs  # base loader + raw readout
from o4_necessity import locate_layers, logits_of, necessity_readouts, split
from gemma4_lab.interp.directions import diff_of_means_direction

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "o4_7_base_rank_necessity.json"
CORPUS = Path("data/eval/entity_knowledge_contrast.json")
RNG_SEED = 100
SHARPNESS_CONFOUND_GE = 0.8
GENUINELY_WEAKER_LT = 0.4
ORTH_MAX_DEMOTE = 0.2


def all_layer_resid(model, layers, inputs) -> dict:
    import logfire
    sink = {}

    def mk(i):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            sink[i] = h[0, -1, :].detach().float().cpu()
        return hook
    hs = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    try:
        with logfire.span("o4_7.capture_all_layers"):
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
    prereg = {
        "model": "google/gemma-3-4b-pt (BASE, raw cloze)",
        "metric": "rank-based necessity (sharpness-invariant): frac_demoted / frac_lost_top1 under d_know ablation",
        "intervention": "reused verbatim from o4_necessity.necessity_readouts",
        "candidate_layers": CANDIDATE_LAYERS, "control": "orthogonal per layer",
        "it_reference": "O4.4: ~100% gold demoted across the strong band (L11 65%, L9/16/18/20 100%)",
        "verdict_rule": (f"on max-over-layers frac_demoted(d_know): SHARPNESS_CONFOUND iff >= {SHARPNESS_CONFOUND_GE} "
                         f"(necessity present on base; O4.6 KL gap was sharpness); NECESSITY_GENUINELY_WEAKER iff "
                         f"< {GENUINELY_WEAKER_LT} (O4.6 regime-dependent stands); else PARTIAL. Requires orth control "
                         f"frac_demoted < {ORTH_MAX_DEMOTE}."),
    }
    OUT.write_text(json.dumps({"objective": "O4.7", "status": "prereg_written", "prereg": prereg}, indent=2),
                   encoding="utf-8")

    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)

    tok, model = load_base()
    layers = locate_layers(model)
    print(f"loaded base: {len(layers)} layers; rank-based necessity (raw cloze)\n", flush=True)

    k_tr_res = [all_layer_resid(model, layers, raw_inputs(tok, known[i]["prompt"])) for i in k_tr]
    u_tr_res = [all_layer_resid(model, layers, raw_inputs(tok, unknown[i]["prompt"])) for i in u_tr]
    d_know = {L: diff_of_means_direction([r[L] for r in k_tr_res], [r[L] for r in u_tr_res]) for L in CANDIDATE_LAYERS}

    g = torch.Generator().manual_seed(RNG_SEED)
    d_model = int(d_know[CANDIDATE_LAYERS[0]].shape[0])
    orth = {}
    for L in CANDIDATE_LAYERS:
        rr = torch.randn(d_model, generator=g)
        o = rr - (rr @ d_know[L]) * d_know[L]
        orth[L] = o / o.norm()

    val = []
    for i in k_va:
        it = known[i]
        inp = raw_inputs(tok, it["prompt"])
        gold = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        val.append((inp, gold, logits_of(model, inp)))

    rows = []
    for L in CANDIDATE_LAYERS:
        with logfire.span("o4_7.layer", layer=L):
            dk_dem = dk_lost = orth_dem = 0
            abl_ranks = []
            for inp, gold, clean in val:
                dk = necessity_readouts(model, layers, L, inp, gold, clean, d_know[L])
                ot = necessity_readouts(model, layers, L, inp, gold, clean, orth[L])
                if dk["abl_rank"] > dk["clean_rank"]:
                    dk_dem += 1
                if dk["clean_rank"] == 0 and dk["abl_rank"] > 0:
                    dk_lost += 1
                if ot["abl_rank"] > ot["clean_rank"]:
                    orth_dem += 1
                abl_ranks.append(dk["abl_rank"])
            n = len(val)
            rows.append({"layer": L, "frac_demoted_dknow": round(dk_dem / n, 3),
                         "frac_lost_top1_dknow": round(dk_lost / n, 3),
                         "frac_demoted_orth": round(orth_dem / n, 3),
                         "mean_abl_rank_dknow": round(sum(abl_ranks) / n, 2)})
            print(f"  L{L:>2}: d_know demoted {dk_dem/n:.0%} lost-top1 {dk_lost/n:.0%}  | orth demoted {orth_dem/n:.0%}  | mean abl rank {sum(abl_ranks)/n:.1f}", flush=True)

    best = max(rows, key=lambda r: r["frac_demoted_dknow"])
    max_dem = best["frac_demoted_dknow"]
    max_orth = max(r["frac_demoted_orth"] for r in rows)
    if max_dem >= SHARPNESS_CONFOUND_GE and max_orth < ORTH_MAX_DEMOTE:
        verdict = "SHARPNESS_CONFOUND"
    elif max_dem < GENUINELY_WEAKER_LT:
        verdict = "NECESSITY_GENUINELY_WEAKER"
    else:
        verdict = "PARTIAL"
    interp = {
        "SHARPNESS_CONFOUND": ("Ablating d_know DOES demote the gold token on base (rank-based), comparably to -it -> "
                               "the O4.6 KL gap (19 vs 55) was largely a sharpness artifact; the causal necessity is "
                               "present on base too. Revise O4.6: necessity present on both, -it's larger KL reflects "
                               "sharper -it distributions, not (only) more causality."),
        "NECESSITY_GENUINELY_WEAKER": ("Ablating d_know barely demotes the gold token on base -> the necessity is "
                                       "genuinely weaker on base, not a sharpness artifact. O4.6's regime-dependent "
                                       "conclusion stands and strengthens."),
        "PARTIAL": "Intermediate gold demotion on base -> necessity is partly present; the regime amplifies it but does not create it.",
    }[verdict]

    out = {"objective": "O4.7", "status": "done", "model_id": "google/gemma-3-4b-pt", "prereg": prereg,
           "rows": rows, "max_frac_demoted_dknow": max_dem, "max_frac_demoted_orth": round(max_orth, 3),
           "best_layer": best["layer"], "verdict": verdict, "interpretation": interp}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("O4.7 RANK-BASED NECESSITY ON BASE (sharpness control)")
    print("=" * 72)
    print(f"  max d_know gold-demotion on base = {max_dem:.0%} @L{best['layer']}  (orth max {max_orth:.0%})")
    print(f"  -it reference: ~100% demoted across the strong band")
    print(f"  VERDICT: {verdict}")
    print(f"  {interp}")
    print(f"  wrote {OUT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
