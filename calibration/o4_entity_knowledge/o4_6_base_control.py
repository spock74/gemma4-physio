"""O4.6 — base-vs-it control: is the O4.5 GENERIC verdict a -it/instructed-readout
artifact, or intrinsic to the known/unknown axis?

Runs the O4.5 recall-specificity test on the BASE checkpoint google/gemma-3-4b-pt
with its NATURAL readout — RAW CLOZE, no chat template (base completes clozes
directly; the assistant-prefill workaround is an -it-only need). The intervention is
reused VERBATIM from o4_necessity.necessity_readouts (KL field). d_know is the
known/unknown diff-of-means, refit per layer on the seed-0 TRAIN split; eval on VAL.

This bundles two changes vs the -it run (model: base-vs-it AND readout: raw-vs-template)
— it does not isolate which. That is the pre-registered caveat; a follow-up (template
vs raw on one model) would separate them. But it is the decisive first cut:

PRE-REGISTERED comparison (written to the output JSON before results; frozen):
  - readout-validity gate (base must recall via raw cloze): median gold rank < 5 AND
    top-5 >= 0.7; else flag the comparison as readout-limited.
  - necessity gate (d_know must be causal on base): a band of >=3 candidate layers with
    in-recall median KL >= 20 AND controls inert there (< 1 nat); else "no causal band
    on base — specificity comparison moot".
  - primary = in-band median KL_recall / in-band median KL_nonrecall (d_know), on base.
  - verdict vs the -it O4.5 ratio (read from o4_5_recall_specificity.json, not hardcoded):
      CONFOUND_OF_IT   iff base_ratio > 3   (recall-specific on base, generic on -it)
      INTRINSIC        iff base_ratio < 1.5 (generic on base too)
      PARTIAL          otherwise
  A null / intrinsic verdict is a valid result; thresholds are not tuned post-hoc.

Run (CORE env):  python calibration/o4_entity_knowledge/o4_6_base_control.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoTokenizer

from o4_necessity import locate_layers, logits_of, necessity_readouts, split  # reuse intervention + helpers
from gemma4_lab.config import hf_token_or_none
from gemma4_lab.interp.directions import diff_of_means_direction, projection, rank_auc

MODEL_ID = "google/gemma-3-4b-pt"        # BASE checkpoint
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "o4_6_base_control.json"
IT_O45 = HERE / "results" / "o4_5_recall_specificity.json"
CORPUS = Path("data/eval/entity_knowledge_contrast.json")
NONRECALL_JSON = Path("data/eval/nonrecall_knownfixed.json")
FLUENCY_JSON = Path("data/eval/fluency_neutral.json")

# ---- PRE-REGISTERED constants (frozen) ----
CANDIDATE_LAYERS = list(range(4, 31))    # mid-network, excludes early/final (as on -it)
N_RANDOM = 20
RNG_SEED = 100
NECESSITY_MIN_KL = 20.0                   # match the -it band threshold (dknow recall KL)
CONTROL_SMALL_KL = 1.0
PRIMARY_SPECIFIC = 3.0
PRIMARY_GENERIC = 1.5
READOUT_MEDIAN_RANK_MAX = 5
READOUT_TOP5_MIN = 0.7


def load_base():
    token = hf_token_or_none()
    if not token:
        raise RuntimeError(f"HF_TOKEN absent in config — {MODEL_ID} is gated (expected a token, found none)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, token=token, dtype=torch.bfloat16,
        attn_implementation="eager").to(DEVICE)
    model.eval()
    return tok, model


def raw_inputs(tok, text: str):
    """BASE readout: raw cloze, BOS prepended by the tokenizer, NO chat template."""
    return tok(text, return_tensors="pt").to(DEVICE)


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
        with logfire.span("o4_6.capture_all_layers"):
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

    if not IT_O45.exists():
        raise RuntimeError(f"-it O4.5 result missing at {IT_O45} (need its ratio to compare; found nothing)")
    it_ratio = json.loads(IT_O45.read_text())["verdict"]["primary_ratio_recall_over_nonrecall"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prereg = {
        "model": MODEL_ID, "readout": "raw cloze (no chat template) — base's natural readout",
        "intervention": "reused verbatim from o4_necessity.necessity_readouts (KL field)",
        "candidate_layers": CANDIDATE_LAYERS, "n_random": N_RANDOM,
        "necessity_min_kl": NECESSITY_MIN_KL, "control_small_kl": CONTROL_SMALL_KL,
        "primary_specific_gt": PRIMARY_SPECIFIC, "primary_generic_lt": PRIMARY_GENERIC,
        "readout_validity": {"median_gold_rank_lt": READOUT_MEDIAN_RANK_MAX, "top5_ge": READOUT_TOP5_MIN},
        "it_primary_ratio_for_comparison": it_ratio,
        "primary": "in-band median KL_recall / in-band median KL_nonrecall (d_know), on base",
        "verdict_rule": ("CONFOUND_OF_IT iff base_ratio>3 (recall-specific on base, generic on -it); "
                         "INTRINSIC iff base_ratio<1.5 (generic on base too); else PARTIAL"),
        "caveat": "bundles model(base-vs-it) AND readout(raw-vs-template); does not isolate which",
    }
    OUT.write_text(json.dumps({"objective": "O4.6", "status": "prereg_written", "prereg": prereg}, indent=2),
                   encoding="utf-8")

    tok, model = load_base()
    layers = locate_layers(model)
    print(f"loaded {MODEL_ID}: {len(layers)} layers (BASE, raw-cloze readout)\n", flush=True)

    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    nonrec = json.loads(NONRECALL_JSON.read_text())["items"]
    fluency = json.loads(FLUENCY_JSON.read_text())["items"]
    k_tr, k_va = split(len(known), 0)
    u_tr, u_va = split(len(unknown), 1)

    def gold_id(ans):
        return int(tok(" " + ans.strip(), add_special_tokens=False)["input_ids"][0])

    # --- readout validity: does BASE recall via raw cloze? ---
    ranks = []
    for it in known:
        lg = logits_of(model, raw_inputs(tok, it["prompt"]))
        ranks.append(int((lg > lg[gold_id(it["answer"])]).sum()))
    med_rank = sorted(ranks)[len(ranks) // 2]
    top5 = sum(1 for r in ranks if r < 5) / len(ranks)
    readout_ok = bool(med_rank < READOUT_MEDIAN_RANK_MAX and top5 >= READOUT_TOP5_MIN)
    print(f"readout-validity (raw cloze): median gold rank {med_rank}, top-5 {top5:.0%} -> {'OK' if readout_ok else 'WEAK'}", flush=True)

    # --- fit d_know per candidate layer on TRAIN (raw cloze), held-out AUC ---
    k_tr_res = [all_layer_resid(model, layers, raw_inputs(tok, known[i]["prompt"])) for i in k_tr]
    u_tr_res = [all_layer_resid(model, layers, raw_inputs(tok, unknown[i]["prompt"])) for i in u_tr]
    k_va_res = [all_layer_resid(model, layers, raw_inputs(tok, known[i]["prompt"])) for i in k_va]
    u_va_res = [all_layer_resid(model, layers, raw_inputs(tok, unknown[i]["prompt"])) for i in u_va]
    d_know = {L: diff_of_means_direction([r[L] for r in k_tr_res], [r[L] for r in u_tr_res]) for L in CANDIDATE_LAYERS}

    def auc_at(L):
        sc = [projection(r[L], d_know[L]) for r in k_va_res] + [projection(r[L], d_know[L]) for r in u_va_res]
        lab = ["yes"] * len(k_va_res) + ["no"] * len(u_va_res)
        a = rank_auc(sc, lab)
        return max(a, 1 - a)
    decodability = {L: round(auc_at(L), 4) for L in CANDIDATE_LAYERS}

    # --- random controls (shared) ---
    g = torch.Generator().manual_seed(RNG_SEED)
    d_model = int(d_know[CANDIDATE_LAYERS[0]].shape[0])
    rands = [(lambda r: r / r.norm())(torch.randn(d_model, generator=g)) for _ in range(N_RANDOM)]

    # --- condition inputs (raw cloze) on VAL ---
    recall_items = [raw_inputs(tok, known[i]["prompt"]) for i in k_va]
    nonrecall_items = [raw_inputs(tok, nonrec[i]["stem"]) for i in k_va]
    fluency_items = [raw_inputs(tok, it["stem"]) for it in fluency]

    def dknow_kl(items, Ls):
        cleans = [logits_of(model, inp) for inp in items]
        return {L: median([necessity_readouts(model, layers, L, inp, 0, cl, d_know[L])["kl"]
                            for inp, cl in zip(items, cleans)]) for L in Ls}, cleans

    # --- necessity band on base: recall d_know KL per candidate layer; band = KL >= 20 ---
    print("scanning recall d_know KL across candidate layers (necessity) ...", flush=True)
    with logfire.span("o4_6.recall_band_scan"):
        recall_kl_cand, recall_cleans = dknow_kl(recall_items, CANDIDATE_LAYERS)
    band = sorted(L for L in CANDIDATE_LAYERS if recall_kl_cand[L] >= NECESSITY_MIN_KL)
    print(f"  base causal band (recall KL >= {NECESSITY_MIN_KL}): {band}", flush=True)

    result_extra = {}
    if len(band) >= 3:
        control_layers = sorted({band[0], band[len(band) // 2], band[-1]})
        # controls (recall) at control layers -> inert?
        ctrl = {}
        for L in control_layers:
            per_dir = [[] for _ in range(N_RANDOM)]
            for inp, cl in zip(recall_items, recall_cleans):
                for j, rd in enumerate(rands):
                    per_dir[j].append(necessity_readouts(model, layers, L, inp, 0, cl, rd)["kl"])
            ctrl[L] = max(median(c) for c in per_dir)
        controls_inert = all(v < CONTROL_SMALL_KL for v in ctrl.values())

        with logfire.span("o4_6.nonrecall"):
            nonrecall_kl, _ = dknow_kl(nonrecall_items, band)
        with logfire.span("o4_6.fluency"):
            fluency_kl, _ = dknow_kl(fluency_items, band)

        recall_med = median([recall_kl_cand[L] for L in band])
        nonrecall_med = median(list(nonrecall_kl.values()))
        fluency_med = median(list(fluency_kl.values()))
        base_ratio = recall_med / nonrecall_med if nonrecall_med > 1e-9 else float("inf")
        fluency_frac = fluency_med / recall_med if recall_med > 1e-9 else float("inf")
        result_extra = {
            "band": band, "control_layers": control_layers,
            "control_max_kl_recall": {str(k): round(v, 4) for k, v in ctrl.items()},
            "controls_inert": controls_inert,
            "inband_median_KL": {"recall": round(recall_med, 4), "nonrecall": round(nonrecall_med, 4),
                                 "fluency": round(fluency_med, 4)},
            "base_primary_ratio": round(base_ratio, 3), "base_fluency_fraction": round(fluency_frac, 3),
        }
        if base_ratio > PRIMARY_SPECIFIC:
            verdict = "CONFOUND_OF_IT"
        elif base_ratio < PRIMARY_GENERIC:
            verdict = "INTRINSIC"
        else:
            verdict = "PARTIAL"
        if not readout_ok:
            verdict += "_READOUT_LIMITED"
    else:
        verdict = "NO_CAUSAL_BAND_ON_BASE"
        base_ratio = None

    interp = {
        "CONFOUND_OF_IT": "Recall-SPECIFIC on base but GENERIC on -it -> the O4.5 generic verdict is an "
                          "artifact of the -it instructed readout / fine-tuning. d_know is a recall gate "
                          "in the base model; the instruction apparatus makes it look load-bearing.",
        "INTRINSIC": "GENERIC on base too -> the load-bearing role of d_know is intrinsic to the known/unknown "
                     "axis, not a -it/readout artifact. O4.5's downgrade stands for base and -it alike.",
        "PARTIAL": "Base recall-specificity is intermediate -> the -it generic result is partly readout/fine-tuning "
                   "driven; not a clean confound nor fully intrinsic.",
        "NO_CAUSAL_BAND_ON_BASE": "No band where ablating d_know collapses recall on base (raw cloze) -> the "
                                  "necessity itself does not replicate on base with this readout; the specificity "
                                  "comparison is moot. Likely the known/unknown causal role is readout/-it dependent.",
    }.get(verdict.replace("_READOUT_LIMITED", ""), "")

    out = {"objective": "O4.6", "status": "done", "model_id": MODEL_ID, "prereg": prereg,
           "readout_validity": {"median_gold_rank": med_rank, "top5": round(top5, 3), "ok": readout_ok},
           "decodability_auc": decodability,
           "recall_kl_per_candidate_layer": {str(k): round(v, 4) for k, v in recall_kl_cand.items()},
           **result_extra,
           "it_primary_ratio": it_ratio, "base_primary_ratio": base_ratio,
           "verdict": verdict, "interpretation": interp}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"O4.6 BASE-vs-IT CONTROL — {MODEL_ID} (raw cloze)")
    print("=" * 72)
    print(f"  readout-validity: median gold rank {med_rank}, top-5 {top5:.0%} ({'OK' if readout_ok else 'WEAK'})")
    print(f"  base causal band (recall KL>=20): {band}")
    if base_ratio is not None:
        ed = result_extra["inband_median_KL"]
        print(f"  in-band median KL  recall={ed['recall']}  nonrecall={ed['nonrecall']}  fluency={ed['fluency']}")
        print(f"  base primary ratio = {base_ratio:.2f}   (-it was {it_ratio})   controls_inert={result_extra['controls_inert']}")
    print(f"  VERDICT: {verdict}")
    print(f"  {interp}")
    print(f"  wrote {OUT}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
