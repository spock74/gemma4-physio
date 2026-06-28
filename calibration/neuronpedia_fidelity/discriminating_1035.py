"""STEP #1 (doc 04) — discriminating control for feature 1035 (core env).

Question: is 1035's one-point ablation SPECIFIC to factual recall, or generic
destruction of next-token coherence? (1035 is always-on/entity-independent — the
typical generic-channel profile; this control resolves the demoted claim
"carries this logit (recall-specificity uncontrolled)" one way or the other.)

The knife does NOT change: base E2B, projection-ablation of unit W_dec[1035] at
layer 28 / last position — exactly the STEP D2 protocol (direction loaded from
captures/step_d2_directions.npz; no SAE at runtime).

Context sets (raw text, tokenizer prepends BOS, no chat template):
  RECALL        = the 20 VAL known clozes (held-out: selection used TRAIN).
  NONREC_UNKNOWN= the 20 VAL unknown/fictional clozes — structure-matched (same
                  corpus family/shape), next token is confabulation, no fact.
  NONREC_GENERIC= 20 neutral narrative continuations (similar length), next token
                  is generic syntax/continuation; written blind to 1035's behavior.

Per context: CLEAN and ABLATED full next-token distributions; KL(clean||ablated),
entropies, Δentropy, top-1 change. RECALL also re-checks the gold drop (sanity:
must reproduce ~+20.7).

PRE-REGISTERED READING (fixed before running; mirrors 02-results spec):
  - "recall-specific":  median KL_recall >= 5x median KL of BOTH non-recall sets
                        AND top-1 change-rate exceeds both by >= 50 pp.
                        -> report claim upgrades to "recall-specific".
  - "generic-channel":  median KL_recall <= 2x median KL_generic (hurts both
                        alike) -> claim becomes "generic coherence channel;
                        'ablates recall' is a special case of breaking generation".
  - "answer-commit":    RECALL ~ UNKNOWN (within 2x) and both >= 5x GENERIC ->
                        the channel commits to producing a SPECIFIC answer
                        (confabulation included), not factual recall per se.
  - anything else: mixed — report the numbers as-is, no upgrade.
Either outcome is reportable; nothing is tuned after seeing results.

Output: data/eval/results/feature1035_discriminating_<stamp>.json
Run (CORE env):  python calibration/neuronpedia_fidelity/discriminating_1035.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from gemma4_lab.config import Settings
from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.interp.directions import ablating, rank_auc
from gemma4_lab.interp.recorder import ActivationRecorder

BASE_MODEL = "google/gemma-4-E2B"
LAYER = 28
HERE = Path(__file__).resolve().parent

GENERIC_PROMPTS = [
    "She opened the door and",
    "After lunch, they decided to",
    "The meeting was moved to",
    "He picked up the box and",
    "It started to rain, so we",
    "The children were playing in",
    "I was about to leave when",
    "The recipe says to mix the",
    "They walked slowly along the",
    "My favorite part of the day is",
    "The store was closed, so she",
    "He looked out the window and",
    "We packed our bags and",
    "The dog ran across the",
    "She wrote a quick note and",
    "The music was so loud that",
    "He turned off the lights and",
    "On the way home, they stopped",
    "The coffee was still too hot to",
    "I opened the book and began to",
]

PREREG = {
    "recall_specific": "median KL_recall >= 5x BOTH non-recall medians AND top-1 "
                       "change-rate exceeds both by >= 50 pp",
    "generic_channel": "median KL_recall <= 2x median KL_generic",
    "answer_commit": "RECALL ~ UNKNOWN (within 2x) and both >= 5x GENERIC",
    "otherwise": "mixed — report as-is, no upgrade",
}


def _kl_and_entropy(clean: torch.Tensor, abl: torch.Tensor) -> tuple[float, float, float]:
    lp, lq = torch.log_softmax(clean, -1), torch.log_softmax(abl, -1)
    p = lp.exp()
    kl = float((p * (lp - lq)).sum())
    h_clean = float(-(p * lp).sum())
    h_abl = float(-(lq.exp() * lq).sum())
    return kl, h_clean, h_abl


def main() -> int:
    import logfire

    from gemma4_lab import observability
    observability.setup()

    d = np.load(HERE / "captures" / "step_d_data.npz")
    dirs = np.load(HERE / "captures" / "step_d2_directions.npz")
    settings = Settings()
    corpus = json.loads((settings.data_dir / "eval" / "entity_knowledge_contrast.json").read_text())
    k_va, u_va = d["k_va"].tolist(), d["u_va"].tolist()
    recall_items = [corpus["known"][i] for i in k_va]
    unknown_prompts = [corpus["unknown"][i]["prompt"] for i in u_va]
    gold_ids = d["gold_ids"][k_va]
    t_dir = torch.tensor(dirs["target_dir"], dtype=torch.float32)
    target = int(dirs["target"])

    gemma = GemmaLocal(settings)
    gemma.model_id = BASE_MODEL
    rec = ActivationRecorder(gemma)
    layer_slice = rec.layers[LAYER:LAYER + 1]
    tok = rec.tokenizer

    def both_logits(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        clean = rec.next_token_logits(prompt, templated=False)
        with ablating(layer_slice, t_dir, positions="last"):
            abl = rec.next_token_logits(prompt, templated=False)
        if not (torch.isfinite(clean).all() and torch.isfinite(abl).all()):
            raise ValueError(f"non-finite logits for {prompt!r} — withheld")
        return clean, abl

    def run_set(name: str, prompts: list[str], golds: np.ndarray | None) -> list[dict]:
        rows = []
        for j, p in enumerate(prompts):
            with logfire.span("calibration.disc1035.context", set=name, i=j):
                clean, abl = both_logits(p)
            kl, h_c, h_a = _kl_and_entropy(clean, abl)
            t1c, t1a = int(clean.argmax()), int(abl.argmax())
            row = {
                "prompt": p, "kl": round(kl, 4),
                "entropy_clean": round(h_c, 4), "entropy_ablated": round(h_a, 4),
                "d_entropy": round(h_a - h_c, 4),
                "top1_changed": t1c != t1a,
                "top1_clean": tok.decode([t1c]), "top1_ablated": tok.decode([t1a]),
            }
            if golds is not None:
                g = int(golds[j])
                row["gold_drop"] = round(float(clean[g]) - float(abl[g]), 3)
            rows.append(row)
            if (j + 1) % 10 == 0:
                print(f"  [{name}] {j + 1}/{len(prompts)}", flush=True)
        return rows

    with logfire.span("calibration.disc1035.run", target=target, layer=LAYER):
        rec_rows = run_set("recall", [it["prompt"] for it in recall_items], gold_ids)
        unk_rows = run_set("nonrec_unknown", unknown_prompts, None)
        gen_rows = run_set("nonrec_generic", GENERIC_PROMPTS, None)

    def agg(rows: list[dict]) -> dict:
        kls = [r["kl"] for r in rows]
        return {
            "n": len(rows),
            "kl_median": round(median(kls), 4), "kl_mean": round(mean(kls), 4),
            "kl_min": round(min(kls), 4), "kl_max": round(max(kls), 4),
            "top1_change_rate": round(sum(r["top1_changed"] for r in rows) / len(rows), 3),
            "d_entropy_median": round(median(r["d_entropy"] for r in rows), 4),
        }

    a_r, a_u, a_g = agg(rec_rows), agg(unk_rows), agg(gen_rows)
    gold_sanity = round(mean(r["gold_drop"] for r in rec_rows), 3)
    auc_r_vs_g = rank_auc([r["kl"] for r in rec_rows] + [r["kl"] for r in gen_rows],
                          ["yes"] * len(rec_rows) + ["no"] * len(gen_rows))
    auc_r_vs_u = rank_auc([r["kl"] for r in rec_rows] + [r["kl"] for r in unk_rows],
                          ["yes"] * len(rec_rows) + ["no"] * len(unk_rows))

    # --- pre-registered verdict (rules above; evaluated mechanically) ---------
    r, u, g = a_r["kl_median"], a_u["kl_median"], a_g["kl_median"]
    t1r, t1u, t1g = a_r["top1_change_rate"], a_u["top1_change_rate"], a_g["top1_change_rate"]
    eps = 1e-9
    if r >= 5 * max(u, eps) and r >= 5 * max(g, eps) and (t1r - max(t1u, t1g)) >= 0.5:
        verdict = "recall-specific"
    elif r <= 2 * max(g, eps):
        verdict = "generic-channel"
    elif (max(r, u) / max(min(r, u), eps)) <= 2 and min(r, u) >= 5 * max(g, eps):
        verdict = "answer-commit"
    else:
        verdict = "mixed"

    result = {
        "experiment": "feature1035_discriminating_control",
        "doc04_step": 1,
        "model": BASE_MODEL, "layer": LAYER, "positions": "last",
        "knife": "projection-ablation of unit W_dec[1035], identical to STEP D2",
        "preregistered_rules": PREREG,
        "gold_drop_sanity": {"mean": gold_sanity, "expected": "~+20.68 (STEP D2)"},
        "aggregates": {"recall": a_r, "nonrec_unknown": a_u, "nonrec_generic": a_g},
        "effect_sizes": {
            "kl_median_ratio_recall_over_generic": round(r / max(g, eps), 2),
            "kl_median_ratio_recall_over_unknown": round(r / max(u, eps), 2),
            "rank_auc_kl_recall_vs_generic": round(auc_r_vs_g, 4) if auc_r_vs_g else None,
            "rank_auc_kl_recall_vs_unknown": round(auc_r_vs_u, 4) if auc_r_vs_u else None,
        },
        "verdict": verdict,
        "numerical_health": {"all_logits_finite": True,
                             "policy": "non-finite readouts raise (fail-loud)"},
        "contexts": {"recall": rec_rows, "nonrec_unknown": unk_rows,
                     "nonrec_generic": gen_rows},
    }
    out_dir = settings.data_dir / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"feature1035_discriminating_{stamp}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"STEP #1 — 1035 discriminating control (base E2B, L{LAYER}/last, D2 knife)")
    print("=" * 72)
    print(f"  gold-drop sanity: {gold_sanity:+.2f} (expect ~+20.68)")
    print(f"  KL median   — recall {r:.3f} | unknown {u:.3f} | generic {g:.3f}")
    print(f"  top1 change — recall {t1r:.0%} | unknown {t1u:.0%} | generic {t1g:.0%}")
    print(f"  ratios: r/g {result['effect_sizes']['kl_median_ratio_recall_over_generic']}x, "
          f"r/u {result['effect_sizes']['kl_median_ratio_recall_over_unknown']}x | "
          f"AUC r-vs-g {result['effect_sizes']['rank_auc_kl_recall_vs_generic']}")
    print(f"  PRE-REGISTERED VERDICT: {verdict.upper()}")
    print(f"  wrote {out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
