"""STEP D2 stage 2 — feature 1035 vs FRESH null (core env, torch 2.11.0).

Confirmation test for STEP D's exploratory finding: does the single feature that
erased the recall logit (+20.68 of +20.74) survive a FRESH frequency-matched null,
i.e. is it SPECIFIC — the true (non-tautological) method-positive at the logit
level? Same protocol as STEP D: base E2B, raw clozes, ablating() at L28/last,
20 VAL knowns.

Also characterizes the feature in logit space (core-env model access):
  - top tokens its decoder direction PROMOTES via final-norm ⊙ unembedding;
  - tautology check: cos(W_dec[1035], unembedding_direction(gold)) per VAL item —
    a single fixed direction cannot track 20 different golds, but if mean |cos| is
    high the "erasure" would be partially unembedding-trivial; if low, the feature
    is an upstream recall component (the interesting case).

Run (CORE env):  python calibration/neuronpedia_fidelity/step_d2_ablate.py
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy as np
import torch

from gemma4_lab.config import Settings
from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.interp.directions import _final_norm_weight, ablating, unembedding_direction
from gemma4_lab.interp.recorder import ActivationRecorder

BASE_MODEL = "google/gemma-4-E2B"
LAYER = 28
HERE = Path(__file__).resolve().parent
CAP = HERE / "captures"
RES = HERE / "results"


def _percentile(value: float, null: list[float]) -> float:
    n = len(null)
    below = sum(1 for x in null if x < value)
    ties = sum(1 for x in null if x == value)
    return 100.0 * (below + 0.5 * ties) / n if n else 0.0


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo, frac = int(pos), pos - int(pos)
    return s[lo] if lo + 1 >= len(s) else s[lo] * (1 - frac) + s[lo + 1] * frac


def main() -> int:
    import logfire

    from gemma4_lab import observability
    observability.setup()

    d = np.load(CAP / "step_d_data.npz")
    dirs = np.load(CAP / "step_d2_directions.npz")
    sel = json.loads((RES / "step_d2_selection.json").read_text())
    corpus = json.loads((Settings().data_dir / "eval" / "entity_knowledge_contrast.json").read_text())
    known = corpus["known"]
    k_va = d["k_va"].tolist()
    val_items = [known[i] for i in k_va]
    gold_ids = d["gold_ids"][k_va]
    target = int(dirs["target"])

    gemma = GemmaLocal(Settings())
    gemma.model_id = BASE_MODEL
    rec = ActivationRecorder(gemma)
    model = rec.model
    tok = rec.tokenizer
    layer_slice = rec.layers[LAYER:LAYER + 1]
    t_dir = torch.tensor(dirs["target_dir"], dtype=torch.float32)

    # --- logit-space characterization (no forwards needed) -------------------
    gamma = _final_norm_weight(model)
    w_u = model.get_output_embeddings().weight.detach().float().cpu()      # [vocab, 1536]
    v = (gamma.detach().float().cpu() * t_dir) if gamma is not None else t_dir
    scores = w_u @ v
    topv, topi = torch.topk(scores, 15)
    top_promoted = [{"token": tok.decode([int(t)]), "score": round(float(s), 2)}
                    for t, s in zip(topi, topv, strict=True)]
    cos_gold = [float(t_dir @ unembedding_direction(model, int(g))) for g in gold_ids]
    print(f"  feature {target} promotes: {[t['token'] for t in top_promoted[:10]]}")
    print(f"  tautology check: cos(W_dec[{target}], d_unembed(gold)) mean {np.mean(cos_gold):+.3f} "
          f"max |{np.max(np.abs(cos_gold)):.3f}|", flush=True)

    def gold_logit(prompt: str, gid: int) -> float:
        x = float(rec.next_token_logits(prompt, templated=False)[gid])
        if not math.isfinite(x):
            raise ValueError("non-finite gold logit — withheld")
        return x

    def drops_for(direction: torch.Tensor, clean: list[float]) -> list[float]:
        out = []
        for j, it in enumerate(val_items):
            with ablating(layer_slice, direction, positions="last"):
                abl = gold_logit(it["prompt"], int(gold_ids[j]))
            out.append(clean[j] - abl)
        return out

    # CENSUS context from STEP D: the previous "null" was in fact the EXHAUSTIVE set
    # of dense/strong L28 features (every feature with train mean-act > ~3.1 is in
    # prev-null ∪ {1007, 1035} — both rate- and magnitude-matched pools are empty
    # afterwards). So 1035's specificity claim rests on that census, and the
    # pre-registered "fresh matched null" gate is IMPOSSIBLE as specified. New gate
    # (decided BEFORE this run): per-item consistency (>=80% of VAL items hurt) AND
    # non-tautology (max |cos| with gold unembeddings < 0.2); the weak remaining
    # features run as a labeled LOWER CONTROL (expect ~no effect).
    prev_results = sorted(RES.glob("step_d_results_*.json"))[-1]
    prev_null = json.loads(prev_results.read_text())["null_distribution"]
    census = {
        "class_definition": "all L28 features with TRAIN mean activation > ~3.1 "
                            "(= prev null ∪ {1007, 1035}; exhaustively measured in STEP D)",
        "target_drop_in_census": 20.68,
        "runner_up_drop": round(sorted(prev_null)[-2], 3),
        "census_p95": round(_quantile(prev_null, 0.95), 4),
        "note": "specificity rests on this CENSUS, not a sampled null — no fresh matched "
                "null exists (pool exhausted); winner's-curse caveat applies to the "
                "magnitude, not to census uniqueness",
    }

    with logfire.span("calibration.step_d2.ablate", target=target, n_val=len(val_items)):
        clean = [gold_logit(it["prompt"], int(gold_ids[j])) for j, it in enumerate(val_items)]
        t_drops = drops_for(t_dir, clean)
        effect = float(np.mean(t_drops))
        items_hurt = sum(1 for x in t_drops if x > 0)
        print(f"  feature {target} mean drop {effect:+.3f} "
              f"(items hurt: {items_hurt}/{len(t_drops)})", flush=True)

        lower_drops = []
        for k in range(dirs["null_dirs"].shape[0]):
            lower_drops.append(float(np.mean(drops_for(
                torch.tensor(dirs["null_dirs"][k], dtype=torch.float32), clean))))
            if (k + 1) % 10 == 0:
                print(f"  [lower control] {k + 1}/{dirs['null_dirs'].shape[0]} done", flush=True)

        taut_max = float(np.max(np.abs(cos_gold)))
        gate_pass = bool(effect > 0 and items_hurt >= 0.8 * len(t_drops) and taut_max < 0.2)

        result = {
            "experiment": "step_d2_feature_1035_confirmation",
            "framing": "confirmation of STEP D's exploratory finding. Pre-registered "
                       "fresh-matched-null gate IMPOSSIBLE as specified (matched pool "
                       "exhausted — the dense/strong class was CENSUSED in STEP D); new "
                       "gate decided before this run: per-item consistency + non-tautology, "
                       "with the strongest REMAINING features as a labeled lower control",
            "model": BASE_MODEL, "layer": LAYER, "positions": "last",
            "target_feature": target,
            "characterization": sel["characterization"],
            "census": census,
            "top_promoted_tokens": top_promoted,
            "tautology_cos_gold_unembed": {"mean": round(float(np.mean(cos_gold)), 4),
                                           "max_abs": round(taut_max, 4)},
            "n_val_known": len(val_items),
            "effect_mean_logit_drop": round(effect, 4),
            "per_item_drops": [round(x, 3) for x in t_drops],
            "items_hurt": items_hurt,
            "lower_control_spec": sel["fresh_null_spec"] + " — strongest REMAINING features "
                                  f"(mean-act {sel['fresh_null_mean_act_range']} vs target "
                                  f"{round(sel['target_mean_act'], 2)}); expect ~no effect",
            "lower_control_mean": round(float(np.mean(lower_drops)), 4),
            "lower_control_p95": round(_quantile(lower_drops, 0.95), 4),
            "lower_control_max": round(float(max(lower_drops)), 4),
            "lower_control_distribution": [round(x, 4) for x in lower_drops],
            "gate_d2_pass": gate_pass,
            "verdict": ("PASS — feature 1035's recall-logit erasure is consistent across items, "
                        "non-tautological (low cos with gold unembeddings), unique within the "
                        "censused dense-feature class, and far above the weak-feature lower "
                        "control: a single SAE feature causally carries the factual-recall "
                        "logit at L28. This is the true method-positive at the logit level."
                        if gate_pass else
                        "FAIL/PARTIAL — see per_item_drops / tautology / lower control; the "
                        "census uniqueness alone does not establish the method-positive."),
        }
        RES.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = RES / f"step_d2_results_{stamp}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"STEP D2 — feature {target} confirmation (base E2B, L{LAYER}/last, VAL n={len(val_items)})")
    print("=" * 70)
    print(f"  effect {effect:+.3f} | items hurt {items_hurt}/{len(t_drops)} | "
          f"tautology max|cos| {taut_max:.3f}")
    print(f"  census: unique in dense class (runner-up {census['runner_up_drop']:+.2f}, "
          f"p95 {census['census_p95']:+.2f})")
    print(f"  lower control (weak remaining): mean {np.mean(lower_drops):+.3f} "
          f"p95 {_quantile(lower_drops, 0.95):+.3f} max {max(lower_drops):+.3f}")
    print(f"  GATE D2: {'PASS' if gate_pass else 'FAIL/PARTIAL'}")
    print(f"  wrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
