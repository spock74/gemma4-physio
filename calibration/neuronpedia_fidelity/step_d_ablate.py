"""STEP D stage 3 — LOCALIZED ABLATION vs feature null (core env, torch 2.11.0).

The method-level positive control at the logit level: on base E2B (raw clozes,
SAE on-distribution), projection-ablate the SELECTED entity feature's W_dec unit
direction at layer 28 / final position (identical machinery to E1c's d_know test:
ablating(), positions="last") and measure the gold-logit drop on the 20 VAL knowns.
Null = the 50 frequency-matched SAE features, same localization, same protocol.

Verdict framing (agreed): METHOD/APPARATUS positive at the logit level on BASE —
NOT a causal replication of d_know(-it). A null here is ambiguous (redundancy, cf.
d_refusal, or wrong feature) and is reported as such, not iterated to positive.
Top-2/3 train-diff features run as EXPLORATORY only (no verdict flip).

Run (CORE env):  python calibration/neuronpedia_fidelity/step_d_ablate.py
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
from gemma4_lab.interp.directions import ablating
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
    dirs = np.load(CAP / "step_d_directions.npz")
    sel = json.loads((RES / "step_d_selection.json").read_text())
    corpus = json.loads((Settings().data_dir / "eval" / "entity_knowledge_contrast.json").read_text())
    known = corpus["known"]
    k_va = d["k_va"].tolist()
    val_items = [known[i] for i in k_va]
    gold_ids = d["gold_ids"][k_va]
    med_rank = int(d["median_gold_rank"])

    gemma = GemmaLocal(Settings())
    gemma.model_id = BASE_MODEL
    rec = ActivationRecorder(gemma)
    layer_slice = rec.layers[LAYER:LAYER + 1]

    def gold_logit(prompt: str, gid: int) -> float:
        v = float(rec.next_token_logits(prompt, templated=False)[gid])
        if not math.isfinite(v):
            raise ValueError(f"non-finite gold logit for {prompt!r} — withheld")
        return v

    def mean_drop(direction: np.ndarray, clean: list[float]) -> float:
        t = torch.tensor(direction, dtype=torch.float32)
        drops = []
        for j, it in enumerate(val_items):
            with ablating(layer_slice, t, positions="last"):
                abl = gold_logit(it["prompt"], int(gold_ids[j]))
            drops.append(clean[j] - abl)
        return sum(drops) / len(drops)

    with logfire.span("calibration.step_d.ablate", layer=LAYER, n_val=len(val_items),
                      selected=int(dirs["selected"])):
        clean = [gold_logit(it["prompt"], int(gold_ids[j])) for j, it in enumerate(val_items)]
        print(f"  clean gold logits on VAL (n={len(clean)}): mean {np.mean(clean):+.2f}", flush=True)

        effect = mean_drop(dirs["sel_dir"], clean)
        print(f"  selected feature {int(dirs['selected'])}: mean drop {effect:+.4f}", flush=True)

        null_drops = []
        for k in range(dirs["null_dirs"].shape[0]):
            null_drops.append(mean_drop(dirs["null_dirs"][k], clean))
            if (k + 1) % 10 == 0:
                print(f"  [null] {k + 1}/{dirs['null_dirs'].shape[0]} features done", flush=True)

        pct = _percentile(effect, null_drops)
        p95 = _quantile(null_drops, 0.95)
        gate_pass = bool(effect > 0 and pct >= 95.0)

        result = {
            "experiment": "step_d_feature_ablation",
            "framing": "METHOD/APPARATUS positive at the LOGIT level on BASE E2B — not a "
                       "causal replication of d_know(-it)",
            "model": BASE_MODEL, "layer": LAYER, "positions": "last",
            "readout": "raw cloze (valid on base)",
            "recall_precondition_median_gold_rank": med_rank,
            "selected_feature": int(dirs["selected"]),
            "selection": sel["selection_split"],
            "n_val_known": len(val_items),
            "effect_mean_logit_drop": round(effect, 4),
            "null_spec": sel["null_spec"],
            "null_mean": round(float(np.mean(null_drops)), 4),
            "null_p95": round(p95, 4),
            "null_max": round(max(null_drops), 4),
            "percentile_in_null": round(pct, 2),
            "ratio_effect_over_p95": round(effect / p95, 2) if p95 > 1e-6 else None,
            "cos_dknow28_base_vs_selected_W_dec": sel["cos_dknow28_base_vs_selected_W_dec"],
            "null_distribution": [round(x, 4) for x in null_drops],
            "gate_d_pass": gate_pass,
            "verdict": ("PASS — a data-selected SAE feature, ablated at one point, specifically "
                        "drops factual-recall logits: the diff-of-means-style selection + "
                        "localized-ablation METHOD detects a causal-at-logit feature "
                        "(non-tautological, on-distribution)." if gate_pass else
                        "NULL/INCONCLUSIVE — no specific drop vs the feature null; ambiguous "
                        "between redundancy (cf. d_refusal) and wrong feature. Reported as is."),
        }
        RES.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = RES / f"step_d_results_{stamp}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"STEP D — feature {int(dirs['selected'])} ablation @ L{LAYER}/last (base E2B, VAL n={len(val_items)})")
    print("=" * 70)
    print(f"  effect drop {effect:+.4f} | null mean {np.mean(null_drops):+.4f} "
          f"p95 {p95:+.4f} max {max(null_drops):+.4f} | percentile {pct:.1f}")
    print(f"  recall precondition: median gold rank {med_rank}")
    print(f"  cos(d_know@28_base, W_dec) = {sel['cos_dknow28_base_vs_selected_W_dec']}")
    print(f"  GATE D: {'PASS' if gate_pass else 'NULL/INCONCLUSIVE'}")
    print(f"  wrote {out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
