"""E1c — positive/negative controls for the LOCALIZED necessity test.

See docs/research/02-positive-control-plan.md. E1's global-ablation necessity failed
its negative control (random ≈ d_know). That is ambiguous: is d_know truly not
causal, or can the test detect NO specific direction? This adds POSITIVE controls run
through the same localized pipeline:

  - d_unembed (apparatus): the unembedding row of the gold token, ablated at the FINAL
    layer / final position. Near-tautological — it MUST collapse the gold logit and sit
    in the extreme tail of a random null, or the plumbing is broken (GATE 1).
  - d_know (test): the entity-knowledge direction, ablated LOCALIZED (layer L*, final
    position) — the global test's blind spot. Read the verdict off the matrix.
  - d_refusal (method): in refusal_direction.py (behavioral readout).

Each direction reports its PERCENTILE within an N>=50 random-direction null at the
SAME localization (a p-value-like statistic), plus an orthogonal-direction marker.
-it only. Logfire span on the run; readouts reuse the assistant-prefill recall readout.

Run:  python -m gemma4_lab.interp.e1c_controls
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

import torch

from ..config import Settings
from .directions import ablating, diff_of_means_direction, unembedding_direction
from .entity_knowledge import RECALL_INSTRUCTION, _answer_token_id, _split_indices
from .recorder import ActivationRecorder, contaminated_layers_from_residuals


def _random_unit(d: int, g: torch.Generator) -> torch.Tensor:
    r = torch.randn(d, generator=g)
    return r / (r.norm() + 1e-8)


def _orthogonalize(r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    o = r - (r @ d) * d
    return o / (o.norm() + 1e-8)


def _percentile(value: float, null: list[float]) -> float:
    """Percentile of `value` within `null` (% of null at-or-below; ties half-weight)."""
    n = len(null)
    below = sum(1 for x in null if x < value)
    ties = sum(1 for x in null if x == value)
    return 100.0 * (below + 0.5 * ties) / n if n else 0.0


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    pos = q * (len(s) - 1)
    lo = int(pos)
    frac = pos - lo
    return s[lo] if lo + 1 >= len(s) else s[lo] * (1 - frac) + s[lo + 1] * frac


def _extract_d_know(
    rec: ActivationRecorder, known: list[dict], unknown: list[dict], instr: str,
    layer: int, val_frac: float, split_seed: int,
) -> tuple[torch.Tensor | None, list[dict], dict[int, int]]:
    """Reproduce E1's held-out d_know: capture assistant-prefill last-token residuals,
    stratified TRAIN/VAL split (same seeds as entity_knowledge), diff-of-means on TRAIN
    at `layer`. Returns (d_know | None-if-layer-contaminated, val_known_items,
    contaminated_layers) — never silently extracts from a contaminated layer."""
    cands = list(range(rec.n_layers))
    kres = [rec.last_token_residuals(instr, cands, True, it["prompt"]) for it in known]
    ures = [rec.last_token_residuals(instr, cands, True, it["prompt"]) for it in unknown]
    contaminated = contaminated_layers_from_residuals(kres + ures)
    k_tr, k_va = _split_indices(len(known), val_frac, split_seed)
    u_tr, _ = _split_indices(len(unknown), val_frac, split_seed + 1)
    if layer in contaminated:
        return None, [known[i] for i in k_va], contaminated
    d_know = diff_of_means_direction(
        [kres[i][layer] for i in k_tr], [ures[i][layer] for i in u_tr]
    )
    return d_know, [known[i] for i in k_va], contaminated


def _necessity_with_null(
    rec: ActivationRecorder, items: list[dict], instr: str,
    layer_slice: slice, positions: str | list[int] | None,
    direction_for: Any, n_random: int, seed: int,
) -> dict:
    """Ablate a (possibly per-item) direction LOCALIZED and compare its mean gold-logit
    drop to an N-random null at the SAME localization (+ an orthogonal marker)."""
    tok_ids = [_answer_token_id(rec.tokenizer, it["answer"]) for it in items]
    clean_gold = [
        float(rec.next_token_logits(instr, True, it["prompt"])[tid])
        for it, tid in zip(items, tok_ids, strict=True)
    ]
    bad = [items[i]["answer"] for i, g in enumerate(clean_gold) if not math.isfinite(g)]
    if bad:
        raise ValueError(f"e1c: non-finite CLEAN gold logits for items {bad} — readout "
                         "contaminated; verdict withheld (see interp.numerical_health).")

    def mean_drop(dirs: list[torch.Tensor]) -> float:
        drops = []
        for i, it in enumerate(items):
            with ablating(rec.layers[layer_slice], dirs[i], positions=positions):
                abl = float(rec.next_token_logits(instr, True, it["prompt"])[tok_ids[i]])
            if not math.isfinite(abl):
                raise ValueError(f"e1c: non-finite ABLATED gold logit for {it['answer']!r} — "
                                 "readout contaminated; verdict withheld.")
            drops.append(clean_gold[i] - abl)
        return sum(drops) / len(drops)

    real_dirs = [direction_for(i, it, tok_ids[i]) for i, it in enumerate(items)]
    effect = mean_drop(real_dirs)

    d_model = int(real_dirs[0].shape[0])
    g = torch.Generator().manual_seed(seed)
    null = [mean_drop([_random_unit(d_model, g)] * len(items)) for _ in range(n_random)]
    orth_effect = mean_drop([_orthogonalize(_random_unit(d_model, g), real_dirs[i]) for i in range(len(items))])

    p95 = _quantile(null, 0.95)
    return {
        "effect_mean_logit_drop": round(effect, 3),
        "null_mean": round(sum(null) / len(null), 3),
        "null_p95": round(p95, 3),
        "null_max": round(max(null), 3),
        "orth_mean_logit_drop": round(orth_effect, 3),
        "percentile_in_null": round(_percentile(effect, null), 2),
        "ratio_effect_over_p95": round(effect / p95, 2) if p95 > 1e-6 else None,
        "n_random": n_random,
        "null_distribution": [round(x, 3) for x in null],
    }


def run(
    settings: Settings | None = None,
    corpus_path: str | Path | None = None,
    layer: int = 26,
    n_random: int = 50,
    val_frac: float = 0.5,
    split_seed: int = 0,
    out_dir: str | Path | None = None,
) -> dict:
    """Passo 1 (d_unembed, apparatus) + Passo 3 (d_know, test), both localized."""
    import logfire

    from .. import observability
    from ..inference.hf_local import GemmaLocal

    observability.setup()
    settings = settings or Settings()
    rec = ActivationRecorder(GemmaLocal(settings))
    corpus_path = Path(corpus_path or (settings.data_dir / "eval" / "entity_knowledge_contrast.json"))
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    known, unknown = corpus["known"], corpus["unknown"]

    with logfire.span("interp.e1c_controls.run", n_random=n_random, layer=layer):
        d_know, val_known, contaminated = _extract_d_know(
            rec, known, unknown, RECALL_INSTRUCTION, layer, val_frac, split_seed
        )
        numerical_health = {
            "contaminated_layers": sorted(contaminated),
            "policy": "withhold any control whose target layer is contaminated; no winsorizing",
            "winsorize": False,
        }
        model = rec.model
        last = rec.n_layers - 1

        # Passo 1 — apparatus positive: d_unembed at FINAL layer, final position.
        if last in contaminated:
            d_unembed: dict = {"verdict": "withheld",
                               "reason": f"final layer {last} contaminated (non-finite activations)"}
            print(f"  !! d_unembed control WITHHELD — {d_unembed['reason']}", flush=True)
        else:
            d_unembed = _necessity_with_null(
                rec, val_known, RECALL_INSTRUCTION, slice(last, last + 1), "last",
                direction_for=lambda i, it, tid: unembedding_direction(model, tid),
                n_random=n_random, seed=split_seed + 200,
            )
            _g1 = d_unembed["effect_mean_logit_drop"] > 0 and d_unembed["percentile_in_null"] >= 99.0
            print(f"  [interim] d_unembed drop={d_unembed['effect_mean_logit_drop']:+.2f} "
                  f"percentile={d_unembed['percentile_in_null']:.2f} -> GATE 1 "
                  f"{'PASS' if _g1 else 'FAIL (apparatus broken — stop)'}", flush=True)

        gate1_pass = bool(
            "effect_mean_logit_drop" in d_unembed
            and d_unembed["effect_mean_logit_drop"] > 0
            and d_unembed["percentile_in_null"] >= 99.0
        )

        # Passo 3 — test: d_know LOCALIZED at layer L*, final position. Only
        # interpretable if the apparatus positive passed — on GATE 1 failure we
        # abort early (saves ~2h of compute that could not be read either way).
        if not gate1_pass:
            d_know_loc: dict = {
                "verdict": "not_run",
                "reason": "GATE 1 failed/withheld — apparatus cannot demonstrably detect a "
                          "known direction; fix unembedding_direction/hooks and re-run",
            }
            print("  !! GATE 1 not passed — aborting before d_know (uninterpretable either way)",
                  flush=True)
        elif d_know is None:
            d_know_loc = {"verdict": "withheld",
                          "reason": f"extraction layer {layer} contaminated "
                                    f"({contaminated.get(layer, 0)} prompt(s) affected)"}
            print(f"  !! d_know control WITHHELD — {d_know_loc['reason']}", flush=True)
        else:
            d_know_loc = _necessity_with_null(
                rec, val_known, RECALL_INSTRUCTION, slice(layer, layer + 1), "last",
                direction_for=lambda i, it, tid: d_know,
                n_random=n_random, seed=split_seed + 300,
            )
        result = {
            "experiment": "e1c_controls",
            "model_variant": settings.model_variant,
            "n_val_known": len(val_known),
            "layer_d_know": layer,
            "layer_d_unembed": last,
            "gate1_apparatus_pass": gate1_pass,
            "numerical_health": numerical_health,
            "d_unembed": d_unembed,
            "d_know_localized": d_know_loc,
        }
        out_dir = Path(out_dir or (settings.data_dir / "eval" / "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"e1c_controls_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def line(name: str, r: dict) -> str:
        if "verdict" in r:  # withheld / not_run
            return f"  {name:10}: {r['verdict'].upper()} — {r['reason']}"
        return (f"  {name:10}: drop={r['effect_mean_logit_drop']:+8.2f}  "
                f"percentile={r['percentile_in_null']:6.2f}  ratio/p95={r['ratio_effect_over_p95']}  "
                f"(null mean {r['null_mean']:+.2f}, p95 {r['null_p95']:+.2f}, max {r['null_max']:+.2f}; "
                f"orth {r['orth_mean_logit_drop']:+.2f})")

    print("\n" + "=" * 72)
    print(f"E1c CONTROLS — localized necessity vs N={n_random} random null (held-out, n={len(val_known)})")
    print("=" * 72)
    print(line("d_unembed", d_unembed) + "   <- apparatus positive (must be ~100 pct)")
    print(line("d_know", d_know_loc) + "   <- the test")
    print(f"  GATE 1 (apparatus): {'PASS' if gate1_pass else 'FAIL — plumbing broken, stop'}")
    print(f"  wrote {out_path}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    run()
