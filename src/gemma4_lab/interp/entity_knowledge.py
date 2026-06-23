"""E1 — Entity-knowledge direction (causal).

Anchor: Ferrando, Obeso, Rajamanoharan, Nanda 2024/ICLR 2025, "Do I Know This
Entity?" (arXiv:2411.14257). Entity-recognition directions causally gate
hallucination vs. knowledge-refusal. They discovered the feature with SAEs; we
extract the same axis with difference-of-means (SAE-free, M2-feasible).

Hypotheses:
    H1 necessity:   ablating d_know on a KNOWN cloze lowers the logit of the
                    correct answer token.
    H2 sufficiency: adding d_know on an UNKNOWN/fictional prompt lowers next-token
                    entropy (more confident confabulation).

Readout = next-token logits at the final position (NO generation), so the whole
experiment is forward-pass bound, not generation bound (~0.4 tok/s on M2).

This upgrades the sibling LISP repo's CORRELATIONAL drift metric to a CAUSAL
direction-ablation claim on the same should_know/cannot_know contrast.

Run:
    python -m gemma4_lab.interp.entity_knowledge
See ../../docs/research/01-experiment-plan.md for the full design.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from statistics import median, stdev
from typing import Any

import torch

from ..config import Settings
from .directions import (
    ablating,
    diff_of_means_direction,
    projection,
    rank_auc,
    steering,
)
from .recorder import ActivationRecorder, contaminated_layers_from_residuals


def _answer_token_id(tokenizer: Any, answer: str) -> int:
    """First token id of the answer as a continuation (leading space matters for
    SentencePiece/Gemma tokenizers: 'Paris' continues 'is ' as ' Paris'). A pilot
    simplification — multi-token answers are scored on their first token only."""
    ids = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Answer {answer!r} tokenized to empty sequence.")
    return int(ids[0])


def _entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return float(-(p * logp).sum())


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Reproducible (train, val) index split of range(n). Stratification is by
    calling this once per class, so each class keeps its proportion."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, round(n * val_frac))
    val = set(perm[:n_val])
    return [i for i in range(n) if i not in val], sorted(val)


def _random_unit(d: int, g: torch.Generator) -> torch.Tensor:
    r = torch.randn(d, generator=g)
    return r / (r.norm() + 1e-8)


# The -it model fed a raw cloze ECHOES the context (median gold-token rank ~600).
# Instructed in the user turn with the cloze stem prefilling its OWN turn, it
# completes the fact as the immediate next token (gold rank 0). This is the only
# readout under which the necessity/sufficiency causal tests are valid on -it; a
# base/-pt model would complete raw clozes directly (templated=False).
RECALL_INSTRUCTION = "Answer with the fact, continuing the sentence."


def run(
    gemma: Any = None,
    settings: Settings | None = None,
    corpus_path: str | Path | None = None,
    layer: int | None = None,
    steer_coeff: float = 8.0,
    templated: bool = True,
    instruction: str | None = None,
    val_frac: float = 0.5,
    n_random: int = 5,
    split_seed: int = 0,
    out_dir: str | Path | None = None,
) -> dict:
    """Extract the entity-knowledge direction, then test necessity and sufficiency
    causally via logit readouts. Returns the result dict and writes it to disk.

    templated=True (default) uses the assistant-prefill recall readout on -it;
    templated=False feeds the raw cloze (valid only on a base/-pt checkpoint).

    Held-out: layer + d_know are fit on a TRAIN split; separation AUC and the
    causal tests are reported on a disjoint VAL split (fixes selection circularity).
    Specificity: necessity ablates d_know AND n_random random unit directions AND
    one direction orthogonal to d_know — same ablating() across all layers — so a
    generic 1-D-ablation artifact is separated from real d_know necessity."""
    import logfire

    from .. import observability
    from ..inference.hf_local import GemmaLocal

    observability.setup()
    settings = settings or Settings()
    gemma = gemma or GemmaLocal(settings)
    rec = ActivationRecorder(gemma)

    corpus_path = Path(corpus_path or (settings.data_dir / "eval" / "entity_knowledge_contrast.json"))
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    known = corpus["known"]      # [{"prompt": "...", "answer": "Paris"}, ...]
    unknown = corpus["unknown"]  # [{"prompt": "..."}, ...]

    with logfire.span(
        "interp.entity_knowledge.run",
        n_known=len(known),
        n_unknown=len(unknown),
        templated=templated,
    ) as span:
        n_layers = rec.n_layers
        candidate_layers = list(range(n_layers))
        instr = instruction or RECALL_INSTRUCTION

        # Readout builders: on -it (templated) the stem prefills the model turn so
        # the answer is the immediate continuation; raw mode is for a base/-pt model.
        def capture(stem: str) -> dict[int, torch.Tensor]:
            if templated:
                return rec.last_token_residuals(instr, candidate_layers, True, stem)
            return rec.last_token_residuals(stem, candidate_layers, False)

        def read_logits(stem: str) -> torch.Tensor:
            if templated:
                return rec.next_token_logits(instr, True, stem)
            return rec.next_token_logits(stem, False)

        # --- preflight: a cloze answer must be a single space-prefixed token (the
        # "Paris" pattern). Digit/number answers tokenize as [space, digit], so
        # first-token scoring would silently read the whitespace logit. Warn loudly
        # rather than score garbage (keeps the Step-5 corpus expansion honest). ---
        bad = [
            it["answer"]
            for it in known
            if (ids := rec.tokenizer(" " + it["answer"].strip(), add_special_tokens=False).input_ids)
            and rec.tokenizer.decode([ids[0]]).strip() == ""
        ]
        if bad:
            print(f"  [warn] {len(bad)} known answer(s) score a whitespace first token "
                  f"(fix the corpus, see _answer_token_id): {bad}")
        span.set_attribute("n_malformed_answers", len(bad))

        # --- capture last-token residuals for every layer, one forward/prompt ---
        known_res = [capture(it["prompt"]) for it in known]
        unknown_res = [capture(it["prompt"]) for it in unknown]

        # --- numerical health: detect -> log loudly -> recover EXPLICITLY ---------
        # bf16 Gemma 4 can emit NaN/inf activations; a contaminated layer must be
        # excluded from the sweep (reported), and a verdict on a contaminated layer
        # is WITHHELD — never silently recomputed elsewhere.
        contaminated = contaminated_layers_from_residuals(known_res + unknown_res)
        numerical_health = {
            "contaminated_layers": sorted(contaminated),
            "n_prompts_affected_per_layer": {str(k): v for k, v in contaminated.items()},
            "policy": "exclude contaminated layers from sweep; withhold verdict if the "
                      "explicit/chosen layer is contaminated; no winsorizing",
            "winsorize": False,
        }
        span.set_attribute("contaminated_layers", sorted(contaminated))

        def _withhold(reason: str) -> dict:
            print(f"\n  !! VERDICT WITHHELD — {reason}")
            print("     (numerical-health policy: no silent fallback; see numerical_health in the JSON)")
            res = {
                "experiment": "entity_knowledge",
                "model_id": settings.model_id,
                "model_variant": settings.model_variant,
                "verdict": "withheld",
                "withhold_reason": reason,
                "numerical_health": numerical_health,
            }
            od = Path(out_dir or (settings.data_dir / "eval" / "results"))
            od.mkdir(parents=True, exist_ok=True)
            p = od / f"entity_knowledge_WITHHELD_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            p.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"     wrote {p}")
            return res

        if contaminated:
            print(f"  [numerical-health] excluding contaminated layers {sorted(contaminated)} "
                  f"from the sweep (prompts affected per layer: {contaminated})")
            candidate_layers = [L for L in candidate_layers if L not in contaminated]
        if not candidate_layers:
            return _withhold("ALL layers contaminated with non-finite activations")
        if layer is not None and layer in contaminated:
            return _withhold(f"requested extraction layer {layer} is contaminated "
                             f"(non-finite activations in {contaminated[layer]} prompt(s))")

        # --- held-out split: select layer + fit d_know on TRAIN, evaluate on VAL ---
        k_tr, k_va = _split_indices(len(known), val_frac, split_seed)
        u_tr, u_va = _split_indices(len(unknown), val_frac, split_seed + 1)
        ktr = [known_res[i] for i in k_tr]
        kva = [known_res[i] for i in k_va]
        utr = [unknown_res[i] for i in u_tr]
        uva = [unknown_res[i] for i in u_va]

        def direction_at(L: int, kres: list, ures: list) -> torch.Tensor:
            return diff_of_means_direction([r[L] for r in kres], [r[L] for r in ures])

        def sep_auc(L: int, d: torch.Tensor, kres: list, ures: list) -> float:
            scores = [projection(r[L], d) for r in kres] + [projection(r[L], d) for r in ures]
            labels = ["yes"] * len(kres) + ["no"] * len(ures)  # "yes" = known
            a = rank_auc(scores, labels)
            return max(a, 1 - a) if a is not None else 0.5

        # Layer chosen by TRAIN separation only (no peeking at VAL).
        sweep_train = [(L, sep_auc(L, direction_at(L, ktr, utr), ktr, utr)) for L in candidate_layers]
        if layer is None:
            layer, train_auc = max(sweep_train, key=lambda t: t[1])
        else:
            train_auc = dict(sweep_train)[layer]
        d_know = direction_at(layer, ktr, utr)            # fit on TRAIN
        val_auc = sep_auc(layer, d_know, kva, uva)         # honest, held-out
        # Full-data per-layer sweep is reported for the layer PROFILE only (not selection).
        sweep_full = [(L, sep_auc(L, direction_at(L, known_res, unknown_res), known_res, unknown_res))
                      for L in candidate_layers]
        span.set_attribute("extraction_layer", layer)
        span.set_attribute("train_auc", train_auc)
        span.set_attribute("val_auc", val_auc)

        # --- specificity controls: d_know vs random unit dirs vs an orthogonal dir ---
        d_model = int(d_know.shape[0])
        gdir = torch.Generator().manual_seed(split_seed + 100)
        rand_dirs = [_random_unit(d_model, gdir) for _ in range(n_random)]
        _r = torch.randn(d_model, generator=gdir)
        orth_dir = _r - (_r @ d_know) * d_know            # remove the d_know component
        orth_dir = orth_dir / (orth_dir.norm() + 1e-8)

        def ablation_drop(prompt: str, tok_id: int, cl: float, direction: torch.Tensor) -> tuple[float, int]:
            with ablating(rec.layers, direction):
                abl = read_logits(prompt)
            return cl - float(abl[tok_id]), int((abl > abl[tok_id]).sum())

        # --- H1 necessity on VAL known: ablate d_know AND the control directions ---
        necessity = []
        for it in (known[i] for i in k_va):
            tok_id = _answer_token_id(rec.tokenizer, it["answer"])
            clean = read_logits(it["prompt"])
            cl = float(clean[tok_id])
            dk_drop, dk_rank = ablation_drop(it["prompt"], tok_id, cl, d_know)
            rand_drops = [ablation_drop(it["prompt"], tok_id, cl, rd)[0] for rd in rand_dirs]
            orth_drop, _ = ablation_drop(it["prompt"], tok_id, cl, orth_dir)
            necessity.append({
                "prompt": it["prompt"],
                "answer": it["answer"],
                "clean_logit": cl,
                "delta": dk_drop,                          # d_know logit drop (>0 = hurt recall)
                "dknow_ablated_rank": dk_rank,
                "random_drops": [round(x, 3) for x in rand_drops],
                "random_mean_drop": sum(rand_drops) / len(rand_drops),
                "random_max_drop": max(rand_drops),
                "orth_drop": orth_drop,
                "clean_rank": int((clean > clean[tok_id]).sum()),
                "clean_top1": rec.tokenizer.decode([int(clean.argmax())]),
            })

        # --- H2 sufficiency on VAL unknown: steer d_know up; measure entropy change ---
        sufficiency = []
        for it in (unknown[i] for i in u_va):
            clean = read_logits(it["prompt"])
            with steering(rec.layers, d_know, steer_coeff):
                steered = read_logits(it["prompt"])
            sufficiency.append({
                "prompt": it["prompt"],
                "clean_entropy": _entropy(clean),
                "steered_entropy": _entropy(steered),
                "delta": _entropy(clean) - _entropy(steered),  # >0 = steering sharpened
            })

        # Readout-health gate: a single non-finite logit/entropy would silently turn
        # every mean below into NaN-as-a-number. Withhold instead (fail-loud policy).
        bad_nec = [r["answer"] for r in necessity
                   if not all(math.isfinite(r[k]) for k in ("clean_logit", "ablated_logit", "delta"))]
        bad_suf = [s["prompt"][:40] for s in sufficiency
                   if not all(math.isfinite(s[k]) for k in ("clean_entropy", "steered_entropy", "delta"))]
        if bad_nec or bad_suf:
            return _withhold(f"non-finite readout values (necessity items {bad_nec}, "
                             f"sufficiency prompts {bad_suf})")

        nec_deltas = [r["delta"] for r in necessity]
        suf_deltas = [r["delta"] for r in sufficiency]
        clean_ranks = [r["clean_rank"] for r in necessity]
        # Per-random-direction mean drop over items, then aggregate across the K dirs.
        rand_dir_means = [
            sum(r["random_drops"][k] for r in necessity) / len(necessity)
            for k in range(n_random)
        ] if necessity else []
        random_mean = sum(rand_dir_means) / len(rand_dir_means) if rand_dir_means else None
        random_max = max(rand_dir_means) if rand_dir_means else None  # most-damaging random dir
        orth_mean = (sum(r["orth_drop"] for r in necessity) / len(necessity)) if necessity else None
        dk_mean = (sum(nec_deltas) / len(nec_deltas)) if nec_deltas else None
        baseline = max(random_max or 0.0, orth_mean or 0.0)
        specificity_ratio = (dk_mean / baseline) if (dk_mean is not None and baseline > 1e-6) else None
        gate_pass = bool(specificity_ratio is not None and specificity_ratio > 2.0 and (dk_mean or 0) > 0)
        summary = {
            "necessity_mean_logit_drop": dk_mean,
            "necessity_fraction_hurt": (sum(1 for d in nec_deltas if d > 0) / len(nec_deltas)) if nec_deltas else None,
            # Specificity controls — same ablating() across all layers, all unit dirs.
            "necessity_random_mean_logit_drop": random_mean,
            "necessity_random_max_logit_drop": random_max,
            "necessity_orth_mean_logit_drop": orth_mean,
            "specificity_ratio": specificity_ratio,  # drop(d_know) / max(random_max, orth)
            "specificity_gate_pass": gate_pass,       # True only if d_know >> controls (>2x)
            "specificity_verdict": (
                "SPECIFIC: d_know ablation hurts recall far more than random/orthogonal "
                "directions — necessity is specific to d_know."
                if gate_pass else
                "ARTIFACT/INCONCLUSIVE: random/orthogonal 1-D ablations hurt recall "
                "comparably (ratio <= 2) — the necessity signal is NOT specific to d_know."
            ),
            "sufficiency_mean_entropy_drop": (sum(suf_deltas) / len(suf_deltas)) if suf_deltas else None,
            "sufficiency_fraction_sharpened": (sum(1 for d in suf_deltas if d > 0) / len(suf_deltas)) if suf_deltas else None,
            # One-sample effect size = mean / SD of the d_know drops (NOT a two-group
            # Cohen's d; there is no second group — it measures how consistently >0 the drop is).
            "necessity_effect_size_one_sample": (dk_mean / stdev(nec_deltas)) if len(nec_deltas) > 1 and stdev(nec_deltas) > 0 else None,
            # Readout-validity guard: necessity only means something if the model places
            # the gold token high in the CLEAN pass (else -it is echoing, not recalling).
            "necessity_median_clean_rank": median(clean_ranks) if clean_ranks else None,
            "necessity_frac_recalled_top5": (sum(1 for r in clean_ranks if r < 5) / len(clean_ranks)) if clean_ranks else None,
            # Held-out separation (fixes the in-sample optimism of selecting on the same data).
            "train_auc": train_auc,
            "val_auc": val_auc,
            "n_train": len(k_tr), "n_val_known": len(k_va), "n_val_unknown": len(u_va),
        }

        result = {
            "experiment": "entity_knowledge",
            "model_id": settings.model_id,
            "model_variant": settings.model_variant,
            "templated": templated,
            "readout": "assistant_prefill" if templated else "raw_cloze",
            "instruction": instr if templated else None,
            "extraction_layer": layer,
            "n_layers": n_layers,
            "separation_train_auc": train_auc,
            "separation_val_auc": val_auc,
            "n_random_controls": n_random,
            "layer_auc_sweep_full": [[L, round(a, 4)] for L, a in sweep_full],
            "steer_coeff": steer_coeff,
            "summary": summary,
            "numerical_health": numerical_health,
            "necessity": necessity,
            "sufficiency": sufficiency,
        }

        out_dir = Path(out_dir or (settings.data_dir / "eval" / "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"entity_knowledge_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = summary
    ratio = s["specificity_ratio"]
    print("\n" + "=" * 68)
    print("E1 ENTITY-KNOWLEDGE DIRECTION — causal summary (held-out)")
    print("=" * 68)
    print(f"  layer {layer}/{n_layers} (chosen on TRAIN)   AUC train={train_auc:.3f} | val={val_auc:.3f}"
          f"   [n_val {s['n_val_known']}+{s['n_val_unknown']}]")
    print(f"  H1 necessity   : d_know logit drop = {s['necessity_mean_logit_drop']:+.2f}  "
          f"(hurt {s['necessity_fraction_hurt']:.0%})")
    print(f"  SPECIFICITY    : d_know {s['necessity_mean_logit_drop']:+.2f}  vs  "
          f"random(mean {s['necessity_random_mean_logit_drop']:+.2f} / max {s['necessity_random_max_logit_drop']:+.2f})  "
          f"vs  orth {s['necessity_orth_mean_logit_drop']:+.2f}")
    print(f"  GATE           : specificity_ratio = {ratio:.2f}  ->  "
          f"{'PASS (specific to d_know)' if s['specificity_gate_pass'] else 'FAIL (generic ablation artifact)'}"
          if ratio is not None else "  GATE: ratio undefined")
    print(f"  H2 sufficiency : entropy drop under steering = {s['sufficiency_mean_entropy_drop']:+.3f}  "
          f"(sharpened {s['sufficiency_fraction_sharpened']:.0%})")
    print(f"  readout guard  : median clean rank = {s['necessity_median_clean_rank']}  "
          f"(top-5 {s['necessity_frac_recalled_top5']:.0%})")
    print(f"  wrote {out_path}")
    print("=" * 68)
    return result


if __name__ == "__main__":
    run()
