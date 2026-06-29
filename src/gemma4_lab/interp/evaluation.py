from __future__ import annotations

import math
import torch
import logfire
from typing import Any

from .recorder import ActivationRecorder
from .directions import ablating, steering

def _answer_token_id(tokenizer: Any, answer: str) -> int:
    ids = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Answer {answer!r} tokenized to empty sequence.")
    return int(ids[0])

def _entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return float(-(p * logp).sum())

def evaluate_causal_necessity(
    rec: ActivationRecorder,
    dataset: list[dict],
    direction: torch.Tensor,
    instruction: str = "Answer with the fact, continuing the sentence.",
    templated: bool = True
) -> dict:
    """
    Evaluates necessity (H1): ablating the direction on KNOWN facts should lower 
    the logit of the correct answer token.
    """
    with logfire.span("interp.evaluation.necessity"):
        necessity = []
        for it in dataset:
            tok_id = _answer_token_id(rec.tokenizer, it["answer"])
            
            def read_logits(stem: str) -> torch.Tensor:
                if templated:
                    return rec.next_token_logits(instruction, True, stem)
                return rec.next_token_logits(stem, False)
                
            clean = read_logits(it["prompt"])
            cl = float(clean[tok_id])
            
            with ablating(rec.layers, direction):
                abl = read_logits(it["prompt"])
                
            dk_drop = cl - float(abl[tok_id])
            dk_rank = int((abl > abl[tok_id]).sum())
            
            necessity.append({
                "prompt": it["prompt"],
                "answer": it["answer"],
                "clean_logit": cl,
                "delta": dk_drop,
                "dknow_ablated_rank": dk_rank,
                "clean_rank": int((clean > clean[tok_id]).sum()),
            })
            
        bad_nec = [r["answer"] for r in necessity
                   if not all(math.isfinite(r[k]) for k in ("clean_logit", "delta"))]
        if bad_nec:
            raise ValueError(f"Non-finite readout values in necessity evaluation: {bad_nec}")

        nec_deltas = [r["delta"] for r in necessity]
        dk_mean = (sum(nec_deltas) / len(nec_deltas)) if nec_deltas else 0.0
        
        return {
            "mean_logit_drop": dk_mean,
            "fraction_hurt": (sum(1 for d in nec_deltas if d > 0) / len(nec_deltas)) if nec_deltas else 0.0,
            "results": necessity
        }

def evaluate_causal_sufficiency(
    rec: ActivationRecorder,
    dataset: list[dict],
    direction: torch.Tensor,
    steer_coeff: float = 8.0,
    instruction: str = "Answer with the fact, continuing the sentence.",
    templated: bool = True
) -> dict:
    """
    Evaluates sufficiency (H2): steering the direction up on UNKNOWN facts 
    should lower next-token entropy.
    """
    with logfire.span("interp.evaluation.sufficiency"):
        sufficiency = []
        for it in dataset:
            def read_logits(stem: str) -> torch.Tensor:
                if templated:
                    return rec.next_token_logits(instruction, True, stem)
                return rec.next_token_logits(stem, False)
                
            clean = read_logits(it["prompt"])
            
            with steering(rec.layers, direction, steer_coeff):
                steered = read_logits(it["prompt"])
                
            sufficiency.append({
                "prompt": it["prompt"],
                "clean_entropy": _entropy(clean),
                "steered_entropy": _entropy(steered),
                "delta": _entropy(clean) - _entropy(steered),
            })

        bad_suf = [s["prompt"][:40] for s in sufficiency
                   if not all(math.isfinite(s[k]) for k in ("clean_entropy", "steered_entropy", "delta"))]
        if bad_suf:
            raise ValueError(f"Non-finite readout values in sufficiency evaluation: {bad_suf}")

        suf_deltas = [r["delta"] for r in sufficiency]
        return {
            "mean_entropy_drop": (sum(suf_deltas) / len(suf_deltas)) if suf_deltas else 0.0,
            "fraction_sharpened": (sum(1 for d in suf_deltas if d > 0) / len(suf_deltas)) if suf_deltas else 0.0,
            "results": sufficiency
        }
