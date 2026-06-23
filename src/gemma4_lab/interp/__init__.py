"""Mechanistic-interpretability / probing track (research Phase 6).

Affordable probing on Gemma 4 E2B: forward-pass activation capture, difference-of-
means direction extraction, and causal interventions (directional ablation /
steering) via output-rewriting hooks. NO SAE training, NO large activation harvests
— see ../../docs/research/00-methodology-and-value.md for why.

Public surface:
    ActivationRecorder  — capture residual stream + next-token logits (recorder.py)
    diff_of_means_direction, projection, ablating, steering, rank_auc, cohens_d
                        — causal-direction toolkit (directions.py)
    entity_knowledge.run — E1 experiment (entity_knowledge.py)
    linear_cka, fit_logistic_probe, match_layers
                        — E2 elastic-transfer toolkit (matformer_elastic.py)
"""

from __future__ import annotations

from .directions import (
    ablating,
    cohens_d,
    diff_of_means_direction,
    projection,
    rank_auc,
    steering,
    unembedding_direction,
)
from .matformer_elastic import fit_logistic_probe, linear_cka, match_layers
from .recorder import ActivationRecorder, resolve_text_layers

__all__ = [
    "ActivationRecorder",
    "resolve_text_layers",
    "diff_of_means_direction",
    "projection",
    "ablating",
    "steering",
    "unembedding_direction",
    "rank_auc",
    "cohens_d",
    "linear_cka",
    "fit_logistic_probe",
    "match_layers",
]
