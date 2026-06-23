# O4 — Entity-knowledge known/unknown, all layers, on `-it`

The real science (charter O4). Design: `../../docs/research/05-O4-entity-knowledge-plan.md`.
Reuses `src/gemma4_lab/interp/directions.py` + `RECALL_INSTRUCTION`. Corpus:
`data/eval/entity_knowledge_contrast.json` (40 known / 40 unknown). Core env.

```
python calibration/o4_entity_knowledge/o4_decodability.py 270m   # O4.1 + O4.2
python calibration/o4_entity_knowledge/o4_decodability.py 4b
python calibration/o4_entity_knowledge/o4_necessity.py 270m 17    # O4.3 (pipeline check)
python calibration/o4_entity_knowledge/o4_necessity.py 4b 11      # O4.3 (the result)
```

## Results

**O4.1 decodability (held-out, all layers).** Known/unknown diff-of-means axis is
held-out decodable on both `-it` models — 270m val AUC **0.958 @ L17**; 4b val AUC
**0.922 @ L33** with a well-calibrated mid-layer bump at **L11 (train 0.853 ≈ val
0.848)**. The all-layer profile the layer-17-only gemma-4 track could not produce.

**O4.2 readout validity.** Assistant-prefill recall elicits REAL recall: 270m top-5
**95%** (median gold rank 0), 4b top-5 **100% (40/40)** → on 4b every "known" item
is genuinely known, so the contrast is clean.

**O4.3 necessity, profiled across ALL layers** (`o4_necessity_sweep.py`; single-layer
ablation per layer, recall position, matched orthogonal control; `o4_necessity.py`
is the single-point full 5-random+orth deep-dive). **Result: a broad mid-network
causal BAND, in BOTH models — not one special layer.**

| Model | causal band | peak layer | peak Δlog-prob | final-layer |
|---|---|---|---|---|
| **4b-it** | **L6–L32** | L18 | **+55.6 nats** (100% demoted) | L33 → null |
| **270m-it** | mid band | L7 | **+31.9 nats** (ratio 53.6, 100% demoted) | L17 → null |

Orthogonal control inert at **every** layer (|Δlog-prob| < 0.12) → the band is
specific to `d_know`. Single-point deep-dive at 4b L11 (one column of the band):
ratio 69.8, gold rank 0.05 → 170.2, 65% demoted; heterogeneous — less-redundant
facts (Tokyo, Au, "seven") collapse, famous/redundant ones (Jupiter, Vinci, London)
resist → recall redundancy.

### Correction (the sweep overturned the first pass)

The first single-layer run read **"4b L11 PASS / 270m L17 NULL"** as *model
specificity*. The per-layer sweep **refutes** that:
- 270m L17 is simply the readout-adjacent **final** layer, where the effect vanishes
  in *both* models; at 270m's **mid** layers (peak L7) it carries the same strong,
  specific signal.
- 4b L11 (+8.9 nats) is one of the **weaker** points of a band that peaks ~6× higher.

So **no model-specificity story is supported** — the carrier is a distributed
mid-network band in both. Lesson: **sweep, don't pick** — a single-layer pick
invites a forking-paths objection; the sweep corrected our own first read.

## O4.4 — pre-registered full-control layer sweep (`o4_4_layer_sweep.py`)

Answers the forking-paths objection to O4.3's L11 pick. The intervention is **reused
verbatim** from `o4_necessity.necessity_readouts` (lifted to module level; O4.3 numbers
unchanged) — the sweep only wraps it in a layer loop. Same seed-0 split, d_know refit
per layer on TRAIN, ablation on VAL. **N=20 matched-random + 1 orthogonal control per
layer.** Pre-registration written to the output JSON **before** results (frozen, not tuned):
`PASS@L iff specificity_ratio > 2 AND |control Δlog-prob| < 1 nat`; headline survives iff
L11 PASS AND in a contiguous active band (≥2 adjacent). Schedule: full range(34), spiral
order from L11, incremental save. Logfire span per layer.

**Verdict: HEADLINE SURVIVES.** PASS band **L4–L32 — 29/34 layers, fully contiguous**.

| region | layers | d_know Δlog-prob | controls | PASS |
|---|---|---|---|---|
| early | L0–L3 | 0.1–0.4 | <0.17 | ✗ (inactive) |
| band | **L4–L32** | 1.7 → **55.6** (peak L18) | random-max <0.7, orth ~0 | ✓ |
| L11 (the original pick) | L11 | 8.9 | 0.24 | ✓ (ratio 17.8 — a mid-strength column) |
| final | L33 | 0.00 | 0.02 | ✗ (readout-adjacent) |

20-random and orthogonal controls stay **< 0.7 nat at every layer** → the band is
specific to `d_know`. O4.3's L11 result is **not** a single-layer selection artifact;
it is one mid-strength column of a broad, control-clean causal band. Output:
`results/o4_4_layer_sweep_4b.json` (per-layer rows + prereg + verdict).

## Positioning (no absolute-first overclaim)

Directional ablation to suppress behaviour on `-it`/chat models already exists —
the refusal-direction lineage (Arditi et al. 2024, Llama-2-chat/Qwen-chat); this
toolkit's ablation is a single-vector reimpl of it, and the entity-knowledge axis is
Ferrando et al. 2024. What is new here is the narrower, still-publishable
combination: **localized, matched-control, rank/KL-confirmed causal necessity of
entity-knowledge factual recall on Gemma 3 `-it`, profiled across all layers** — the
"middle-ground intervention class" the program named as its viability gate, which the
old gemma-4 track (base-only, single-layer SAEs) never reached. The discipline that
produced it: localized (never global) interventions, matched orthogonal/random
controls, and softmax-aware readouts (raw-logit drop alone is common-mode — the
STEP #1 lesson).

## Notes

- 4b is loaded **bf16** (fp32 4b ≈ 17 GB > 16 GB host); fp32 is only needed for the
  O2 SAE-vs-Neuronpedia magnitude match, not for diff-of-means / AUC / ablation.
- VAL AUC uses a single 20/20 split (seed 0/1); multi-seed CIs are a cheap future
  hardening if a tighter number is wanted.
