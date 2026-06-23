# O4 — Entity-knowledge known/unknown, all layers, on `-it` (design)

Successor-charter objective O4 (see `../../NEW-PROJECT-CHARTER.md`). Port the
"Do I Know This Entity?" known/unknown study (Ferrando et al. 2024, arXiv:2411.14257)
to a **fully-instrumented `-it` Gemma 3 model, across ALL layers**, on the apparatus
validated by gates O1–O3. The recurring scientific question: is there a linear
known/unknown axis, where does it live, and is it **causally necessary** for factual
recall — measured properly.

## What is new vs the old gemma-4 E1/E1c track

| Old track (gemma-4) | O4 (gemma-3 `-it` + Gemma Scope 2) |
|---|---|
| Layer-17-only (SAE coverage) | **All layers** — full depth profile |
| Causality base-only (base-vs-it gap) | **`-it` directly** — gap closed |
| Necessity tested with **global** ablation → artifact (random/orth matched) | **Localized** ablation (single layer / band, recall position) + matched controls |
| Raw-logit drop as necessity readout → misleading (STEP #1) | **Pair** logit drop with **rank + KL** Δ, always |

## Rigor bar (non-negotiable — from the repo's hard-won lessons)

1. **Held-out.** Fit extraction layer + `d_know` on TRAIN; report separation AUC
   and every causal test on a disjoint VAL split.
2. **Specificity.** Every intervention is matched against N random unit directions
   and one direction orthogonal to `d_know`. A claim survives only if `d_know` ≫
   controls (ratio > 2).
3. **Localized, never global.** O3 re-proved global ablation is destructive enough
   that a random direction reproduces the effect. Interventions act at one layer
   (or a band) at the recall position — the charter's "middle-ground" viability gate.
4. **Readout validity.** On `-it`, a raw cloze is ECHOED, not recalled. Use the
   assistant-prefill recall readout (`interp.entity_knowledge.RECALL_INSTRUCTION`):
   the stem prefills the model turn so the fact is the immediate next token. A
   necessity claim is only meaningful where the gold token ranks LOW in the clean
   pass (the model actually recalls).
5. **Decodable ≠ causal.** Report decodability (AUC) and causality (ablation)
   separately; high AUC is not evidence of a causal carrier.
6. **Logit drop is not enough.** Pair every necessity readout with Δlog-prob, gold
   rank change, and KL(clean‖ablated) — raw-logit drops can be common-mode and
   softmax-invariant (the STEP #1 / D2 lesson).
7. **Nulls are results.** Withhold/負 verdicts are reported, never tuned away.

## Model staging

`gemma-3-270m-it` first (fast, validates the port), then `gemma-3-4b-it` for the
real claim. Caveat: 270m may not genuinely KNOW many of the 40 facts, so its
"known" class is partly "thinks-it-knows" — decodability still meaningful, but the
necessity claim belongs on 4b-it where recall is real (readout-validity gate decides
per item).

## Phases (executed with review between — no big-bang run)

- **O4.1 — Decodability across all layers.** Per-layer held-out known/unknown AUC,
  on `-it`. Foundational, non-causal, safe. Expected: high mid-layer AUC (old track
  hit ~0.96 on gemma-4). Deliverable: the layer AUC profile, both models.
- **O4.2 — Readout validity.** Per known item, clean gold-token rank under the
  assistant-prefill readout. Gates which items support a necessity claim. (Run with
  O4.1 — same forward passes.)
- **O4.3 — Localized necessity.** At the best decodable layer(s): ablate `d_know` at
  the recall position, single-layer/band, vs matched random + orthogonal controls.
  Readouts: Δlogit AND Δlog-prob AND gold-rank AND KL. Specificity gate > 2.
  **HELD for review after O4.1/O4.2.**
- **O4.4 — Sufficiency (optional).** Localized steering on unknown prompts; entropy
  / confabulation-confidence change. Only if O4.3 is informative.

## Corpus

`data/eval/entity_knowledge_contrast.json` — 40 known (with gold answers) / 40
unknown (fictional). Reused as-is; single-space-prefixed single-token answers are
preflighted (the digit-answer whitespace trap).

## Decision criteria

O4 is a SUCCESS of the *method port* if O4.1 shows a clean, held-out, all-layer
decodability profile on `-it` and O4.2 confirms real recall on 4b-it. The
*scientific* claim (localized necessity specific to `d_know`) stands only if O4.3's
specificity gate passes with the rank/KL-paired readout — otherwise the honest
result is "decodable but not a localized causal carrier," which is itself a finding.
