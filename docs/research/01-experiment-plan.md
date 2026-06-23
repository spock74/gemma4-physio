# Experiment plan — probing track

Companion to `00-methodology-and-value.md`. Defines the candidate set, the
selection, and the full design of the two chosen experiments. Code lives in
`src/gemma4_lab/interp/`.

## Candidate set (ranked for this stack)

| # | Idea | Method tier | Runs on | Status |
|---|------|-------------|---------|--------|
| E1 | **Entity-knowledge direction** — does a single direction causally gate factual recall vs. confabulation? | 3→**2** | **-it** (deployed model) | **done — separation real (held-out VAL AUC 0.96); H1 necessity FAILS the specificity control (`specificity_ratio` 0.98 → artifact, not d_know-specific)** (`entity_knowledge.py`, `02-results.md`) |
| E2 | **Elastic interpretability** — is the entity geometry aligned across the E2B/E4B MatFormer granularities? | 2 | -it, both variants | **done (n_repr 2054): mean CKA 0.76, known/unknown held-out decodable in BOTH (≈0.99), weak deep-layer inner-slice nesting (head 0.77 > random 0.75)** (`matformer_elastic.py`, `02-results.md`) |
| E3 | Refusal single-direction + format-gating (Arditi × BioRefusalAudit) | 3 (causal) | **-it only** | designed, deferred (needs -it; see sibling LISP repo's `refusal_direction.py`) |
| E4 | *Borrowed Geometry* head-importance replication on the nested 2B | 2→3 | -pt | folds into E2 |

Dropped as flagship: audio phoneme-geometry probe — foundation paper (Borrowed
Geometry) does not support the phoneme-grapheme hypothesis, and the audio encoder
pressures 16 GB on top of the text model. Keep as a later stretch.

Selection rationale: E1 is the cheapest causal result and directly upgrades the
LISP repo's correlational drift metric using the same should_know/cannot_know
contrast. E2 is the most novel (unique to Gemma 4's MatFormer) and reuses the
existing `model_variant` Setting. E3 is sharp but needs -it and is already
scaffolded in the sibling repo. Sequence: **E1 → E2**, with E4 as E2's head-level
companion.

---

## E1 — Entity-knowledge direction (causal)

**Anchor.** Ferrando et al. 2024/ICLR 2025, *Do I Know This Entity?*
(arXiv:2411.14257). They found entity-recognition directions (the model
"knows that it knows" about an entity); steering them causally induces
hallucination or knowledge-refusal. Crucially the directions live in the *base*
model's representations — so the phenomenon is present on both -pt and -it. They
*discovered* the feature with SAEs; we extract the same axis with **difference-of-
means**, which is the SAE-free, M2-feasible approximation.

**Hypothesis.** A single linear "known-entity" direction `d_know`, extracted by
diff-of-means over known vs. unknown referents, is *causally* used by Gemma 4 E2B
for factual recall:

- H1 (necessity): ablating `d_know` on a **known** cloze prompt lowers the logit
  of the correct continuation token.
- H2 (sufficiency): adding `d_know` on an **unknown/fictional** prompt sharpens the
  continuation (lower next-token entropy) — more confident confabulation.

**Method.**
1. Build a known/unknown contrast (see `data/eval/entity_knowledge_contrast.json`).
   Known items are clozes with a single high-probability answer token
   (`"The capital of France is" → "Paris"`); unknown items are fictional/private/
   false-premise referents with no truthful continuation.
2. Extract `d_know` by diff-of-means of the last-token residual at one layer
   (Ferrando found entity recognition peaking ~layer 9 in Gemma 2 2B — **sweep**
   layers here; E2B has its own depth).
3. **Readout = next-token logits at the final position (NO generation).** This is
   the key feasibility move: forward-pass only, so hundreds of prompts run despite
   0.4 tok/s. Necessity = drop in correct-token logit under ablation on known
   items; sufficiency = drop in next-token entropy under steering on unknown items.
4. Baseline rung-1/2 comparison: report diff-of-means **separation AUC** (known vs.
   unknown projection onto `d_know`) so the causal result sits next to the
   correlational one it replaces. AUC and Cohen's d computed in-repo (no sklearn).

**Falsification (this is the point).** If ablation does not move the correct-token
logit, recall is not mediated by this direction at 2B — a real negative result. If
the projection separates known/unknown but ablation has no causal effect, you have
reproduced the "probe reads a direction the model does not use" failure mode, which
is itself worth reporting.

**Contrast vs. the LISP drift metric.** Drift says "drift_cos separates categories
at AUC ~0.7 (n=25)". E1 says "ablating one direction lowers correct-token recall
logits on known entities and sharpens confabulation on unknown ones" — same cheap
machinery, causal claim, and it explains the signal drift only correlated with.

**Cost.** Direction extraction: ~2× corpus forward passes. Readout: 1 forward pass
per prompt per condition (clean / ablated / steered), no generation. Overnight.

---

## E2 — Elastic interpretability across MatFormer granularities (placeholder)

**Anchor.** Devvrit et al. 2023, *MatFormer* (arXiv:2310.07707). E2B shares all
attention and the first `m_i` FFN neurons of E4B. MatFormer shows submodels are
*behaviorally* strong; nobody has asked whether they are *representationally
aligned*.

**Hypothesis.** A direction or linear probe fitted on E4B activations transfers to
E2B at matched layers (high CKA; probe-transfer accuracy ≈ in-distribution), up to
the layer/feature where elastic shrinkage breaks alignment.

**Method (target).**
1. Run the same contrast (reuse E1's corpus) through `model_variant="e4b"` and
   `model_variant="e2b"` — one at a time (each ~9.5 GB), capturing matched-layer
   residuals.
2. Compare with **CKA** and **probe-transfer**: fit `d_know` (or a logistic probe,
   implemented in torch — no sklearn) on E4B activations, evaluate on E2B and
   vice-versa.
3. Report per-layer transfer to locate where "free-lunch" Mix'n'Match stops being
   representationally free.

**Payoff.** If directions transfer across granularities → "fit interpretability
once, deploy on any submodel," directly useful for on-device elastic deployment.
Novel by question, cheap by method, causal where it composes with E1's ablation.

**Status / premise correction (verified 2026-06-09).** The "nested, shared-residual"
premise above is **falsified by the shipped configs**: E2B is `d_model` 1536 / 35
layers, E4B is 2560 / 42 layers. Canonical MatFormer (Gemma 3n) nests the FFN
intermediate dim only — `d_model`/depth fixed, directions transfer by identity — but
Gemma 4 **also compresses `d_model` and depth**, so the residual streams are different
spaces and literal "fit `d_know` on E4B (2560-d), apply to E2B (1536-d)" is
dimensionally ill-posed. `transformers` 5.7 exposes **no MatFormer slicing API**, so
only the two shipped checkpoints exist (no intermediate granularities to sweep).
E2 is therefore **reframed** (and implemented in `matformer_elastic.py`): dimension-
agnostic **CKA** at depth-matched layers + a head-slice **nesting test** + per-model
**probe decodability** (torch GD, no sklearn); direct probe transfer is reported
N/A-with-reason. See `docs/research/02-results.md`. Cross-model run needs E4B (~16 GB).

**E4 companion.** Per-head ablation over a layer slice (Borrowed Geometry method,
arXiv:2605.00333) to test whether the nested 2B preserves the 31B's single
token-matching head or redistributes it.

---

## Mapping to the codebase

| Concern | File |
|---|---|
| Activation capture + logit readout (shared infra) | `src/gemma4_lab/interp/recorder.py` |
| Diff-of-means, ablation, steering | `src/gemma4_lab/interp/directions.py` |
| E1 experiment (functional) | `src/gemma4_lab/interp/entity_knowledge.py` |
| E2 experiment (placeholder) | `src/gemma4_lab/interp/matformer_elastic.py` |
| Contrast corpus | `data/eval/entity_knowledge_contrast.json` |
| Results | `data/eval/results/` (created at run time) |

All capture/intervention is wrapped in Logfire spans (project rule: observability
first). No new runtime deps — AUC / Cohen's d / probes are computed with torch +
stdlib.
