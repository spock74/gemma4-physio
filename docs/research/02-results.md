# Results — probing track (Phase 6)

Run on Mac Mini M2 (16 GB, bf16, MPS+CPU split), `google/gemma-4-E2B-it`,
transformers 5.7.0, 2026-06-09. Raw result JSON in `data/eval/results/`.

> **Successor track (Gemma 3 + Gemma Scope 2).** The probing work pivoted off
> Gemma 4 (no official SAE) onto **Gemma 3 `-it`** with the official **Gemma Scope 2**
> suite. Charter: `../../NEW-PROJECT-CHARTER.md`; report: `gemma3-interp-report.html`;
> O4 design: `05-O4-entity-knowledge-plan.md`. Gates O1–O3 passed (load, SAE
> calibration Pearson 0.9999, 5/5 toolkit positive controls). The O4 entity-knowledge
> study now runs on `-it`, across all layers.
>
> **O4.4 — pre-registered full-control per-layer necessity sweep (gemma-3-4b-it).**
> Re-ran the O4.3 single-layer ablation (intervention reused verbatim) across all 34
> layers with **N=20 matched-random + 1 orthogonal control per layer**, pre-registering
> `PASS@L iff specificity_ratio > 2 AND |control Δlog-prob| < 1 nat`. **Verdict: necessity
> confirmed; NOT a selection artifact** — the L11-specific / model-specificity framing stays
> refuted (from O4.3); what survives the forking-paths test is the necessity claim, control-clean
> across a contiguous **L4–L32 (29/34)**, controls < 0.7 nat at every layer. **Tiers, not a binary
> count:** PASS = ratio>2 over a near-zero control, not uniform effect; strong core **L8–L25**
> (≥20 nats, ~100% demotion, peak **L18 55.6**) bar the L11 valley (8.9 nats, a local min between
> L10≈31/L12≈35); edges L4–L7, L26–L32 control-clean but weak (1.7–18 nats).
> The earlier "270m null / 4b L11 pass" model-specificity read was refuted by
> the per-layer sweep: the carrier is a distributed mid-network band in both models,
> null only at the readout-adjacent final layers. Full table:
> `calibration/o4_entity_knowledge/results/o4_4_layer_sweep_4b.json`.
> Positioning: directional ablation on chat/`-it` models predates this (Arditi 2024
> refusal-direction lineage; our ablation is a single-vector reimpl); the novel part is
> the localized, matched-control, rank/KL-confirmed necessity of **entity-recall** on
> Gemma 3 `-it`, profiled across all layers — not an absolute first.
>
> **O4.5 — recall-specificity control (gemma-3-4b-it). Verdict: GENERIC CHANNEL — the
> "knowledge gate" reading is DOWNGRADED.** Same `d_know`, same intervention (reused
> verbatim), three eval prompt conditions, common metric KL(clean‖ablated) at the readout,
> pre-registered before results. In-band median KL: recall **38.4**, non-recall (known-ness
> fixed) **32.7**, neutral fluency **32.7**. Pre-registered primary KL_recall/KL_nonrecall =
> **1.17** (needed >3); fluency hit **0.85×** recall (needed <0.33). The 20-random + orth
> controls are **inert in every condition** (KL ≤0.4), so the effect is specific to `d_know`
> (not noise) but its causal role is **general mid-network computation, not a recall gate**.
> O4.3/O4.4's "erases recall" is restated as **"`d_know` is a load-bearing mid-network
> direction"**; known/unknown decodability (O4.1) stays real and recall-correlated, but the
> causal necessity is not recall-specific. Datasets `data/eval/nonrecall_knownfixed.json`,
> `data/eval/fluency_neutral.json`; output `calibration/o4_entity_knowledge/results/o4_5_recall_specificity.json`.
> (Secondary perplexity diagnostic unreliable — excluded; verdict rests on KL. Generic was a
> pre-registered valid outcome.)
>
> **O4.6 — base-vs-`it` control. KL weaker on base (sharpness artifact — corrected by O4.7).** Ran the recall-specificity test on
> BASE `gemma-3-4b-pt` with raw-cloze readout (intervention reused verbatim; pre-registered). Base
> recalls fine (top-5 97.5%) and the known/unknown axis is **equally decodable** (AUC up to 0.92,
> ≈ `-it`). But ablating `d_know` perturbs recall **~2–3× more weakly on base** (peak **19.0 nats**)
> than on `-it` (peak 55.6, median 38.4) — no base layer meets the `-it`-matched ≥20-nat bar. This
> looked regime-dependent, but KL is sharpness-confounded → tested in O4.7.
> Output `calibration/o4_entity_knowledge/results/o4_6_base_control.json`.
>
> **O4.7 — sharpness control (rank-based necessity on base). Verdict: necessity PRESENT ON BASE; the
> O4.6 KL gap was a sharpness artifact.** Re-tested with scale-invariant gold-RANK demotion
> (intervention reused verbatim; pre-registered). On base, ablating `d_know` demotes the gold token
> in **100% of items across L10–L30** (gold → rank 2k–78k, lost top-1 70%), orthogonal control ≤15%
> at every band layer — matching `-it`'s ~100%. So the causal load of `d_know` is **regime-INDEPENDENT**:
> present on base and `-it` alike; the 19-vs-55 KL gap reflected `-it`'s sharper distributions, not
> weaker causality. (Frozen label PARTIAL, driven by one off-band noise layer; band-restricted it is an
> unambiguous sharpness-confound — conservative rule disclosed, not re-tuned.) **Net O4 (corrected):**
> the axis is decodable everywhere AND causally load-bearing everywhere (gold demoted ~100% base &
> `-it`); and it is load-bearing, not a recall gate (O4.5). The causal role is generic and
> regime-independent; the `-it` regime changed only the KL magnitude (sharpness), not the phenomenon.
> Output `calibration/o4_entity_knowledge/results/o4_7_base_rank_necessity.json`.

## E1 — Entity-knowledge direction (causal)

**Headline: strong, held-out LINEAR SEPARABILITY — but the causal necessity claim
does NOT survive a specificity control. The "rung-3" result is downgraded to rung-2.**

`d_know` = unit diff-of-means of last-token residuals (known − unknown). Readout =
next-token logits at the final position, no generation. Held-out: layer + `d_know`
fit on a TRAIN split; AUC and causal tests reported on a disjoint VAL split.

### Methodological finding 1 — the readout matters more than the model
The instruction-tuned model fed a **raw cloze echoes the context** ("The capital of
France is" → " France"), gold token at median rank ~600 — any causal test there is
off-distribution. Staying on `-it` (the deployed model — we do NOT switch to base
`-pt`), an **assistant-prefill** readout (user turn = *"Answer with the fact,
continuing the sentence."*, the cloze stem prefilling the model turn) makes `-it`
emit the fact at **rank 0** (median clean rank 0, top-5 100 %). See
`entity_knowledge.RECALL_INSTRUCTION`.

### Methodological finding 2 — the necessity "result" was an ablation artifact
Naïve necessity (ablate `d_know` across all 35 layers × all positions) gave a huge
**+32.9 logit drop, 40/40 hurt** — which *looks* like strong necessity. A
**specificity control** kills it. Ablating, with the *same* protocol, K=5 **random**
unit directions and one direction **orthogonal** to `d_know`:

| direction (held-out VAL, n=20 known) | mean gold-logit drop |
|---|---|
| `d_know` | **+29.3** |
| random unit (mean of 5 / max of 5) | +15.0 / **+29.9** |
| orthogonal to `d_know` | +14.5 |

**`specificity_ratio` = drop(d_know) / max(random_max, orth) = 0.98 → GATE FAIL.**
Removing *any* 1-D direction across all layers damages the residual stream enough to
knock the gold token down; the effect is **not specific to `d_know`**. The earlier
"necessity confirmed" was a generic-aggressive-ablation artifact. Per the stop-gate,
no `d_know`-specific necessity is claimed.

### Results (held-out VAL; assistant-prefill; layer chosen on TRAIN)
| Claim | Metric | Verdict |
|---|---|---|
| Separation (rung 2) | TRAIN AUC 1.00 → **held-out VAL AUC 0.96** | **real & generalizes** (held-out fixed the in-sample 1.0 optimism) |
| H1 necessity (rung 3) | `specificity_ratio` **0.98** (d_know ≈ random ≈ orth) | **FAIL — artifact, not d_know-specific** |
| H2 sufficiency (rung 3) | mean entropy drop under steering = **−1.7**, 5 % sharpened | **NULL** (unknowns already low-entropy: `-it` confidently confabulates) |

**Layer:** the argmax peak is **not stably localized** (layer 8/35 at n=16, 26/35 at
n=40, both AUC≈1.0) — the axis is present across a broad band, so the "≈ Ferrando
layer-9 in Gemma 2 2B" comparison is at best loose.

**Bottom line (superseded by E1c below):** under *global* ablation E1 lands at
**rung 2** — a real, held-out *linear* known/unknown direction (VAL AUC 0.96) whose
all-layer ablation effect is an artifact (a random direction does as much damage).
The surgical follow-up was then run as **E1c** and partially revives H1 — see next
section for the controlled verdict.

## E1c — Localized necessity with positive controls (the controlled verdict)

Design: `docs/research/02-positive-control-plan.md`; code `interp/e1c_controls.py`,
`interp/refusal_control.py`; results `data/eval/results/e1c_controls_*.json`,
`refusal_control_*.json` (2026-06-10, E2B-it, held-out VAL n=20 known, N=50 random
null per direction at its own localization, orthogonal marker, Logfire throughout).
A pre-run **numerical-health audit** (`interp/numerical_health.py`) confirmed all 35
layers fully finite on this readout (the community bf16 NaN report does not reproduce
on E2B here), so no verdict below is contaminated; guards now fail loud everywhere.

| Direction (localization) | mean gold-logit drop | percentile in own N=50 null | ratio /p95 | orthogonal |
|---|---|---|---|---|
| `d_unembed` — apparatus positive (final layer, last pos) | **+14.05** | **100.0** | 44.6× | +0.02 |
| `d_know` — the test (layer 26, last pos) | **+0.60** | **96.0** | 2.21 | −0.05 |
| `d_refusal` — method positive, behavioral readout (layer 8, last pos) | refusal rate **0.81 → 0.81** (no change) | 49.0 (null bulk) | — | 0.88 |

**GATE 1 (apparatus): PASS, decisively.** The localized pipeline detects a known
direction at the 100th percentile with a tight null (sd 0.18) — logit-level readouts
through this plumbing are power-validated.

**The test: `d_know` localized sits at the 96th percentile** (48/50 nulls below; the
null max +0.70 exceeds its +0.60), ratio 2.21, orthogonal at the null bulk. Per the
interpretation matrix (≥95): the earlier global-ablation FAIL was **intervention
blindness**, and H1 revives as a **small, localized, direction-specific logit-level
effect** — verdict **SPECIFIC-when-localized (weak, borderline-tail)**. Calibration:
the absolute effect is +0.6 logits (the apparatus floor is +14), N=50 gives 2 %
percentile resolution, and one random direction beat it. The decisive (un-run)
firm-up is a larger null (N≥100–200) and/or more VAL items — left as future work, not
run-until-positive.

**GATE 2 (method, behavioral): FAIL — and that failure is the calibration.**
`d_refusal` separates harmful/harmless at AUC 1.000 (decodable, again), the corpus
elicits refusals (clean rate 0.81), the forward-only first-token null readout is
validated against the generation matcher out-of-derivation (agreement 0.94/1.00) —
yet ablating `d_refusal` at its extraction layer/last position changes **nothing**
(13/16 → 13/16, identical generations, percentile 49). Single-layer/single-position
diff-of-means ablation **cannot flip a redundantly-mediated behavior on this model,
not even canonical refusal**. Consequences: (a) `d_know`'s 96th-pct result is a
**logit-level** necessity signal — no behavior-level claim ("gates recall") is made;
(b) `d_refusal` joins `d_know` as *decodable-but-not-single-point-causal* — a
consistent picture across features.

**Recorded caveats (by design, not fixed post-hoc):** the `d_unembed` null shares one
random direction across items while the real `d_unembed` is per-item (gold-token row)
— an acceptable asymmetry for the near-tautological apparatus floor; the `d_know`
comparison is fully apples-to-apples. The three directions live at **different
localizations**, so each is compared only to **its own** null — absolute drops are
not comparable across rows. Refusal n=16 → coarse CI on the rate.

## E1d — SAE fidelity control (third-party-attested, PASS)

Code/data: `calibration/neuronpedia_fidelity/` (`MANIFEST.lock.json`, `gate_c_*.py`,
`reference/`, `results/gate_c_results.json`). Run 2026-06-11. This is the
**non-tautological, externally-attested** positive control E1c lacked (`d_unembed`
was tautological; `d_refusal` was behavioral and failed).

**Correction first:** the methodology doc's "no public SAE for Gemma 4 E2B" was
**false**. Public SAEs exist: **`decoderesearch/gemma-4-saes`** (Chanin / Decode
Research; E2B at layers 6/17/28; on Neuronpedia). We can't train an SAE locally, but
we can load one and `sae.encode()` our own captured activations.

**Apparatus + isolation.** `sae-lens` 6.x pulls transformer-lens → torchvision →
`torch==2.7.1`, which would downgrade the core env (torch 2.11.0) and break the
locked stack. So sae-lens lives in an isolated venv (`calibration/.venv-sae`); the
**capture** runs in the core env, only `sae.encode` on saved activations runs in the
venv. We use `SAE.from_pretrained` + `sae.encode` only — **no HookedSAETransformer /
TransformerLens** (it does not support Gemma 4).

**The buffer cuvette.** Browser check of Neuronpedia: this SAE set is hosted **only
at layer 17** (layers 6/28 → 404). The SAE cfg (GATE A) pinned: trained on
**`google/gemma-4-E2B` (BASE, not -it)**, hook `model.language_model.layers.17`
(= forward-output = resid-post L17 = our `rec.layers[17]`), `prepend_bos=True`,
float32. So GATE C captures **base E2B** on the **exact** Neuronpedia token sequence
(`convert_tokens_to_ids`, BOS prepended), grabs resid-post L17, runs `sae.encode`,
drops the BOS position, and compares per-token to Neuronpedia's published values.

| feature | NP peak idx | local peak idx | Pearson | cosine | scale (local/NP) |
|---|---|---|---|---|---|
| 0 | 94 | 94 | **1.000** | 1.000 | 1.000 |
| 5 | 28 | 28 | **1.000** | 1.000 | 1.000 |
| 19 | 111 | 111 | **1.000** | 1.000 | 0.993 |

**GATE C: PASS, exactly.** Local activations reproduce Neuronpedia's published
per-token values at **Pearson 1.000 / cosine 1.000**, same peak positions, **and
matching absolute magnitudes** (scale ≈ 1.0). The apparatus — layer indexing, hook
point (resid-post vs -pre), BOS handling, dtype, raw-text protocol — is validated
against a third party on the real Gemma 4 E2B. Every interp number in this report
rests on the same capture machinery, now externally calibrated. (Layer used: 17, the
only one Neuronpedia hosts; the SAE also exists at layer 28 — adjacent to E1's L26 —
used by STEP D below.)

### STEP D — feature-level ablation (method test): NULL on selection, with teeth

Run 2026-06-11 (`step_d_*.py`, `results/step_d_results_*.json`). **Framing (agreed
in advance):** method positive at the **logit level on BASE E2B** — raw-cloze readout
(valid on base: recall precondition **median gold rank 0, top-5 98%**), layer-28 SAE
on-distribution, selection on TRAIN only (seed-0 split), ablation evaluated on the 20
disjoint VAL knowns, null = **50 frequency-matched SAE features** (apples-to-apples),
identical `ablating()` localization as E1c (one layer, last position). Never framed
as a causal replication of `d_know(-it)`.

| quantity | value |
|---|---|
| selected feature (max TRAIN known−unknown act diff) | **1007** (fires 100% of train; mean act 32 on VAL knowns) |
| its ablation effect on VAL gold logit | **−6.53** (a BOOST), percentile **0.0** in the null |
| null (50 matched features) | mean +0.71, p95 +2.09, negative envelope only −0.62 |
| largest null-feature drop | feature **1035: +20.68** (clean mean is +20.74) |
| cos(`d_know@28` base, W_dec[1007]) | 0.13 |

**Pre-registered verdict: NULL/INCONCLUSIVE** — the activation-diff-selected feature
does not specifically *drop* recall logits, so "diff-style selection + localized
ablation finds the causal carrier" is **not** demonstrated. Two clearly-labeled
exploratory observations sharpen what failed:

1. **The apparatus is not the bottleneck.** A single matched feature (1035) erases
   essentially the *entire* gold-token logit (+20.68 of +20.74) under the same one-point
   protocol — so there is **no logit-level redundancy excuse** at L28 (contrast
   `d_refusal`, where behavior was redundantly mediated). Post-hoc from the null, so
   no gate claim — but as existence proof, single-feature ablation CAN move this
   logit massively.
2. **The selection heuristic is.** The top known−unknown feature (1007) turns out to
   be *suppressive*: removing it **boosts** the gold logit, 10× outside the null's
   negative envelope — a specific, opposite-sign role. Activation-diff finds a
   correlate of knownness, not the recall carrier: **decodable ≠ causal at the
   SAE-feature grain**, the same dissociation the whole Phase 6 arc keeps finding.

### STEP D2 — feature 1035 confirmed: the TRUE method-positive (PASS)

Run 2026-06-11 (`step_d2_*.py`, `results/step_d2_results_*.json`). Follow-up on the
exploratory 1035 finding, with the circularity handled head-on: 1035 was the *max*
of STEP D's null, so testing it against that same null is vacuous, and a **fresh
matched null turned out not to exist** — after excluding the previous 50 + {1007,
1035}, the nearest features by firing rate sit at 0.17–0.25 (target 1.0) and by mean
activation at 0.72–3.13 (target 18.5). Discovery about the design: **STEP D's null
was a census** — every dense/strong L28 feature was already measured. The
replacement gate (fixed before the run): per-item consistency + non-tautology, with
the strongest *remaining* features as a labeled lower control.

| test | result |
|---|---|
| per-item consistency | **20/20** VAL items hurt (mean drop **+20.68** of clean +20.74) |
| non-tautology | max \|cos(W_dec[1035], `d_unembed`(gold))\| = **0.029** — orthogonal to every answer direction |
| census uniqueness (from STEP D) | runner-up **+8.83**, p95 **+2.09** in the exhaustively measured dense class |
| lower control (weak remaining, N=50) | mean **+0.05**, p95 +0.10, max +1.69 — weak features do nothing |
| what 1035 is | always-on (rate 1.0, acts ~17–20 on knowns AND unknowns), train-diff rank 65529/65536, cos(`d_know@28`) = −0.24; its direct unembedding readout is noise-like → acts via **downstream composition** (L29–34), not by writing the answer |

**GATE D2: PASS — scoped to what was measured.** A single, always-on SAE feature —
orthogonal to all gold unembeddings, invisible to known/unknown contrasts —
**single-point-ablates this gold-token logit to ~zero**, uniquely within the
censused dense class. This is the **non-tautological method-positive at the logit
level** the track lacked: the capture + SAE + one-point-ablation pipeline *can*
demonstrate single-feature causality on a logit. The winner's-curse caveat applies
to the +20.68 magnitude (selected as a max), not to the census uniqueness or the
four confirmations above.

**Construct caveat (recall-specificity UNCONTROLLED).** What was measured is the
gold-token logit on recall prompts — not "factual recall" as a construct. 1035 is
always-on and entity-independent, the typical profile of a *generic* computation
channel: removing it may degrade coherent next-token prediction in general, with
the gold-logit collapse as a special case. No control discriminates these readings
(the §3.3 specificity gap, now at the feature grain). The discriminating control —
not run — is cheap: compare clean-vs-ablated **full distributions** (KL, entropy,
what becomes top-1) on the same prompts AND on non-recall contexts (unknowns,
neutral text); recall-specific ⇒ distribution mostly intact except fact tokens;
generic ⇒ distribution-wide destruction. Until then the claim stays demoted:
**"carries this logit (recall-specificity uncontrolled)"**, not "carries recall".

**Methods moral of D+D2, in one line:** *selection by correlation fails (1007,
suppressive); exhaustive causal screening finds the causal feature for this logit
(1035).* The knownness-correlated directions (`d_know`, 1007) are decodable but at
most weakly / oppositely causal on it.

### STEP #1 (doc 04) — discriminating control: 1035 is a GENERIC channel

Run 2026-06-11 (`discriminating_1035.py`,
`data/eval/results/feature1035_discriminating_*.json`). Same knife as D2 (base
E2B, unit `W_dec[1035]`, L28/last position); RECALL = the 20 held-out VAL knowns,
NON-RECALL = the 20 structure-matched VAL unknowns + 20 blind generic
continuations; full clean-vs-ablated next-token distributions per context.
**Reading pre-registered before running.**

| metric | RECALL | UNKNOWN | GENERIC |
|---|---|---|---|
| KL(clean‖ablated), median | 0.553 | 0.326 | 0.327 |
| top-1 change rate | **10 %** | 45 % | 30 % |
| Δentropy, median | −1.50 | −0.37 | −0.54 |

Gold-drop sanity: **+20.68 reproduced exactly.** KL ratios recall/non-recall =
**1.69×** (pre-registered rule: ≤2× ⇒ generic). **Verdict: GENERIC-CHANNEL** —
the demoted claim resolves downward, not upward.

**Exploratory observation (labeled; the deeper finding):** the D2 "+20.7 erasure"
is largely a **common-mode logit shift, softmax-invariant**. Despite the −20.7
raw drop, the gold answer **remains top-1 after ablation in 18/20 recall items**
(' Tokyo'→' Tokyo', ' Rome'→' Rome', ' Au'→' Au'…), KL stays ~0.55, and recall
distributions actually *sharpen* (Δentropy −1.5). Distribution-level recall
**survives** the ablation. 1035 is a **logit-scale / common-mode channel** —
consistent with its profile (always-on, orthogonal to specific unembeddings,
noise-like promoted tokens).

**Methods lesson (the real yield of STEP #1):** raw-logit drop is **not a
sufficient necessity readout** — it is confounded by softmax-invariant
common-mode components. Every future necessity claim must pair the raw-logit
delta with **distribution-referenced metrics** (Δlog-prob of the target, rank,
KL). This retro-scopes GATE D2's "PASS": the pipeline demonstrably finds the
feature that dominates a *raw* logit; what that meant for the *distribution* was
answered only by this control.

## E2 — Elastic interpretability (premise corrected; empirical run pending)

**The design-doc premise is falsified by the shipped configs** (verified): E2B is
`d_model` 1536 / 35 layers, E4B is 2560 / 42 layers. Canonical MatFormer (Gemma 3n)
nests the **FFN intermediate dim only**, keeping `d_model` and depth fixed — there
directions transfer by identity. Gemma 4 **also compresses `d_model` and depth**, so
the residual streams live in different vector spaces and literal `d_know` transfer is
**dimensionally ill-posed**. transformers 5.7 exposes **no MatFormer slicing API**,
so only the two shipped checkpoints exist (no arbitrary Mix'n'Match granularities).

**Reframed E2** (`matformer_elastic.py`): dimension-agnostic linear **CKA** at
depth-matched layers; a **nesting test** (E2B vs E4B's first-1536 "head" dims vs the
tail-1536 vs a **random-slice** baseline); and per-model **held-out decodability**
(k-fold; 1-D diff-of-means primary, strongly-L2 torch probe secondary). Direct probe
transfer is N/A-with-reason (dims differ), not silently skipped.

### Results (E2B-it 1536d/35L vs E4B-it 2560d/42L; multi-position, n_repr = 2054)
| Question | Metric | Finding |
|---|---|---|
| Cross-granularity similarity | mean linear CKA over matched layers = **0.76** (0.52–0.91) | moderate alignment — the two MatFormer granularities are *similar, not identical*, despite the d_model+depth compression |
| Matryoshka inner-slice nesting | head **0.773** > random **0.750** > tail **0.730** (`head>random=True`) | **weak, depth-concentrated** evidence: mean margin only +0.02, but up to **+0.18 in deep layers** (e.g. L27→33: 0.87 vs 0.69). The relative margin is robust to template inflation |
| Decodability of known/unknown | held-out diff-of-means AUC, **E2B 0.998 / E4B 0.999** (rising with depth) | the entity-knowledge geometry **survives compression to E2B** and is decodable held-out in both |
| Direct probe transfer | — | **N/A** (1536 vs 2560 dims) — reported, not skipped |

**Caveat (reported in-result):** multi-position CKA includes the **shared chat-template
positions** (identical tokens across all 80 prompts), which inflates absolute CKA — so
0.76 is an upper-ish estimate of *content* alignment. The **nesting margin**
(head − random) is the cleaner signal, since the template positions enter head, random
and tail identically and cancel in the comparison.

**Bottom line (E2):** despite Gemma 4 compressing both `d_model` and depth (unlike
canonical FFN-only MatFormer), the known/unknown entity geometry is **moderately
aligned across the two granularities and decodable (held-out ≈0.99) in both**, with
only **weak, deep-layer-concentrated** evidence that E2B occupies E4B's inner
("head") dimensions. Consistent with E1's finding that the *direction is real and
decodable* — and, like E1, this is a **separation/decodability** result, not a causal
one (no intervention is run cross-model).

**E4B numerical health (the validity of the E2 numbers above).** A direct
all-layer/all-position E4B audit (`interp/numerical_health.py e4b`) is **infeasible on
this 16 GB host**: E4B (16 GB) saturates RAM → accelerate disk-offload → the process
OOM-killed at prompt 10 on the first try, and on the retry (with per-prompt
`empty_cache` added) the machine reached the reset-precursor state (~0 MB free,
multi-GB compressor) before the first prompt finished, so it was killed to avoid a
second hard reset. The E4B layers are instead **validated indirectly**: the E2 run's
**105 E4B-derived per-layer statistics** (CKA + held-out decode AUC across all **35
matched layers**) are **every one finite** (CKA 0.31–0.91, AUC 0.63–0.99, zero
NaN/null) — a non-finite E4B activation at any matched layer would have propagated to
a NaN there. So the published E2 numbers rest on numerically-sane E4B captures at the
layers they use. (The 7 layers skipped by depth-matching are unaudited but back no
reported number.) E2B, by contrast, audits clean directly: **all 35 layers finite**.
