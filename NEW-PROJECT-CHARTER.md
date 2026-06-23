# Project charter — standard probing on a fully-instrumented model

> Successor scope for the probing track. North star: **make standard probing
> methods work, reproducibly, on a model where every layer is officially
> instrumented.** Everything else is out of scope until that holds.

## Fixed decisions (do not relitigate)

- **No fallback, ever.** If a precondition is not met (model won't load, hook
  name missing, SAE absent, device wrong), **raise a clear error naming expected
  vs found** — never silently substitute a smaller model, different layer, or CPU
  path. Staging dev deliberately on a smaller model (see O1) is an *explicit plan
  step*, not an automatic substitution on failure.
- **Dev/target models:** toolkit is built and gated first on `google/gemma-3-270m-it`
  (text-only, ~268M — instant load, fast iteration), then the real science runs on
  `google/gemma-3-4b-it` (bf16). Both instruction-tuned, matching the downstream
  `-it` target. The 270m step de-risks the apparatus separately from the 4b load.
- **SAE/transcoder anchor:** **Gemma Scope 2** (`google/gemma-scope-2-4b-it`).
  Official DeepMind, *all layers*, three sites (resid / MLP / attn) + transcoders,
  Matryoshka. Loaded via **SAELens** (isolated venv — sae-lens pins torch).
- **Calibration ground-truth:** Neuronpedia (`gemma-scope-2`).
- **Why not Gemma 4 E2B:** no official SAE suite; community `decoderesearch/gemma-4-saes`
  covers only layers 6/17/28 and is base-only. That single restriction is what
  forced layer-17-only work and tangled the old track.
- **Why not Gemma 2 2B (primary):** SAEs mostly base, not `-it`; does not close the
  base-vs-it convergence gap. Keep as *optional secondary* only for circuit-tracer
  attribution graphs (which currently support Gemma-2-2B, not Gemma 3).

## Objectives (ordered; each gated by a positive control)

1. **O1 — Load & capture. ✅ DONE (both gates pass).** Two staged gates, both on
   M2/MPS, full-depth activation capture at *any* layer:
   - **O1a (270m-it): ✅ PASS.** `gemma-3-270m-it` (text-only, `Gemma3ForCausalLM`)
     loads bf16 on MPS; 18 layers, d_model 640; resid-post finite at [0,9,17];
     logits finite (argmax 'Paris'). Decoder path `model.layers`.
   - **O1b (4b-it): ✅ PASS.** `gemma-3-4b-it` (`Gemma3ForConditionalGeneration`)
     loads bf16 on MPS **without the buffer wall** — plain `.to("mps")`, no
     accelerate split needed (largest tensor ~1.34 GB ≪ ~7 GiB cap). 34 layers,
     d_model 2560; resid-post finite at [0,17,33]; logits finite. **Decoder hook
     path = `model.language_model.layers`** (recorder's 1st candidate — no change).
2. **O2 — SAE calibration. ✅ DONE (PASS).** Loaded `gemma-scope-2-270m-it-res /
   layer_12_width_16k_l0_medium` (d_in 640, hook `blocks.12.hook_resid_post`,
   model_name `google/gemma-3-270m-it` — SAE on the **-it** model) via the existing
   isolated `.venv-sae` (sae-lens 6.44.2 already ships all 84 gemma-scope-2 releases
   — no rebuild). Reproduced 5 Neuronpedia features at **mean Pearson 0.9999** over
   NP-reported positions, ~0.5% median magnitude error, all 5 peaks matched.
   Pipeline: `calibration/gemmascope2_fidelity/` (gate_a → fetch_reference →
   o2_capture → o2_encode). **Gate (Pearson ≥0.99 on ≥3 features): exceeded.**
   Protocol lessons baked into the scripts: (a) Neuronpedia renders sentencepiece
   `'▁'` as a plain space → token→id needs a fix-up; (b) snippets already include
   `<bos>` and are chat-formatted → feed ids directly, no extra BOS; (c) NP masks
   BOS and reports only up to an example-specific window → compare on NP-reported
   positions, not full-sequence (the local SAE is a superset).
3. **O3 — Standard toolkit works end-to-end. ✅ DONE (5/5 controls PASS).** Each
   method validated by a positive control on gemma-3-270m-it
   (`calibration/o3_toolkit/o3_controls.py`, reusing `interp/directions.py`):
   - linear probe / diff-of-means → sea-vs-code held-out **AUC 1.0**
   - logit lens → `head(final_norm(last-layer resid))` reproduces model top-1,
     **max|Δlogit| 1e-4** (validates norm+head wiring)
   - activation patching → clean(France)→corrupt(Japan) last-pos flips the gap
     **−8.26 → +17.10 @L17** (Tokyo→Paris)
   - ablation → **localized** (last layer/pos) drop of d_unembed **38.3** vs random
     **−0.46** (global all-layer ablation is destructive — the E1 artifact lesson)
   - steering → **localized**, coeff scaled to ‖resid‖, **monotone** logit response
   **Gate met: every method passes before any science claim.** Lesson re-confirmed:
   interventions must be localized + matched-random-controlled, never global.
4. **O4 — One real question. ✅ DONE (O4.1–O4.3, incl. per-layer sweep).**
   Port the entity-knowledge known/unknown study across *all* layers, on `-it`.
   Design: `docs/research/05-O4-entity-knowledge-plan.md`. Code:
   `calibration/o4_entity_knowledge/o4_decodability.py` (reuses `directions.py` +
   `RECALL_INSTRUCTION`). Corpus: the existing 40 known / 40 unknown.
   - **O4.1 decodability (all layers, held-out):** known/unknown diff-of-means axis
     is held-out decodable on both `-it` models — **270m-it val AUC 0.958 @ L17**;
     **4b-it val AUC 0.922 @ L33** with a clean, well-calibrated mid-layer bump
     (L11, train 0.853 ≈ val 0.848). The all-layer profile the old layer-17-only
     gemma-4 track could never produce.
   - **O4.2 readout validity:** the assistant-prefill recall readout elicits REAL
     recall — 270m **top-5 95%** (median gold rank 0), 4b **top-5 100% (40/40)**.
     So on 4b every "known" item is genuinely known → the contrast is clean and
     necessity can be tested on all items.
   - **O4.3 necessity, profiled across ALL layers — ✅ DONE.** Single-layer ablation
     of `d_know` at the recall position, matched orthogonal control, paired readouts
     (Δlogit, Δlog-prob, gold-rank, KL). Code: `o4_necessity.py` (single point, full
     5-random+orth) and `o4_necessity_sweep.py` (per-layer). **Result: a broad
     mid-network causal BAND, in BOTH models — not one special layer.**
     - **4b-it:** `d_know` ablation hurts recall across **L6–L32** (peak **L18**,
       gold Δlog-prob **+55.6 nats**, 100% demoted); orthogonal control inert at
       every layer (|Δlog-prob| < 0.12). Final layer L33 → null.
     - **270m-it:** same shape — mid band, peak **L7** (+31.9 nats, ratio 53.6, 100%
       demoted); final layer L17 → null.
     - **Correction (the sweep overturned the first pass).** The first single-layer
       run read "4b L11 PASS / 270m L17 NULL" as model specificity. The sweep
       **refutes** that: 270m L17 is just the readout-adjacent final layer (null in
       both models), and 4b L11 (+8.9) is a *weak* point of a band peaking ~6× higher.
       **No model-specificity story holds**; the carrier is a distributed mid band in
       both. Lesson: sweep, don't pick — single-layer picks invite forking-paths.
     - L11 single-point deep-dive retained as one column: ratio 69.8 (orth-only), gold
       0.05→170.2, 65% demoted; heterogeneous (Tokyo/Au/"seven" collapse,
       Jupiter/Vinci/London resist → recall redundancy).
   - **O4.4 pre-registered full-control sweep — ✅ DONE. Necessity confirmed; not a
     selection artifact (L11-specific / model-specificity stays refuted).**
     `o4_4_layer_sweep.py` (intervention reused verbatim from `o4_necessity.necessity_readouts`)
     re-ran the per-layer ablation with **N=20 matched-random + 1 orthogonal control
     per layer**, pre-registering `PASS@L iff ratio>2 AND |control Δlog-prob|<1 nat` and
     the survival rule (L11 PASS AND contiguous band ≥2) BEFORE results. The PASS region
     is a fully contiguous **L4–L32 (29/34 layers)**, 20-random and orth controls **<0.7
     nat at every layer**; only L0–L3 and final L33 fail.
     **Tiers, not a binary count:** PASS = ratio>2 over a near-zero control, not uniform
     effect size. Strong causal **core L8–L25** (≥20 nats, ~100% demotion, peak **L18
     55.6**) bar the L11 valley; edges L4–L7 and L26–L32 are control-clean but weak
     (1.7–18 nats). The causal claim rests on the core, not the layer count.
     **L11 is a local minimum** inside the band (8.9 nats, between L10≈31 and L12≈35) —
     likely a noisier diff-of-means direction (consistent with its lower decodability AUC
     0.848), not a special role. Output:
     `calibration/o4_entity_knowledge/results/o4_4_layer_sweep_4b.json`.
   - **O4.5 recall-specificity control — ✅ DONE. Verdict: GENERIC CHANNEL (knowledge-gate
     reading DOWNGRADED).** `o4_5_recall_specificity.py` (intervention reused verbatim;
     only the prompt condition varies). Same `d_know`, refit per band-layer on TRAIN, three
     eval conditions, common metric KL(clean‖ablated) at the readout, pre-registered before
     results. In-band median KL: **recall 38.4 / non-recall (known-fixed) 32.7 / fluency
     32.7**. Pre-registered primary `KL_recall/KL_nonrecall = 1.17` (needed >3); fluency hit
     **0.85×** recall (needed <0.33). The 20-random+orth controls are **inert in every
     condition (KL ≤0.4)** → the effect is specific to `d_know` (not noise) but its causal
     role is **general mid-network computation, not a recall/knowledge gate**. So O4.3/O4.4's
     "erases recall" is restated as **"`d_know` is a load-bearing mid-network direction"**;
     known/unknown *decodability* (O4.1) stays real and recall-correlated, but the *causal*
     necessity is not recall-specific. Datasets: `data/eval/nonrecall_knownfixed.json`,
     `data/eval/fluency_neutral.json`. Output: `results/o4_5_recall_specificity.json`.
     (Secondary perplexity diagnostic was unreliable — pathological values from teacher-forcing
     over the chat-templated sequence — and is excluded; verdict rests on KL. Generic was a
     pre-registered valid outcome, not tuned away.)
   - **O4.6 base-vs-`it` control — ✅ DONE. KL weaker on base (peak 19 vs 55) — but this
     "regime-dependent" read was a sharpness artifact, CORRECTED by O4.7.**
     `o4_6_base_control.py` ran the recall-specificity
     test on the BASE `gemma-3-4b-pt` with its natural **raw-cloze** readout (intervention
     reused verbatim; `d_know` refit per layer on base; pre-registered). Base **recalls fine**
     (top-5 97.5%) and the known/unknown axis is **equally decodable** (AUC up to 0.92, ≈ `-it`).
     **But** ablating `d_know` perturbs recall **~2–3× more weakly on base** (peak **19.0 nats**
     @L20) than on `-it` (peak 55.6, in-band median 38.4): **no base layer meets the
     `-it`-matched ≥20-nat bar** → pre-registered label NO_CAUSAL_BAND_ON_BASE (present-but-weaker,
     grazes 19, not absent). So the strong O4.3/O4.4 necessity is **substantially regime-dependent**:
     decodability is regime-independent, the strong causal load is not. Caveats (pre-registered):
     bundles model(base-vs-it) + readout(raw-vs-template) — does not isolate which; part of the
     19-vs-55 gap may be a sharpness confound (`-it` sharper → larger KL); base recall-vs-non-recall
     specificity unmeasured (necessity gate failed first). Output: `results/o4_6_base_control.json`.
   - **O4.7 sharpness control — ✅ DONE. Verdict: necessity PRESENT ON BASE (the O4.6 KL gap
     was a sharpness artifact).** `o4_7_base_rank_necessity.py` re-tested with a
     scale/sharpness-invariant **gold-rank demotion** metric (intervention reused verbatim;
     pre-registered). On base, ablating `d_know` demotes the gold token in **100% of items across
     L10–L30** (gold → rank 2k–78k, lost top-1 in 70%), orthogonal control **≤15% at every band
     layer** — matching `-it`'s ~100%. So by the sharpness-invariant metric the causal load of
     `d_know` is **regime-INDEPENDENT**: present on base and `-it` alike; the 19-vs-55 KL gap was
     `-it`'s sharper distributions, not weaker causality. (Frozen verdict label PARTIAL, driven by
     one off-band noise layer L4 with orth 25% where `d_know`=0%; band-restricted it is an
     unambiguous sharpness-confound — conservative rule disclosed, not re-tuned.) Output:
     `results/o4_7_base_rank_necessity.json`.
     **Net O4 picture (corrected):** the axis is *decodable everywhere* AND *causally load-bearing
     everywhere* (gold demoted ~100% on base and `-it`, O4.7; KL-magnitude difference was sharpness,
     O4.6); and it is *load-bearing, not a recall gate* (O4.5). So decodable ≠ causal (it IS a real
     causal direction), but the causal role is **generic and regime-independent** — the `-it` regime
     changed only the KL magnitude (sharpness), not the underlying causal phenomenon.
   **Positioning (corrected after review — no absolute-first overclaim):** directional
   ablation to suppress behaviour on `-it`/chat models already exists (Arditi et al.
   2024, refusal direction, Llama-2-chat/Qwen-chat; our `directions.py` is a
   single-vector reimpl of it), and the entity-knowledge axis is Ferrando et al. 2024.
   What is new HERE is the narrower, still-publishable combination: **localized,
   matched-control, rank/KL-confirmed causal necessity of entity-knowledge factual
   recall on Gemma 3 `-it`, profiled across all layers** — the "middle-ground
   intervention class" the program decision named as its viability gate, which the
   old gemma-4 track never reached (it stayed "decodable but global-ablation artifact
   / logit-level only", stuck on base-only single-layer SAEs).

## Explicitly OUT of scope (the ramifications that caused sprawl)

- MatFormer / elastic-interp / E2B-vs-E4B nesting experiments.
- E4B numerical-health audits, MoE, 12B/26B models.
- Multi-backend inference (MLX/GGUF/QAT). **Quantization never touches interp** —
  it changes activation statistics and breaks SAE/logit-lens calibration.
- Multi-token prediction, synthetic-data, library-agent (those belong to the
  original lab's Phases 4–5, not here).

## Code to carry over (refactorable as-is or near)

| Keep / copy | From | Change needed |
|---|---|---|
| `observability.py` | `src/gemma4_lab/` | none — Logfire bootstrap is model-agnostic |
| `config.py` (secrets-as-constants pattern) | `src/gemma4_lab/` | swap model id / paths only |
| `interp/recorder.py` (capture + logit readout) | `src/gemma4_lab/` | re-point to gemma-3-4b-it; re-verify hook names |
| `interp/directions.py` (diff-of-means/ablation/steering) | `src/gemma4_lab/` | model-agnostic; revalidate |
| `interp/entity_knowledge.py` + contrast corpus | `src/gemma4_lab/`, `data/eval/` | reuse `RECALL_INSTRUCTION`; rerun on new model |
| `agentq` queue (optional) | `src/gemma4_lab/agentq.py` | none |

## Code to drop / rewrite

- `hf_local.py` — rewrite for gemma-3-4b-it; **re-verify the MPS load** (4B bf16 may
  or may not hit the ~7 GiB single-buffer cap — first thing to test on the Code host).
- SAE loading — replace `decoderesearch/gemma-4-saes` plumbing with SAELens +
  Gemma Scope 2.
- All Gemma-4 / MatFormer-specific modules — delete, do not port.

## Open risks — verification log (Code host, has MPS)

- ~~**Does gemma-3-4b-it bf16 load on M2 16 GB without the MPS buffer error?**~~
  **RESOLVED (O1b PASS):** plain `.to("mps")` loads it — no buffer wall, no
  accelerate split. The Gemma-4 9.51 GiB single-allocation failure does **not**
  recur (Gemma-3-4b's largest tensor is ~1.34 GB). Loaded the full multimodal
  wrapper (vision tower included) and a text-only forward pass works.
- ~~**gemma-3-4b-it multimodal hook path**~~ **RESOLVED (O1b):** decoder layers at
  **`model.language_model.layers`** (`Gemma3ForConditionalGeneration`), matching
  `recorder.resolve_text_layers`'s first candidate — no recorder change needed.
- ~~**Gemma Scope 2 4b-it: text-only vs multimodal activations?**~~ **NON-BLOCKING.**
  The DeepMind announcement frames Gemma Scope 2 purely as language-model interp
  (SAEs + transcoders on every layer of Gemma 3, 270M–27B; no mention of images).
  Even if the 4b SAE saw image-conditioned activations in training, O2 calibrates
  against **Neuronpedia**, which serves **text** dashboards — reproducing a published
  per-token activation is text-by-construction. And 270m-it (where O2 runs first)
  has no vision tower at all. Interpretation stays text-only per scope.

## Surface rule (carried from the lab)

Anything needing MPS / host bash / git is `target: code`. Cowork drafts and plans;
the Code tab loads models, runs probes, commits.
