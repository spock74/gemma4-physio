# Probing on Gemma 4 E2B — methodology and where the scientific value is

Research track for gemma4-lab. This document fixes the *method philosophy*; the
concrete experiments and their selection are in `01-experiment-plan.md`.

## 1. The governing constraint

Hardware: Mac Mini M2, 16 GB, bf16, MPS. The E2B-it checkpoint loads as the full
~9.5 GB E4B weight set (E2B is a runtime MatFormer sub-extraction), and on this
machine it is split across MPS+CPU by accelerate (`device_map="auto"` +
`max_memory`), giving correct but slow inference (~0.4 tok/s; most layers land on
CPU). See `src/gemma4_lab/inference/hf_local.py`.

This caps what "probing" can mean here:

- **Affordable**: forward-pass activation capture via `register_forward_hook`;
  difference-of-means direction extraction; linear/logistic probes; directional
  ablation and steering via output-rewriting hooks; logit readouts at the final
  position. Datasets of hundreds of short prompts. The binding cost is *text
  generation* (0.4 tok/s), not activation capture — so prefer readouts that need
  only a forward pass.
- **Not affordable**: training Sparse Autoencoders (SAEs), million-scale
  activation harvests, RL-over-features, automated circuit discovery at scale.

On the SAE route: training one locally is infeasible (million-scale activation
harvests + GPU). **Correction (2026-06-11):** the earlier claim that there is "no
public SAE for Gemma 4 E2B" was **false** — public SAEs exist at
**`decoderesearch/gemma-4-saes`** (Chanin / Decode Research, layers 6/17/28, on
Neuronpedia). So we cannot *train* an SAE here, but we *can* load pretrained ones
and apply `sae.encode()` to locally-captured activations. We used exactly this to
build the **E1d attested control** (`calibration/neuronpedia_fidelity/`): our local
capture reproduces a Neuronpedia feature's published activation at Pearson 1.0 —
see `02-results.md`. The unsupervised dictionary *training* program remains out of
budget; the pretrained-SAE *readout* program is open.

## 2. "Feasible" is not "low value" — value tracks claim structure, not method cost

A recurring worry: the affordable toolkit looks limited next to SAEs. The
literature says otherwise — a large share of high-impact interpretability results
used exactly these cheap primitives. What separates a publishable result from a
weak one is the **structure of the claim**, not the price of the method:

- **Correlational** ("metric M separates label L at AUC x") — weak; reviewers ask
  what the quantity *does* and which confound (e.g. prompt length) drives it.
- **Causal** ("ablating direction d removes behavior B; adding d induces B") —
  strong, even with a one-line method.

Cheap-method, high-impact anchors (all diff-of-means / linear probe / PCA, no SAE):

- Arditi et al. 2024, *Refusal Is Mediated by a Single Direction*
  (arXiv:2406.11717) — diff-of-means + ablation.
- Li et al. 2023, *Inference-Time Intervention* (ITI) — probe + activation shift,
  large TruthfulQA gains.
- Zou et al. 2023, *Representation Engineering* (RepE) — reading vectors via
  PCA/means.
- Marks & Tegmark, *The Geometry of Truth*; Burns et al., *CCS* (unsupervised
  linear probe for latent knowledge); linear world-model probes in Othello-GPT.
- 2026 reasoning-probe wave: a linear probe predicts CoT-trace correctness at
  ~0.95 AUROC from the first reasoning step (arXiv:2605.09502 and related). These
  are logistic regressions on activations.

The SAE program, by contrast, is under active scrutiny (2024–2026): reconstruction
≠ explanation, features often not causal, feature splitting/absorption. SAEs are a
different tool, not a strictly superior one. (State-of-debate as of mid-2025;
re-verify before citing a verdict.)

## 3. The claim-strength ladder (all rungs run on this M2)

The lab's sibling repo (`LLM-LISP-09-JUN-2026`) currently sits on rung 1: it
correlates a residual-stream geometry scalar (drift / trajectory smoothness) with
a hallucination label on ~25 prompts. That is the genuinely weak end — not because
it is cheap, but because it is correlational and confound-prone (whole-sequence
flatten conflates prompt length; labels were assigned by a labeler who had seen the
drift values; n≈10/class).

The upgrade path stays within budget:

1. **Unsupervised geometry vs label** — correlational, weak. *(where the LISP repo is)*
2. **Supervised linear probe** on a defined property — measures decodability.
3. **Diff-of-means direction + intervention** (ablate / steer, measure behavioral
   or logit change) — **causal**. *(the rung this track targets)*

Rungs 2–3 fit overnight on the M2 when the readout needs only a forward pass.

## 4. What actually requires more compute (the honest limit)

SAE / transcoder / crosscoder training; large-scale automated circuit discovery
(ACDC/EAP with thousands of ablations on a slow split model). These need GPU/TPU
or a fast local backend. Phase 2 (MLX/GGUF) would raise throughput and reopen the
generation-bound experiments, but not the SAE-training one.

## 5. Verification of the three source papers that seeded this track

The user brought three 2026 arXiv proposals. Verified against the primary sources:

- **arXiv:2605.30162** — *BioRefusalAudit* (DeLeeuw, 28 May 2026). REAL. Gemma 4
  E2B-IT refused 65/75 biosecurity prompts *with* chat template, 0/75 *without*;
  0% under an 80-token cap. A defensive finding: refusal is shallow / format-
  dependent. (Note: requires the **-it** checkpoint — base models do not refuse.)
- **arXiv:2605.26731** — *Harness Sensitivity Is Non-Monotone* (Cho, 26 May 2026).
  REAL. Gemma4:e2B matches strong-open-tier stability at 91.7% across harnesses,
  but the paper stresses **n=1 model per tier** — do not over-generalize.
- **arXiv:2605.00333** — *Borrowed Geometry* (Bektursun, 1 May 2026, Gemma 4 31B).
  REAL but commonly mis-cited: it is about frozen *text* weights transferring to
  synthetic **token-pattern** tasks (binary copy, associative recall, Rule 90,
  binary addition) — **not audio**. A single head computes the same token-matching
  variable across text and the synthetic tasks.

Architecture facts (verified): Gemma 4 released 2 Apr 2026 (E2B / E4B / 26B MoE /
31B Dense). E2B is a MatFormer sub-extraction of E4B — but note (verified from the
shipped configs, see `02-results.md`) that Gemma 4 compresses **`d_model` (1536 vs
2560) and depth (35 vs 42)**, not just the FFN dim as in canonical MatFormer/Gemma
3n, so E2B and E4B residual streams are *different* vector spaces. The audio tower
(304.8M USM-style Conformer, 12 layers, ≤30 s, `right_context=0`) sits *outside*
the MatFormer nesting and is trained only to feed the text decoder.

## 6. The publication lever for this lab

Method sophistication is not the lever — **novelty of question + causal claim** is.
The chosen experiments (see `01-experiment-plan.md`) are picked on that basis:
cheap method, question nobody has asked, falsifiable causal prediction.

### Key references

- Ferrando, Obeso, Rajamanoharan, Nanda 2024/2025, *Do I Know This Entity?*
  (arXiv:2411.14257, ICLR 2025) — entity-recognition directions found on the base
  model causally steer hallucination vs. knowledge-refusal. Anchor for E1.
- Arditi et al. 2024 (arXiv:2406.11717) — single-direction refusal. Anchor for the
  refusal variant.
- Devvrit et al. 2023, *MatFormer* (arXiv:2310.07707) — nested elastic submodels.
  Anchor for E2.
- Bektursun 2026 (arXiv:2605.00333) — head-importance fingerprints. Anchor for the
  E2 head-level companion.
