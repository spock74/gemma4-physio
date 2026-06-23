# E2B — conclusions & next steps: the ablation program is GO

> Naming note: **"E2" here = the Gemma 4 E2B model** (the 2B variant), NOT the
> MatFormer-elastic experiment "E2" of `01-experiment-plan.md`. This document
> answers one question, posed 2026-06-11: *focusing only on the 2B model, can we
> execute ablation experiments well enough to start the actual studies?*

**Verdict: GO — an *infrastructure* GO, not a science GO.** The infrastructure
phase is complete: capture is third-party attested, ablation is power-validated at
the regime that matters (logit-level, localized), feature-grain intervention works,
numerical health is guarded, and every cost is measured. **Be explicit about what
this is not:** the scientific yield to date is mostly null/negative results plus
apparatus validation — legitimate (the value is the method rigor), but the studies
of the model's knowledge mechanism **have not started**. Feature 1035 is a
**method/apparatus positive**, not a finding about how the model knows things.
"GO" means *the instrument is calibrated*, not *we have results*.

## 1. Validated capability matrix (all on E2B, Mac M2 16 GB)

| Intervention | Readout | Status | Evidence | Cost (measured) |
|---|---|---|---|---|
| Direction projection-ablation, **one point** (layer × last pos) | logit | ✅ **power-validated** | E1c GATE 1: `d_unembed` 100th pct, +14 logits, ratio 44.6× | ~1–2 s/forward; N=50 null ≈ 30 min |
| **SAE-feature** (W_dec) ablation, one point | logit | ✅ validated | STEP D2: feature 1035, +20.68 of +20.74, 20/20 items, census-unique | same |
| Direction ablation, **global** (all layers × all positions) | logit | ⚠️ non-specific | E1: random ≈ `d_know` (ratio 0.98) — generic damage; only usable WITH a random null | same |
| One layer × **all positions** / layer **band** × last pos | logit / behavior | 🔲 **untested — the open go/no-go** | gap between "one point" (can't flip behavior) and "everywhere" (non-specific) | ~1 h to establish |
| Any ablation | **behavior** (generation) | ⚠️ one point insufficient | E1c GATE 2: refusal 0.81→0.81 at pct 49 — redundantly mediated | ~30–60 s/generation; 48 gens ≈ 30–50 min |
| Steering (add direction / feature W_dec) | logit / entropy | ✅ machinery ready | E1 H2 used it (nulls so far, but the hooks bite: Δ57 logits) | cheap |
| SAE feature readout (`sae.encode`) | feature activations | ✅ **externally attested** | E1d GATE C: Pearson 1.000 / cosine 1.000 vs Neuronpedia (3 features, magnitudes incl.) | cheap; isolated venv |
| Residual capture, any layer/position | — | ✅ attested + health-guarded | E1d + `numerical_health`: E2B 35/35 layers finite | ~1–2 s/forward |

## 2. Two structural constraints that shape every study design

1. **Feature grain ⇒ base model; direction grain ⇒ both.** The public SAEs
   (`decoderesearch/gemma-4-saes`, layers 6/17/28) were trained on **base** E2B —
   feature-level work is on-distribution only there. Direction-level work
   (diff-of-means, ablation, steering) runs on **`-it`** too, which is the lab's
   declared target (probe the deployed model; readout = assistant-prefill,
   `entity_knowledge.RECALL_INSTRUCTION`). On base, raw-cloze is valid (median
   gold rank 0, top-5 98%).
2. **Logit readouts are the strong regime.** Cheap, power-validated, and where
   single-point interventions demonstrably work. Behavior-via-generation is
   expensive and single-point ablation does not flip it — behavioral claims need
   the untested middle-ground intervention class first (see Next steps #2).

## 2b. Convergence risk (named, not implied)

**The strongest causal machinery lives where the lab has no declared interest.**
Feature-grain causal screening — the one method that has produced a clean positive
(1035) — runs **only on base** (SAE on-distribution). The lab's declared target is
**direction-grain on `-it`** (the deployed model). The 1035 result is a *base*
result. "Test it on `-it` (off-distribution)" is listed as future work, but it is
precisely the hard and possibly **unanswerable** part: applying a base-trained SAE
to `-it` activations has no validity guarantee, and a null there would be ambiguous
(no transfer? off-distribution encoder? both?). The failure mode to guard against:
**accumulating clean causal stories on base that never transfer to the deployed
model** — months of base-model mechanism work with no bearing on the lab's actual
question. Mitigations, in order of preference: (i) keep the *discovery* on base but
always run the **direction-grain bridge** on `-it` (a feature's W_dec direction is
just a vector — its ablation/steering effect on `-it` can be measured exactly,
with `-it`-native nulls, no SAE needed at runtime); (ii) treat any base-only causal
story as **unconverged by default** in the docs; (iii) revisit if an `-it`-trained
SAE ever ships publicly.

## 3. Claim-calibration ledger (what we may say today)

| Claim | Status |
|---|---|
| Known/unknown direction exists, held-out (VAL AUC 0.96) | rung 2, solid |
| `d_know` necessity, global ablation | artifact (ratio 0.98) |
| `d_know` necessity, localized | weak/borderline (96th pct, +0.6 logits; firm-up needs N≥100–200) |
| Localized pipeline detects known directions | PASS (apparatus floor, 100th pct) |
| Single-point ablation flips behavior | FAIL (not even canonical refusal) |
| Feature 1035 carries **this raw gold-token logit** at L28 (base) | PASS — method-positive at the **raw-logit** level (STEP D2) |
| …and that effect is recall-specific | **NO — resolved by STEP #1 (2026-06-11, pre-registered): GENERIC channel.** KL recall/non-recall ratio 1.69× (≤2× rule); gold stays top-1 in 18/20 recall items despite −20.7 → the drop is largely **common-mode, softmax-invariant**; distribution-level recall survives. Methods lesson: raw-logit deltas must be paired with distribution-referenced metrics (Δlog-prob, rank, KL) |
| Selection by correlation finds causal features | refuted (1007 suppressive); **causal screening (census) works** |

Rigor bar (project memory, items 1–8): specificity controls, held-out splits,
N≥50 nulls with percentiles, numerical-health guards, localization, *name the
construct only after controlling it*.

## 4. Next steps, in order (decision value per hour)

1. ~~**1035 discriminating control**~~ — **DONE 2026-06-11, verdict
   GENERIC-CHANNEL (pre-registered)**: KL recall/non-recall 1.69× (≤2× rule);
   gold remains top-1 in 18/20 recall items → the D2 "+20.7" was largely a
   common-mode, softmax-invariant logit shift; distribution-level recall
   survives the ablation. The demoted claim resolved *downward*. Yield: the
   methods lesson — raw-logit drop is not a sufficient necessity readout; pair
   with Δlog-prob / rank / KL. (`discriminating_1035.py`,
   `data/eval/results/feature1035_discriminating_*.json`; write-up in
   `02-results.md` + report §3c.)
2. **Middle-ground intervention class (~1 h) — NOT a queue item: this is the
   viability gate of the entire program.** One layer × all positions, and a
   2–3-layer band × last position, ablating `d_know`/1035 with the same N=50
   random-null discipline. The stakes, spelled out: if **no** intervention scale
   between "one point" (cannot flip behavior — E1c GATE 2) and "everywhere"
   (non-specific — E1) is simultaneously *specific* and *behaviorally effective*
   on the 2B, then **behavior-level causal claims exit the program permanently on
   this hardware/model**, and the lab's declared interest (behavior of the
   deployed `-it` model) is not attackable here — the program stays logit-scoped
   for good. Run #1 first (10 min, resolves a demoted claim); but #2 is the
   strategic bifurcation.
3. *(Optional)* **`d_know` firm-up** (~2–4 h): N=100–200 null at layer 26
   resolves the 96th-pct borderline. Only if `d_know` stays a protagonist.
4. **The program proper**, once 1–2 land:
   - **Causal-screening census at L17** (base): same census method that found
     1035, at the layer where **Neuronpedia hosts explanations** — hits come out
     pre-labeled. Discovery method = the D+D2 moral (screen causally, don't
     select by correlation). **Cost: TBD — enumerate before committing.** Census
     cost is linear in the dense-class size: at L28 the class was ~51 features
     (≈ 30 min of ablations + 30 min lower control); L17's class size is unknown.
     Step 0 (~15 min): capture L17 residuals + `sae.encode` + count features above
     the activation floor. If the class is ~50–100, the census is ~1.5–2.5 h; if
     it is hundreds, redesign (e.g., pre-filter by decoder-norm × activation)
     before any commitment.
   - **Feature steering** (add W_dec, base): sufficiency at the feature grain.
   - **Direction studies on `-it`**: the deployed model, assistant-prefill
     readouts, direction grain — where the lab's stated interest lives.

## 5. Hardware / process guardrails (non-negotiable, learned the hard way)

- **E2B only on this host.** E4B (16 GB) saturates RAM → disk-offload → OOM or
  machine reset (happened once). E4B validity questions get **indirect** answers.
- One model process at a time; check `PhysMem` before launches; compressor
  multi-GB + ~0 free = kill immediately.
- `sae-lens` only in the isolated venv (`calibration/.venv-sae`) — it forces
  torch 2.7.1 and would break the core stack (2.11.0).
- Background jobs: fire one and wait for the harness notification — no monitor
  sleep-loops (they orphan).
- Logfire span on every capture/intervention; `numerical_health` block in every
  result JSON; commit per green gate with the decision in the message.

## Pointers

| What | Where |
|---|---|
| Results ledger | `02-results.md`, `phase6-report.html` (hand-maintained) |
| E1c control design + matrix | `02-positive-control-plan.md` |
| SAE calibration (E1d, STEP D/D2) | `calibration/neuronpedia_fidelity/` (`MANIFEST.lock.json` pins everything) |
| Capture / interventions code | `src/gemma4_lab/interp/` |
| Task queue | `.agent/` (001, 002 done) |
