# E1c — Positive & negative controls for the localized necessity test

Companion to `01-experiment-plan.md` and the Phase 6 report. Defines the canonical
control experiment that makes E1's causal verdict *interpretable*.

## Why this experiment exists

E1's global-ablation necessity test failed its **negative control** (ablating a
random/orthogonal direction hurt recall as much as `d_know`). That tells us the
all-layer ablation is non-specific — but it leaves a fatal ambiguity:

> Is `d_know` genuinely *not* causally necessary, or is our test simply unable to
> detect *any* specific causal direction?

A negative control alone cannot answer this. We need a **positive control**: a
direction *known* to be causally necessary, run through the *same* pipeline. If the
positive control is detected (large, specific effect) and `d_know` is not, the
`d_know` negative is trustworthy. If even the positive control fails, the apparatus
is broken and no negative means anything.

## Controls taxonomy

| Control | Direction | Expectation | Purpose |
|---|---|---|---|
| Negative (have it) | random unit ×N, orthogonal-to-`d_know` | small / no specific effect | "does a non-causal direction look causal?" |
| Positive — apparatus | `d_unembed` = unembedding row of the gold token | **must** collapse gold logit, specific | proves the ablation+readout+gate plumbing detects a known direction (near-tautological floor) |
| Positive — method | `d_refusal` = Arditi diff-of-means on harmful/harmless (-it) | refusal rate drops, specific | proves *our diff-of-means* finds a real causal feature with this code (the load-bearing positive) |
| Test | `d_know` (localized) | unknown — this is what we're measuring | the actual question |

The apparatus control is cheap and runs today. The method control (refusal) is the
one that gives the result publication weight: *"the same pipeline that recovers the
canonical refusal direction finds no specific entity-knowledge necessity."*

## Protocol

Everything runs at the **localized** intervention that the global test lacked:
ablate only at the **extraction layer L\*** (≈26) **and only at the final token
position** (the readout position). Layer-localization is already possible
(`ablating(rec.layers[L:L+1], d)`); position-localization is the one new capability.

For each direction d ∈ {`d_unembed`, `d_refusal`, `d_know`}:

1. Ablate d (localized) → measure the effect on its native readout:
   - `d_unembed`, `d_know`: drop in the gold-token logit (forward-pass only).
   - `d_refusal`: drop in refusal rate (short generation ≤24 tokens — the only
     generation-bound piece).
2. **Null distribution**: ablate **N ≥ 50** random unit directions (same localization)
   and one orthogonal-to-d direction; record the effect distribution.
3. **Specificity statistic**: report d's **percentile within the random null**
   (a proper p-value-like number), not just a ratio against max-of-5. Keep the ratio
   `effect(d) / p95(random)` as a readable summary.
4. **Gate calibration**: the PASS threshold is no longer the arbitrary 2× — it is
   anchored empirically by where `d_unembed` and `d_refusal` land in the null. Report
   all three on the same axis.

## Interpretation matrix (this is the point)

| `d_unembed` / `d_refusal` | `d_know` (localized) | Conclusion |
|---|---|---|
| pass (far in null tail) | **fails** | `d_know` is genuinely not causally necessary — decodable-but-not-causal. **Trustworthy negative.** |
| pass | **passes** | the earlier failure was global-ablation bluntness; `d_know` IS specific when localized. **H1 resurrected.** |
| **fail** | any | apparatus cannot detect known-causal directions — methodology problem; fix before any claim. |

## Codebase deltas required (small)

1. **Position-localized ablation** — `directions.py`: add `positions: str|list[int]|None`
   to `ablating`/`steering` (and a `layers` subset is already supported). When
   `positions="last"`, only project-out `hidden[:, -1, :]`; default keeps current
   all-position behavior. ~10 lines.
2. **Unembedding-row direction** — new helper `unembedding_direction(model, token_id)`:
   `d = normalize(gamma ⊙ W_U[token])`, where `W_U` is the text LM head weight row and
   `gamma` is the final RMSNorm weight (accounts for the norm the readout applies). If
   `gamma` is not locatable, fall back to `normalize(W_U[token])` — still a valid
   apparatus control. Resolve the head via `model.get_output_embeddings()`; note Gemma
   ties embeddings and soft-caps logits (monotone — does not affect direction logic).
   Ablate this one at the **final layer only**, last position.
3. **Refusal positive control** — port the diff-of-means + refusal-rate logic from the
   sibling repo's `interp/refusal_direction.py`; add a modest low-severity
   harmful/harmless contrast under `data/eval/`. Behavioral readout = refusal rate via
   short greedy generation.

No new deps (torch + stdlib). Logfire span on every capture/intervention. `-it` only
(refusal does not exist on `-pt`).

## Acceptance / DoD

- `d_unembed` localized ablation drops the gold logit and sits in the extreme tail of
  the random null (apparatus validated) — else stop and fix plumbing.
- `d_refusal` localized ablation drops refusal rate and is specific (tail of null) —
  else the diff-of-means+localized pipeline cannot detect a known feature; report that.
- `d_know` reported with its null percentile; verdict read straight off the matrix.
- A null result for `d_know` with positives passing is the **strong** outcome — report
  it; do not tune to flip.

## Falsification & honesty

The experiment is designed to be able to *kill its own headline*: if positives pass
and `d_know` is in the null bulk, "entity-knowledge direction is decodable but not
causally used for recall (necessity test power-validated by positive controls)" — a
clean, citable negative. If `d_know` passes localized, the earlier global-ablation
"FAIL" is correctly attributed to intervention bluntness, not absence of effect.
