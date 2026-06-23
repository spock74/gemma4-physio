# O3 — standard interp toolkit, positive-controlled (gemma-3-270m-it)

Charter O3: make the five standard probing methods work end-to-end, each behind a
**positive control** — a setup rigged so the method MUST show its effect if the
plumbing is correct. Passing validates the *mechanism*; it is not a science claim.

**Result: 5/5 PASS** (`o3_results.json`). Core env, reuses
`src/gemma4_lab/interp/directions.py` (model-agnostic). Run:

```
python calibration/o3_toolkit/o3_controls.py
```

| Method | Positive control | Result |
|---|---|---|
| linear probe / diff-of-means | sea-vs-code sentences separate held-out | **AUC 1.0** |
| logit lens | `head(final_norm(last-layer resid))` == model head | top1 match, **max\|Δlogit\| 1e-4** |
| activation patching | clean(France) last-pos resid patched into corrupt(Japan) | gap **−8.26 → +17.10 @L17** (flips) |
| ablation | localized d_unembed(answer) vs matched random | drop **38.3 vs −0.46** |
| steering | localized +k·‖resid‖·d_unembed, monotone in k | **monotone** [−1022,−628,37,681,1048] |

## The design lesson (why ablation/steering are LOCALIZED)

First pass ran ablation/steering **globally** (all 18 layers, all positions). Both
failed — not a plumbing bug but the repo's own **E1 lesson**: global ablation is so
destructive that a *random* direction tanks the target logit too (drop 39.9 unembed
vs 34.7 random), and global steering destabilizes / a fixed small coeff is
negligible against a ~1e4-norm residual that RMSNorm rescales. The correct control
is **localized** (single layer, last position) with a **matched random control** for
ablation and a **‖resid‖-scaled coeff** for steering. This is control design, not
tuning-to-pass — and it is exactly the discipline O4's science must inherit:
interventions are localized and matched-control'd, never global.
