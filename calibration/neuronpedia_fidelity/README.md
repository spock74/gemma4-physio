# SAE fidelity calibration (E1d) — Neuronpedia buffer cuvette

Third-party-attested positive control for the interp apparatus: reproduce a
Neuronpedia feature's published activation locally. If our local capture tracks
theirs, the apparatus (layer indexing, hook point, dtype, BOS) is validated against
an external source on the real Gemma 4 E2B. Plan: `../../docs/research/03-sae-neuronpedia-calibration-plan.md`.

## Environment isolation (why two envs)

`sae-lens` 6.x imports `transformer-lens` at load, which requires `torchvision<0.23`,
which version-locks `torch==2.7.1`. The core conda env (`gemma4-lab`) is pinned at
**torch 2.11.0** for the E1/E2/E1c experiments. The two cannot coexist, so:

- **Core conda env (torch 2.11.0)** — runs the model + capture (`gemma4_lab`,
  transformers 5.7, cached weights). Does the GATE B assertion and GATE C capture.
- **Isolated venv `calibration/.venv-sae` (torch 2.7.1 + sae-lens)** — loads the SAE
  weights (GATE A) and runs `sae.encode` on *saved* activations (GATE C encode).
  Created with `python -m venv calibration/.venv-sae && .venv-sae/bin/pip install sae-lens`.
  Gitignored.

The two never share a process. Captured activations are float tensors saved to
`captures/`, so encoding them on torch 2.7.1 is version-agnostic.

## Gates

- **A — instrument** (`gate_a_load_sae.py`, sae venv): load `SAE.from_pretrained(
  "decoderesearch/gemma-4-saes", "gemma-4-e2b/btk-mat-layer-28-k-100")`, pin cfg →
  `MANIFEST.lock.json`. PASS: `d_in == 1536`. **Findings:** `hook_name =
  model.language_model.layers.28` (forward-hook output = resid-post L28 = our
  `rec.layers[28]`); `model_name = google/gemma-4-E2B` (**BASE**, not -it) → GATE C
  must capture with base E2B to match Neuronpedia.
- **B — blank** (core env): assert `n_layers == 35`, `d_model == 1536`; record the
  hook→capture-point mapping in the MANIFEST. Architecture is base/it-identical, so
  -it (cached) is fine here.
- **C — buffer** (core env capture → sae venv encode): Neuronpedia protocol = RAW
  text (no chat template), their BOS. Freeze a feature's top-activating text +
  per-token values in `reference/`; capture base-E2B resid at L28; `sae.encode`;
  PASS iff local activation tracks Neuronpedia (same top position, high correlation).
  FAIL → STOP and name the breaking Gemma-4 detail. Output → `results/`.
- **D — bonus** (only if C passed): logit-level method positive control via feature
  ablation at L28.
- **E — register + correct**: write results into the report + `02-results.md`;
  correct the false "no public SAE for Gemma 4 E2B" claim.

`MANIFEST.lock.json` is the single source of pinned numbers — no guessed values.
