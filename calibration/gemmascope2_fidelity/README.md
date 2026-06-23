# O2 — Gemma Scope 2 SAE fidelity calibration (gemma-3-270m-it)

Third-party-attested positive control for the new probing apparatus (charter O2):
load an official **Gemma Scope 2** SAE and reproduce a published **Neuronpedia**
per-token activation locally. If local `sae.encode` tracks Neuronpedia, the
apparatus (model load, layer indexing, hook point, BOS handling, dtype, SAE
loading) is validated against an external source on the real `-it` model.

**Result: PASS** — 5/5 features, mean Pearson **0.9999** over Neuronpedia-reported
positions, ~0.5% median magnitude error, all peaks matched. See `results/o2_results.json`.

## Target

| | |
|---|---|
| Model (captured) | `google/gemma-3-270m-it` (the **-it** model the SAE was trained on) |
| SAE release / id | `gemma-scope-2-270m-it-res` / `layer_12_width_16k_l0_medium` |
| Hook | `blocks.12.hook_resid_post` (resid-post of decoder layer 12; d_in 640) |
| Neuronpedia source | `gemma-3-270m-it/12-gemmascope-2-res-16k` |

## Two-env split (same rule as E1d)

`sae-lens` pins `torch==2.7.1`; the core conda env is `torch 2.11.0`. They never
share a process. Reuses the existing **`calibration/.venv-sae`** (sae-lens 6.44.2 —
already ships all 84 `gemma-scope-2` releases, no rebuild needed).

- **Core conda env** — runs the model + capture (`o2_capture.py`).
- **Isolated `.venv-sae`** — loads the SAE + `sae.encode` on saved activations
  (`gate_a_load_sae.py`, `o2_encode.py`). Captured residuals are float `.npz`, so
  encoding them on torch 2.7.1 is version-agnostic.

## Pipeline (run in order)

```
.venv-sae/bin/python  calibration/gemmascope2_fidelity/gate_a_load_sae.py   # pin SAE cfg -> MANIFEST.lock.json
python                calibration/gemmascope2_fidelity/fetch_reference.py   # Neuronpedia refs -> reference/
python                calibration/gemmascope2_fidelity/o2_capture.py        # resid L12 -> captures/  (CORE env)
.venv-sae/bin/python  calibration/gemmascope2_fidelity/o2_encode.py         # encode + compare -> results/
```

## Protocol lessons (the non-obvious parts, baked into the scripts)

1. **Tokenization:** Neuronpedia renders the sentencepiece marker `'▁'` as a plain
   space `' '`. Mapping display tokens straight through `convert_tokens_to_ids`
   sends ~half to UNK. Fix: try the raw string, then `'▁'+rest` for a leading
   space, then a global `' '→'▁'` swap (`o2_capture.to_id`).
2. **BOS / chat:** the frozen snippets already start with `<bos>` and are
   chat-formatted corpus text. Feed the converted ids **directly** (no extra BOS);
   positions align 1:1 with Neuronpedia's `values`.
3. **Comparison region (the big one):** Neuronpedia masks BOS and reports
   activations only up to an example-specific window L (mid-text, not a document
   boundary). Run over the full sequence and the local SAE is a **superset** of
   NP's reported activations — so full-sequence Pearson under-reads (mean ~0.83).
   The honest metric is Pearson over **NP-reported positions** (mean 0.9999): where
   NP reports, local matches to ~0.5%, with identical peak position+magnitude.
   `n_local_extra` in the results records the superset size per feature.
4. **dtype:** bf16 vs float32 capture made no difference here (mean r 0.7405 →
   0.7427 before the region fix) — precision was never the issue; the window was.

`MANIFEST.lock.json` is the single source of pinned SAE numbers.
