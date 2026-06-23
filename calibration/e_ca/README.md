# E-CA — Cellular Automata Dynamics in Sparse Transformer Representations

Can we understand layer-by-layer computation in a small transformer as a
cellular automaton evolving on the SAE feature grid?  Each prompt produces an
`[n_layers × d_sae]` binary grid (which features are active at each layer for a
given token position).  The "known" vs "unknown" entity-knowledge split lets us
test whether factual recall leaves a distinct spatiotemporal signature in this
grid — whether the automaton-like evolution of sparse features across layers
differs systematically when the model "knows" the answer versus when it does
not.

## Target

| | |
|---|---|
| Model | `google/gemma-3-270m-it` (text-only, 18 layers, d_model 640) |
| SAE release | `gemma-scope-2-270m-it-res` |
| SAE id pattern | `layer_{N}_width_16k_l0_medium` (N = 0..17) |
| SAE width | d_sae = 16384 |
| Hook pattern | `blocks.{N}.hook_resid_post` |
| Corpus | `data/eval/entity_knowledge_contrast.json` (~40 known + ~40 unknown) |

## Two-env split (same rule as O2)

`sae-lens` pins `torch==2.7.1`; the core conda env is `torch 2.11.0`.  They
never share a process.  Reuses the existing **`calibration/.venv-sae`**
(sae-lens 6.44.2).

- **Core conda env** — runs the model and captures residual activations at all
  18 layers (`e_ca_capture.py`).
- **Isolated `.venv-sae`** — loads each SAE, encodes saved activations, and
  binarizes into grids (`e_ca_encode.py`).  Captured residuals are float
  `.npz`, so encoding on torch 2.7.1 is version-agnostic.

## Pipeline (run in order)

```
# Stage 1: capture residuals (CORE conda env)
python calibration/e_ca/e_ca_capture.py [--n-prompts N]

# Stage 2: SAE encode + binarize (SAE venv)
calibration/.venv-sae/bin/python calibration/e_ca/e_ca_encode.py [--n-prompts N]
```

Use `--n-prompts 2` for smoke testing (processes 2 known + 2 unknown).

## What the outputs contain

### captures/ (from Stage 1)

One `.npz` per prompt with:
- `residuals`: float32 `[n_layers, seq_len, d_model]` — resid-post at all 18
  layers, all token positions
- `prompt_id`, `label`, `n_layers`, `seq_len`: metadata

### grids/ (from Stage 2)

One `.npz` per prompt with:
- `grid`: uint8 `[n_layers, seq_len, d_sae]` — binary (0/1), which SAE features
  are active at each layer × position
- `grid_lastpos`: uint8 `[n_layers, d_sae]` — the last-token slice of the grid,
  the primary analysis target (the position where the model must produce the
  next-token prediction)
- `l0_per_layer`: float32 `[n_layers]` — mean L0 (number of active features per
  position) for each layer
- `prompt_id`, `label`: metadata

## What the metrics measure

- **L0 per layer:** mean number of active SAE features per position.  This is
  the sparsity diagnostic: if L0 is too low the grid is trivially empty; if too
  high the binarization is meaningless.  Healthy range is roughly 10–200 for
  width-16k SAEs.
- **Grid structure (downstream):** the binary grids are the raw material for
  cellular-automata-style analysis — birth/death/persistence of features across
  layers, Hamming distance between adjacent layers, cluster structure in the
  known vs unknown split.

## Known limitations

1. **Raw text, not chat:** prompts are fed as raw text (no chat template)
   because the Gemma Scope 2 SAEs were trained on Pile.  The model was
   instruction-tuned, so its internal representations may differ slightly from
   the SAE's training distribution.  This is a deliberate trade-off: raw text is
   closer to on-distribution for the SAE encoder.
2. **Single position focus:** `grid_lastpos` collapses the full sequence to the
   last token.  Multi-position analysis is possible from the full `grid` but is
   not the default analysis target.
3. **Memory:** the full grid `[18, seq_len, 16384]` is large (~1.2 MB per
   prompt as uint8).  With ~80 prompts this is manageable (~100 MB total).
4. **Binarization threshold:** `> 0` is the natural ReLU boundary but discards
   magnitude information.  The raw activations are available in the captured
   residuals for future magnitude-aware analysis.

## Pointer

See the experiment spec for the full E-CA design rationale and downstream
analysis plan.
