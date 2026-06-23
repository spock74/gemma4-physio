# E1d — SAE fidelity calibration (third-party-attested positive control)

Recovered + persisted 2026-06-11 from the 2026-06-10 working session
("Gemma4 E2B proposal evaluation"). This decision and the runnable Code prompt
were agreed verbally but never committed; this file is the durable record.
Runnable task: `.agent/queue/002-*.md` (`target: code`).

## Decision: Path A — one more SAE round on Gemma 4 (NOT a downgrade to Gemma 3)

The earlier verdict "no public SAE for Gemma 4 E2B" is **false** and must be
corrected in `00-methodology-and-value.md` and `phase6-report.html` (Step E below).

- SAE release: **`decoderesearch/gemma-4-saes`** (chanind / Decode Research), `library: saelens`.
- Load (the release id *is* the HF repo, not a `pretrained_saes.yaml` entry):
  ```python
  from sae_lens import SAE
  sae, cfg, sparsity = SAE.from_pretrained(
      "decoderesearch/gemma-4-saes", "gemma-4-e2b/btk-mat-layer-28-k-100")
  ```
- E2B SAEs exist at **layers 6, 17, 28**. **Layer 28 is adjacent to the E1 `d_know`
  extraction layer 26** → the old "SAE is at L17, far from L26" caveat dissolves; we
  validate the apparatus next to the real experiment, and can optionally re-extract
  `d_know` at L28 for a perfect-aligned control.
- Source naming: `btk-mat-layer-28-k-100` = BatchTopK + Matryoshka, layer 28, k=100.

## Why (the gap this closes)

E1c's positive controls were unsatisfying: `d_unembed` passes but is near-tautological;
`d_refusal` is decodable yet does not flip behavior. We still lack a **non-tautological,
third-party-attested** control at the feature/logit level. Reproducing a Neuronpedia
feature's published activation locally is exactly that: if our local activation tracks
theirs, the apparatus (layer indexing, hook point, chat template, dtype) is validated
against an external source on the real model.

## The calibration ladder (spectrophotometer analogy)

Isolated folder `calibration/neuronpedia_fidelity/` (outside `src/`), its own protocol,
with a `MANIFEST.lock.json` as the single source of pinned values (no guessed numbers).

- **Gate A — instrument.** Load the SAE, read and pin `cfg`: `hook_name`, `dtype`,
  `d_in` (must be 1536), `d_sae`, normalization; plus the `google/gemma-4-E2B-it`
  revision SHA. `d_in != 1536` → stop.
- **Gate B — blank.** Assert `n_layers == 35`, `d_model == 1536`. Map `cfg["hook_name"]`
  to the exact capture point in `interp/recorder.py` (`hook_resid_post` of layer L =
  output of L; `hook_resid_pre` of L = output of L-1). Getting this wrong invalidates
  everything; record the mapping in the MANIFEST.
- **Gate C — buffer (this is the minimum shippable result).** Neuronpedia protocol =
  **raw text** (no chat template), their tokenization/BOS. Pick 1–3 features with clear
  explanations, freeze their top-activating text + per-token values in `reference/`,
  run that same raw text locally, capture at the Gate-B point, apply **`sae.encode()`**
  (not raw `W_enc @ x` — encode applies the trained normalization), and PASS iff the
  local activation tracks Neuronpedia (same top position, high correlation within
  tolerance). Fail → STOP and report which Gemma-4 detail breaks (hook point?
  normalization? BOS? dtype?). The failure *is* the finding; do not force forward.
- **Step D — bonus (only if C passed and time remains).** Method-level positive control
  at the logit: find an entity/factual-recall feature, ablate it (subtract `W_dec[i]` or
  zero+reconstruct) at layer 28, final position, measure gold-token logit drop on known
  items vs an N≥50 null. Optional: cosine between re-extracted `d_know@28` and the
  feature's `W_dec`.
- **Step E — register + correct.** Write results into `phase6-report.html` and
  `02-results.md`; correct the false "no public SAE" claim in
  `00-methodology-and-value.md` and the report.

## Guardrails (where this burns the time window if ignored)

1. **No TransformerLens / HookedSAETransformer.** TransformerLens does not support
   Gemma 4. Load only the SAE *weights* via `SAE.from_pretrained`; apply `sae.encode()`
   to activations captured by the existing HF-hook `recorder.py`. Model stays `GemmaLocal`.
2. **`sae-lens` is a calibration-only dep** — `[project.optional-dependencies].calibration`,
   never the core locked stack.
3. **Use `sae.encode()`**, not `W_enc @ x` raw.
4. **Map `hook_name` (resid_pre vs post)** to the exact capture point before Gate C.
5. Logfire on every capture; secrets (`NEURONPEDIA_API_KEY`, `HF_TOKEN`) from env via
   `config.py`; `-it`; English in code/docs; commit per green gate; do not tune.

## Where execution must happen

Loading Gemma E2B + capturing activations needs **MPS / host** → this is a `target: code`
task. Cowork can refine the spec and fetch the Neuronpedia reference data (HTTP), but the
capture runs on the Code host. (Confirmed 2026-06-11: the Code tab is bare-host macOS with
MPS — see `working-modes` memory.)

Neuronpedia API reference: `docs/research/neuronpedia-api.md`. Local SAE repo note: the
user also cloned `neuronpedia` at `~/Documents/PROJETOS/main-projects/neuronpedia`.
LARQL (`github.com/chrishayuk/larql`) was assessed as a useful *external forward-fidelity
oracle* (its `shannon verify` = a ready white-cuvette cross-check), **not** a dependency
and **not** an SAE — keep it as an optional cross-check, do not conflate with this control.
