# Implementation Plan: Autoregressive TDA Sweep (o6_topology.py)

We will implement the new Python script `scripts/o6_topology.py` to perform Topological Data Analysis (TDA) on the `google/gemma-3-4b-it` residual stream. The script will perform a 2D rotational sweep in Layer 12, capture autoregressive activations at Layer 13, and compute persistent homology ($H_1$ persistence and Betti-0) using `ripser`.

## Proposed Changes

### [NEW] [o6_topology.py](../../scripts/o6_topology.py)
We will create a new script in `scripts/o6_topology.py` implementing the following core stages:

1. **Intervention Setup (Layer 12)**:
   * Load the tokenizer and the model `google/gemma-3-4b-it` on `mps`.
   * Capture Layer 12 activations for 8 contrast prompts to compute `v1` (diff-of-means factual direction).
   * Generate `K=30` random vectors from isotropic Gaussian noise, orthogonalize them against `v1` via Gram-Schmidt, and normalize them to yield `v2_k`.

2. **Hook and Tensor Algebra**:
   * Implement a hook on Layer 12. For activation tensor `h` of shape `[batch, seq_len, d_model]`, we:
     * Record the clean $L_2$ norm along the last dimension: `clean_norm = h.norm(dim=-1, keepdim=True)`.
     * Project `h` orthogonally via `einsum`:
       ```python
       proj_v1 = torch.einsum('bsd,d->bs', h, v1).unsqueeze(-1) * v1
       proj_v2 = torch.einsum('bsd,d->bs', h, v2_k).unsqueeze(-1) * v2_k
       h_perp = h - (proj_v1 + proj_v2)
       ```
     * Add the rotational perturbation: `perturbation = R * (cos(theta)*v1 + sin(theta)*v2_k)`.
     * Rescale the patched tensor to match `clean_norm`:
       ```python
       patched_raw = h_perp + perturbation
       patched_norm = patched_raw.norm(dim=-1, keepdim=True) + 1e-8
       patched = patched_raw * (clean_norm / patched_norm)
       ```

3. **Autoregressive Generation and Point Cloud Extraction**:
   * Generate 40 tokens autoregressively under the intervention using `model.generate(max_new_tokens=40, return_dict_in_generate=True, output_logits=True)`.
   * Capture Layer 13 activations *only* when `h.shape[1] == 1` (corresponding to the generated tokens).
   * Extract these activations as a NumPy array of shape `[seq_len_generated, d_model]`.

4. **Topological Computation (`ripser`)**:
   * Call `ripser.ripser(point_cloud, maxdim=1)` to compute the persistence diagrams.
   * Compute **Total H1 Persistence** by summing `death - birth` for all finite 1D cycles in `result['dgms'][1]`.
   * Compute Betti-0 count from `result['dgms'][0]` at a filtration threshold set to `0.5 * mean_baseline_pairwise_distance` of the baseline generation point cloud.
   * Compute the KL divergence of the generated token logits against the baseline generation logits.

5. **I/O and Visualization**:
   * Run the sweep over a grid of:
     * Angles `theta`: `[0, 40, 80, 120, 160, 200, 240, 280, 320, 360]`.
     * Steer magnitudes `R`: `[15000.0]`.
     * Control planes `K`: `30` random orthogonal directions.
   * Save the results in `results/topology_sweep.json`.
   * Plot the persistence barcodes for the $H_1$ cycles and save them as `.png` files in `docs/antigr_reports/` (e.g. `docs/antigr_reports/h1_barcodes_theta_*.png`).

## Verification Plan

### Automated Verification
* Run `python scripts/o6_topology.py` to execute a quick dry-run (e.g. with `K=2` and 3 angles) to verify tensor shapes and ensure `ripser` and plotting run without crashes.
* Run the full TDA sweep script and verify that `results/topology_sweep.json` is successfully created.
