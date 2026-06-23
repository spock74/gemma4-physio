# Tasks Checklist

- `[x] Implement scripts/o6_topology.py`
  - `[x] Load model, tokenizer, and setup target layers`
  - `[x] Calculate primary knowledge direction `v1``
  - `[x] Implement Gram-Schmidt orthogonalized noise control generation for `K=30` control planes`
  - `[x] Implement Layer 12 intervention hook (Einsum projection, RMSNorm bypass norm preservation)`
  - `[x] Implement Layer 13 activation capture for newly generated tokens`
  - `[x] Integrate `ripser` to calculate H1 persistence and Betti-0 counts`
  - `[x] Calculate KL divergence of generated logits against baseline`
  - `[x] Save grid results to `results/topology_sweep.json``
  - `[x] Implement persistence barcode plotting for H1 cycles`
- `[x] Run a quick dry-run of the script to verify no crashes`
- `[x] Run the full sweep over the 10 angles, K=30 planes, R=15000`
- `[x] Update walkthrough artifact`
