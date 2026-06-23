"""E-CA / encode (ISOLATED sae venv, torch 2.7.1 + sae-lens) — load the Gemma
Scope 2 SAEs for each of the 18 layers, encode captured residuals, and binarize
to produce cellular-automata grids.

For each captured .npz from e_ca_capture.py:
  - Loads residuals [n_layers, seq_len, d_model].
  - For each layer, loads the layer's SAE (lazily — one at a time to save RAM),
    runs sae.encode(resid[layer]) -> [seq_len, d_sae].
  - Binarizes: grid = (features > 0) — the SAE's ReLU already enforces sparsity;
    threshold at > 0 is the natural boundary.
  - Saves grid [n_layers, seq_len, d_sae] (uint8) and grid_lastpos
    [n_layers, d_sae] (the last-token slice, the primary analysis target).

CRITICAL: this script runs in the SAE VENV. Do NOT import anything from
gemma4_lab or the core conda env.

Run (SAE venv):
    calibration/.venv-sae/bin/python calibration/e_ca/e_ca_encode.py [--n-prompts N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE

HERE = Path(__file__).resolve().parent


def _load_sae(layer: int, release: str, sae_id_pattern: str, expect_d_in: int) -> SAE:
    """Load a single layer's SAE, validate d_in and hook, return on CPU float32."""
    sae_id = sae_id_pattern.format(N=layer)
    print(f"    loading SAE {release} / {sae_id} ...", end="", flush=True)
    res = SAE.from_pretrained(release, sae_id)
    sae = res[0] if isinstance(res, tuple) else res
    sae = sae.to("cpu").to(torch.float32)
    d_in = int(getattr(sae.cfg, "d_in", -1))
    hook = (getattr(getattr(sae.cfg, "metadata", None), "hook_name", None)
            or getattr(sae.cfg, "hook_name", None))
    assert d_in == expect_d_in, f"SAE d_in {d_in} != {expect_d_in}"
    assert str(layer) in str(hook) and "resid" in str(hook), \
        f"SAE hook {hook} doesn't match layer {layer} resid"
    d_sae = int(getattr(sae.cfg, "d_sae", -1))
    print(f" d_in={d_in} d_sae={d_sae} hook={hook}", flush=True)
    return sae


def main() -> int:
    parser = argparse.ArgumentParser(description="E-CA encode: SAE grids from residuals")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="number of .npz files per label to process (default: all)")
    parser.add_argument("--model-size", type=str, choices=["270m", "4b"], default="270m",
                        help="model size (270m or 4b)")
    parser.add_argument("--release", type=str, default=None,
                        help="SAE release name (default: auto-detected based on model size)")
    parser.add_argument("--sae-id-pattern", type=str, default=None,
                        help="SAE ID pattern (default: auto-detected based on model size)")
    args = parser.parse_args()

    # ── set model variables dynamically ──────────────────────────────
    if args.model_size == "270m":
        release = args.release or "gemma-scope-2-270m-it-res-all"
        sae_id_pattern = args.sae_id_pattern or "layer_{N}_width_16k_l0_small"
        expect_d_in = 640
        expect_n_layers = 18
    else:
        release = args.release or "gemma-scope-2-4b-it-res-all"
        sae_id_pattern = args.sae_id_pattern or "layer_{N}_width_16k_l0_small"
        expect_d_in = 2560
        expect_n_layers = 34

    cap_dir = HERE / "captures" / args.model_size
    grid_dir = HERE / "grids" / args.model_size
    grid_dir.mkdir(parents=True, exist_ok=True)

    # enumerate captures
    npz_files = sorted(cap_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no .npz files in {cap_dir} — run e_ca_capture.py first")
    if args.n_prompts is not None:
        known = [f for f in npz_files if f.stem.startswith("known_")][:args.n_prompts]
        unknown = [f for f in npz_files if f.stem.startswith("unknown_")][:args.n_prompts]
        npz_files = sorted(known + unknown)
    print(f"encoding {len(npz_files)} captures from {cap_dir}", flush=True)

    # ── pre-load SAEs lazily (layer by layer, cached for all prompts) ────
    # Since 16k×640 is small (~40 MB per SAE), we CAN keep all 18 in memory.
    # But we load lazily to print progress and validate each one.
    sae_cache: dict[int, SAE] = {}

    def get_sae(layer: int) -> SAE:
        if layer not in sae_cache:
            sae_cache[layer] = _load_sae(layer, release, sae_id_pattern, expect_d_in)
        return sae_cache[layer]

    all_l0: list[np.ndarray] = []  # [n_prompts, n_layers]

    for fi, npz_path in enumerate(npz_files):
        d = np.load(npz_path, allow_pickle=True)
        residuals = d["residuals"]  # [n_layers, seq_len, d_model]
        prompt_id = str(d["prompt_id"])
        label = str(d["label"])
        n_layers = int(d["n_layers"])
        seq_len = int(d["seq_len"])

        assert residuals.shape[0] == n_layers, \
            f"{npz_path.name}: n_layers {residuals.shape[0]} != {n_layers}"
        assert n_layers == expect_n_layers, \
            f"{npz_path.name}: expected {expect_n_layers} layers, got {n_layers}"
        assert residuals.shape == (n_layers, seq_len, expect_d_in), \
            f"{npz_path.name}: shape {residuals.shape} != ({n_layers}, {seq_len}, {expect_d_in})"

        layer_grids = []        # list of [seq_len, d_sae] uint8
        layer_l0 = []           # L0 per layer
        d_sae_actual = None

        for li in range(n_layers):
            sae = get_sae(li)
            resid_t = torch.tensor(residuals[li], dtype=torch.float32)  # [seq_len, d_model]

            with torch.no_grad():
                feats = sae.encode(resid_t)
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.float().cpu()  # [seq_len, d_sae]

            # numerical health
            n_bad = int((~torch.isfinite(feats)).sum())
            if n_bad:
                raise ValueError(
                    f"{npz_path.name} layer {li}: {n_bad} non-finite SAE activations"
                )

            if d_sae_actual is None:
                d_sae_actual = feats.shape[1]
            assert feats.shape == (seq_len, d_sae_actual), \
                f"layer {li}: shape {tuple(feats.shape)} != ({seq_len}, {d_sae_actual})"

            # binarize: ReLU SAE — features > 0 are the active set
            binary = (feats > 0).to(torch.uint8).numpy()  # [seq_len, d_sae]
            layer_grids.append(binary)

            # L0 = mean number of active features per position
            l0 = float(binary.sum(axis=1).mean())
            layer_l0.append(l0)

        # stack into [n_layers, seq_len, d_sae]
        grid = np.stack(layer_grids, axis=0)  # [n_layers, seq_len, d_sae]
        assert grid.shape == (n_layers, seq_len, d_sae_actual)

        # last-position slice: the primary analysis target
        grid_lastpos = grid[:, -1, :]  # [n_layers, d_sae]
        assert grid_lastpos.shape == (n_layers, d_sae_actual)

        l0_per_layer = np.array(layer_l0, dtype=np.float32)
        all_l0.append(l0_per_layer)

        out = grid_dir / f"{npz_path.stem}.npz"
        np.savez(
            out,
            grid=grid,
            grid_lastpos=grid_lastpos,
            l0_per_layer=l0_per_layer,
            prompt_id=str(prompt_id),
            label=str(label),
        )
        mean_l0 = float(l0_per_layer.mean())
        print(f"  [{fi+1}/{len(npz_files)}] {prompt_id}: "
              f"grid {grid.shape} (uint8), mean L0={mean_l0:.1f}, "
              f"L0 range [{l0_per_layer.min():.1f}, {l0_per_layer.max():.1f}] "
              f"-> {out.name}", flush=True)

    # ── summary ──────────────────────────────────────────────────────────
    if not all_l0:
        print("no captures processed!")
        return 1

    all_l0_arr = np.stack(all_l0, axis=0)  # [n_prompts, n_layers]
    mean_l0_per_layer = all_l0_arr.mean(axis=0)  # [n_layers]
    grand_mean = float(all_l0_arr.mean())

    n_grids = len(list(grid_dir.glob("*.npz")))
    print(f"\n{'='*70}")
    print(f"E-CA encode done: {n_grids} grid files in {grid_dir}")
    print(f"{'='*70}")
    print(f"  grand mean L0 = {grand_mean:.1f} active features per position")
    print(f"  L0 per layer (mean over all prompts):")
    for li in range(expect_n_layers):
        print(f"    layer {li:2d}: L0 = {mean_l0_per_layer[li]:.1f}")
    if grand_mean < 5.0:
        print(f"  WARNING: grand mean L0 {grand_mean:.1f} is very low — "
              "grids are trivially sparse, check SAE quality")
    elif grand_mean > 500.0:
        print(f"  WARNING: grand mean L0 {grand_mean:.1f} is very high — "
              "grids may be too dense, check SAE threshold")
    else:
        print(f"  L0 looks reasonable ✓")
    print(f"  SAE release: {release}")
    print(f"  SAE id pattern: {sae_id_pattern}")
    print(f"  env: sae-lens (torch {torch.__version__})")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
