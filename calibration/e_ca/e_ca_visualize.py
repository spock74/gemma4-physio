"""E-CA Visualization — Generate profile plots and activation heatmaps.

Plots:
1. Jaccard profile across layers (known vs unknown).
2. Entropy profile across layers (known vs unknown).
3. Autocorrelation profile across lag layers (known vs unknown).
4. Binary activation heatmaps for exemplar prompts, showing the cascade of active
   SAE features across layers.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load results.json and return the group_comparison and individual results."""
    with open(results_path) as f:
        data = json.load(f)
    group_comp = data.pop("group_comparison", None)
    return group_comp, data


def plot_profile(
    known_data: list[list[float]],
    unknown_data: list[list[float]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: pathlib.Path,
    x_values: np.ndarray | None = None,
) -> None:
    """Plot profile of known vs unknown with standard error bands."""
    k_arr = np.array(known_data)  # [n_prompts, n_points]
    u_arr = np.array(unknown_data)

    n_points = k_arr.shape[1]
    if x_values is None:
        x = np.arange(n_points)
    else:
        x = x_values

    # compute means and standard errors (SEM)
    k_mean = k_arr.mean(axis=0)
    k_sem = k_arr.std(axis=0) / np.sqrt(k_arr.shape[0])
    u_mean = u_arr.mean(axis=0)
    u_sem = u_arr.std(axis=0) / np.sqrt(u_arr.shape[0])

    plt.figure(figsize=(8, 5))
    plt.plot(x, k_mean, label="Known", color="#1f77b4", linewidth=2.5, marker="o")
    plt.fill_between(x, k_mean - k_sem, k_mean + k_sem, color="#1f77b4", alpha=0.15)

    plt.plot(x, u_mean, label="Unknown", color="#ff7f0e", linewidth=2.5, marker="s")
    plt.fill_between(x, u_mean - u_sem, u_mean + u_sem, color="#ff7f0e", alpha=0.15)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved profile plot to {output_path}")


def generate_heatmap(
    grid_path: pathlib.Path,
    output_path: pathlib.Path,
    prompt_id: str,
    label: str,
) -> None:
    """Generate a binary feature activation cascade heatmap for a prompt."""
    data = np.load(grid_path)
    # shape [n_layers, d_sae]
    grid = data["grid_lastpos"]
    n_layers, d_sae = grid.shape

    # Find features that are active in at least one layer
    active_mask = grid.sum(axis=0) > 0
    active_idx = np.where(active_mask)[0]
    active_grid = grid[:, active_idx]  # [n_layers, n_active_features]

    n_active = len(active_idx)
    if n_active == 0:
        print(f"  Skipping heatmap for {prompt_id}: no active features")
        return

    # Sort active features by their 'birth layer' (first layer where active)
    # For features born at the same layer, sort by sum of activation across layers
    birth_layers = []
    for f in range(n_active):
        first_act = np.where(active_grid[:, f] > 0)[0][0]
        birth_layers.append((first_act, -int(active_grid[:, f].sum()), f))

    birth_layers.sort()
    sorted_indices = [idx for (_, _, idx) in birth_layers]
    sorted_grid = active_grid[:, sorted_indices]  # [n_layers, n_active_features]

    plt.figure(figsize=(10, 6))
    # Transpose so features are on y-axis and layers on x-axis (evolving left to right)
    plt.imshow(
        sorted_grid.T,
        cmap="binary",
        aspect="auto",
        interpolation="none",
        origin="upper",
    )
    plt.xlabel("Layer (Time Step $t$)", fontsize=12)
    plt.ylabel(f"Active Features ({n_active} sorted by birth)", fontsize=12)
    model_name = "Gemma 3 4b-it" if n_layers == 34 else "Gemma 3 270m-it"
    plt.title(
        f"Activation Cascade: {prompt_id} ({label.capitalize()})\n"
        f"{model_name} SAE Feature Evolution",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xticks(np.arange(n_layers))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved heatmap to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="E-CA visualization tool")
    parser.add_argument(
        "--results",
        type=str,
        default="calibration/e_ca/results.json",
        help="Path to results.json",
    )
    parser.add_argument(
        "--grids-dir",
        type=str,
        default="calibration/e_ca/grids",
        help="Path to grids directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="calibration/e_ca/plots",
        help="Directory to save generated plots",
    )
    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    grids_dir = pathlib.Path(args.grids_dir)
    out_dir = pathlib.Path(args.output_dir)

    if not results_path.is_file():
        print(f"ERROR: results file {results_path} not found")
        return 1
    if not grids_dir.is_dir():
        print(f"ERROR: grids directory {grids_dir} not found")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating E-CA visualizations in {out_dir}...")

    # Load results
    group_comp, ind_results = load_results(results_path)

    # 1. Gather profile data
    k_jaccard, u_jaccard = [], []
    k_entropy, u_entropy = [], []
    k_autocorr, u_autocorr = [], []

    for name, metrics in ind_results.items():
        is_known = name.startswith("known_")
        if is_known:
            k_jaccard.append(metrics["jaccard"])
            k_entropy.append(metrics["entropy"])
            k_autocorr.append(metrics["autocorrelation"])
        else:
            u_jaccard.append(metrics["jaccard"])
            u_entropy.append(metrics["entropy"])
            u_autocorr.append(metrics["autocorrelation"])

    # 2. Save profile plots
    n_layers = len(k_entropy[0]) if k_entropy else 18
    plot_profile(
        k_jaccard,
        u_jaccard,
        x_label="Transition (Layer $t \\rightarrow t+1$)",
        y_label="Jaccard Similarity",
        title="Feature Grid Stability (Consecutive Jaccard)",
        output_path=out_dir / "jaccard_profile.png",
        x_values=np.arange(n_layers - 1),
    )

    plot_profile(
        k_entropy,
        u_entropy,
        x_label="Layer",
        y_label="Shannon Entropy (bits)",
        title="Feature Activation Entropy Profile",
        output_path=out_dir / "entropy_profile.png",
        x_values=np.arange(n_layers),
    )

    plot_profile(
        k_autocorr,
        u_autocorr,
        x_label="Lag (Layers)",
        y_label="Autocorrelation",
        title="Spatiotemporal Autocorrelation of Feature Trajectory",
        output_path=out_dir / "autocorrelation_profile.png",
        x_values=np.arange(1, n_layers),
    )

    # 3. Generate heatmaps for exemplar prompts
    # Select 2 known and 2 unknown prompts
    known_keys = sorted([k for k in ind_results.keys() if k.startswith("known_")])[:2]
    unknown_keys = sorted([k for k in ind_results.keys() if k.startswith("unknown_")])[:2]

    for k in known_keys:
        generate_heatmap(
            grids_dir / f"{k}.npz",
            out_dir / f"heatmap_{k}.png",
            prompt_id=k,
            label="known",
        )

    for uk in unknown_keys:
        generate_heatmap(
            grids_dir / f"{uk}.npz",
            out_dir / f"heatmap_{uk}.png",
            prompt_id=uk,
            label="unknown",
        )

    print("Visualization complete!")
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    sys.exit(main())
