"""E-CA metrics — cellular-automata-framing analysis for SAE feature grids.

Treats layers as time-steps and SAE features as cells in a 1-D CA.
Each cell is binary (active / inactive at a given layer).

Grid layout
-----------
- Primary shape: ``[n_layers, n_features]``  (binary 0/1, aggregated at last
  token position).
- Optional shape: ``[n_layers, seq_len, n_features]``  — the per-position
  variant is reshaped to ``[n_layers, seq_len * n_features]`` before
  analysis so the same metrics apply.

All functions are pure numpy / scipy — **no** torch, **no** model, **no** SAE
dependency — so the module works identically in the core conda env and in
``.venv-sae``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_grid(grids: np.ndarray) -> np.ndarray:
    """Validate and normalise grid to 2-D ``[n_layers, n_features]``.

    Accepts 2-D or 3-D arrays.  Raises ``ValueError`` on unexpected shapes.
    """
    if grids.ndim == 3:
        n_layers, seq_len, n_features = grids.shape
        grids = grids.reshape(n_layers, seq_len * n_features)
    elif grids.ndim != 2:
        raise ValueError(
            f"Expected grid of shape [n_layers, n_features] or "
            f"[n_layers, seq_len, n_features], got shape {grids.shape}"
        )
    return grids


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def jaccard_consecutive(grids: np.ndarray) -> np.ndarray:
    """Jaccard similarity between consecutive layers.

    Parameters
    ----------
    grids : np.ndarray
        Binary grid of shape ``[n_layers, n_features]``.

    Returns
    -------
    np.ndarray
        Shape ``[n_layers - 1]``.  ``J(t, t+1)`` is the Jaccard index
        between the active sets at layers *t* and *t + 1*.  When the union
        is empty (both layers fully dead) the value is ``1.0``.
    """
    grids = _validate_grid(grids)
    n_layers = grids.shape[0]
    result = np.empty(n_layers - 1, dtype=np.float64)
    for t in range(n_layers - 1):
        intersection = np.sum(grids[t] & grids[t + 1])
        union = np.sum(grids[t] | grids[t + 1])
        result[t] = 1.0 if union == 0 else intersection / union
    return result


def feature_lifetimes(grids: np.ndarray) -> np.ndarray:
    """Lengths of consecutive-active runs across all features.

    Parameters
    ----------
    grids : np.ndarray
        Binary grid of shape ``[n_layers, n_features]``.

    Returns
    -------
    np.ndarray
        1-D array of integers — every consecutive run of ``1`` s found
        column-wise.  Ready for histogramming.  Empty if no features are
        ever active.
    """
    grids = _validate_grid(grids)
    n_layers, n_features = grids.shape
    runs: list[int] = []
    for f in range(n_features):
        col = grids[:, f]
        run_len = 0
        for t in range(n_layers):
            if col[t]:
                run_len += 1
            else:
                if run_len > 0:
                    runs.append(run_len)
                run_len = 0
        if run_len > 0:
            runs.append(run_len)
    return np.array(runs, dtype=np.int64)


def per_layer_entropy(grids: np.ndarray) -> np.ndarray:
    """Shannon entropy of the binary activation distribution per layer.

    Parameters
    ----------
    grids : np.ndarray
        Binary grid of shape ``[n_layers, n_features]``.

    Returns
    -------
    np.ndarray
        Shape ``[n_layers]``.  ``H = -p log2(p) - (1-p) log2(1-p)``
        where *p* is the fraction of active features.  ``H = 0`` when
        ``p ∈ {0, 1}``.
    """
    grids = _validate_grid(grids)
    n_layers, n_features = grids.shape
    result = np.empty(n_layers, dtype=np.float64)
    for t in range(n_layers):
        p = np.sum(grids[t]) / n_features
        if p == 0.0 or p == 1.0:
            result[t] = 0.0
        else:
            result[t] = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return result


def grid_autocorrelation(
    grids: np.ndarray,
    max_lag: int | None = None,
) -> np.ndarray:
    """Layer-wise autocorrelation of the binary grid.

    For each lag *k*, computes the fraction of features that *match*
    (both 1 or both 0) between layer *t* and layer *t + k*, averaged
    over all valid *t*.

    Parameters
    ----------
    grids : np.ndarray
        Binary grid of shape ``[n_layers, n_features]``.
    max_lag : int | None
        Maximum lag.  Defaults to ``n_layers - 1``.

    Returns
    -------
    np.ndarray
        Shape ``[max_lag]`` with autocorrelation values in ``[0, 1]``.
    """
    grids = _validate_grid(grids)
    n_layers, n_features = grids.shape
    if max_lag is None:
        max_lag = n_layers - 1
    max_lag = min(max_lag, n_layers - 1)
    result = np.empty(max_lag, dtype=np.float64)
    for k in range(1, max_lag + 1):
        matches = 0
        pairs = 0
        for t in range(n_layers - k):
            matches += np.sum(grids[t] == grids[t + k])
            pairs += n_features
        result[k - 1] = matches / pairs if pairs > 0 else 1.0
    return result


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def compute_all_metrics(grids: np.ndarray) -> dict[str, Any]:
    """Compute all CA-framing metrics and return a JSON-serialisable dict.

    Parameters
    ----------
    grids : np.ndarray
        Binary grid of shape ``[n_layers, n_features]``.

    Returns
    -------
    dict
        Keys: ``jaccard``, ``lifetimes``, ``lifetime_mean``,
        ``lifetime_median``, ``entropy``, ``autocorrelation``,
        ``n_layers``, ``n_features``, ``mean_sparsity``.
    """
    grids = _validate_grid(grids)
    n_layers, n_features = grids.shape

    jac = jaccard_consecutive(grids)
    lf = feature_lifetimes(grids)
    ent = per_layer_entropy(grids)
    ac = grid_autocorrelation(grids)
    sparsity = np.mean(np.sum(grids, axis=1) / n_features)

    return {
        "jaccard": jac.tolist(),
        "lifetimes": lf.tolist(),
        "lifetime_mean": float(np.mean(lf)) if len(lf) > 0 else 0.0,
        "lifetime_median": float(np.median(lf)) if len(lf) > 0 else 0.0,
        "entropy": ent.tolist(),
        "autocorrelation": ac.tolist(),
        "n_layers": int(n_layers),
        "n_features": int(n_features),
        "mean_sparsity": float(sparsity),
    }


# ---------------------------------------------------------------------------
# Partition comparison
# ---------------------------------------------------------------------------

def compare_partitions(
    grids_a: list[np.ndarray],
    grids_b: list[np.ndarray],
    labels: tuple[str, str] = ("A", "B"),
) -> dict[str, Any]:
    """Compare CA metrics between two groups of grids.

    Computes per-grid metrics, aggregates means per group, and runs a
    Mann–Whitney U test on the per-grid mean-Jaccard values.

    Parameters
    ----------
    grids_a, grids_b : list[np.ndarray]
        Each element is a binary grid ``[n_layers, n_features]``.
    labels : tuple[str, str]
        Human-readable names for the two groups.

    Returns
    -------
    dict
        Per-group summary stats, deltas, and Mann–Whitney U p-value.
    """
    def _group_stats(grid_list: list[np.ndarray]) -> dict[str, Any]:
        all_metrics = [compute_all_metrics(g) for g in grid_list]
        jac_means = [float(np.mean(m["jaccard"])) for m in all_metrics]
        ent_means = [float(np.mean(m["entropy"])) for m in all_metrics]
        lt_means = [m["lifetime_mean"] for m in all_metrics]
        sparsities = [m["mean_sparsity"] for m in all_metrics]
        return {
            "n_grids": len(grid_list),
            "jaccard_mean": float(np.mean(jac_means)),
            "jaccard_std": float(np.std(jac_means)),
            "jaccard_per_grid": jac_means,
            "entropy_mean": float(np.mean(ent_means)),
            "lifetime_mean": float(np.mean(lt_means)),
            "sparsity_mean": float(np.mean(sparsities)),
        }

    sa = _group_stats(grids_a)
    sb = _group_stats(grids_b)

    # Mann-Whitney U on per-grid mean Jaccard
    if len(sa["jaccard_per_grid"]) >= 1 and len(sb["jaccard_per_grid"]) >= 1:
        u_stat, p_value = stats.mannwhitneyu(
            sa["jaccard_per_grid"],
            sb["jaccard_per_grid"],
            alternative="two-sided",
        )
    else:
        u_stat, p_value = float("nan"), float("nan")

    return {
        labels[0]: sa,
        labels[1]: sb,
        "delta_jaccard_mean": sa["jaccard_mean"] - sb["jaccard_mean"],
        "delta_entropy_mean": sa["entropy_mean"] - sb["entropy_mean"],
        "delta_lifetime_mean": sa["lifetime_mean"] - sb["lifetime_mean"],
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(p_value),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry-point: load grids from .npz directory, print JSON summary."""
    parser = argparse.ArgumentParser(
        description="Compute E-CA metrics on saved SAE feature grids.",
    )
    parser.add_argument(
        "grid_dir",
        type=str,
        help="Directory containing .npz files. Each file must have a 'grid' key "
             "with shape [n_layers, n_features].",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="grid",
        help="Key inside each .npz file to read (default: 'grid').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path. Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    grid_dir = pathlib.Path(args.grid_dir)
    if not grid_dir.is_dir():
        print(f"ERROR: {grid_dir} is not a directory", file=sys.stderr)
        return 1

    npz_files = sorted(grid_dir.glob("*.npz"))
    if not npz_files:
        print(f"ERROR: no .npz files found in {grid_dir}", file=sys.stderr)
        return 1

    all_results: dict[str, Any] = {}
    known_grids = []
    unknown_grids = []

    for npz_path in npz_files:
        data = np.load(npz_path)
        if args.key not in data:
            print(
                f"WARNING: '{args.key}' not in {npz_path.name}, skipping",
                file=sys.stderr,
            )
            continue
        grid = data[args.key].astype(np.int64)
        metrics = compute_all_metrics(grid)
        all_results[npz_path.stem] = metrics

        if npz_path.stem.startswith("known_"):
            known_grids.append(grid)
        elif npz_path.stem.startswith("unknown_"):
            unknown_grids.append(grid)

    if known_grids and unknown_grids:
        print(f"Comparing partitions: {len(known_grids)} known vs {len(unknown_grids)} unknown", file=sys.stderr)
        comparison = compare_partitions(known_grids, unknown_grids, ("known", "unknown"))
        all_results["group_comparison"] = comparison

    output_json = json.dumps(all_results, indent=2)
    if args.output:
        pathlib.Path(args.output).write_text(output_json)
        print(f"Wrote {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
