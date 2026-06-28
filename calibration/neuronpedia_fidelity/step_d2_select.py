"""STEP D2 stage 1 — fresh null + characterization for feature 1035 (sae venv).

Feature 1035 was found POST-HOC as the max drop inside STEP D's null — testing it
against that same null would be circular (max-of-sample beats its sample by
construction). This stage draws a FRESH frequency-matched null of N=50 features,
excluding {1035, 1007} and all 50 previous null features, and characterizes 1035
(train firing rate, known/unknown acts, where it ranked in the train-diff ordering,
cosines to d_know@28 and W_dec[1007], activation on the VAL items to be ablated).

Pre-registered gate (stage 2): drop > 0 AND percentile >= 95 in the FRESH null.
The +20.68 magnitude estimate carries winner's curse from the original selection
and is labeled as such — SPECIFICITY is the tested claim.

Run (SAE VENV):  calibration/.venv-sae/bin/python calibration/neuronpedia_fidelity/step_d2_select.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE

RELEASE = "decoderesearch/gemma-4-saes"
SAE_ID = "gemma-4-e2b/btk-mat-layer-28-k-100"
TARGET = 1035
N_NULL = 50
HERE = Path(__file__).resolve().parent
CAP = HERE / "captures"
RES = HERE / "results"


def unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def main() -> int:
    d = np.load(CAP / "step_d_data.npz")
    prev = np.load(CAP / "step_d_directions.npz")
    resids = torch.tensor(d["resids"], dtype=torch.float32)
    labels = d["labels"]
    k_tr, k_va = d["k_tr"].tolist(), d["k_va"].tolist()
    u_tr = d["u_tr"].tolist()
    n_known = int((labels == 1).sum())
    unk_tr = [n_known + i for i in u_tr]
    train_rows = k_tr + unk_tr

    print(f"Loading SAE {SAE_ID} ...", flush=True)
    res = SAE.from_pretrained(RELEASE, SAE_ID)
    sae = (res[0] if isinstance(res, tuple) else res).to("cpu").to(torch.float32)
    with torch.no_grad():
        acts = sae.encode(resids)
    acts = (acts[0] if isinstance(acts, tuple) else acts).numpy()

    # --- characterize 1035 ---------------------------------------------------
    diff = acts[k_tr].mean(0) - acts[unk_tr].mean(0)
    diff_rank = int((diff > diff[TARGET]).sum())          # 0 = top
    rates = (acts[train_rows] > 0).mean(0)
    W_dec = sae.W_dec.detach().float().cpu().numpy()
    t_dir = unit(W_dec[TARGET])
    char = {
        "feature": TARGET,
        "train_firing_rate": float(rates[TARGET]),
        "mean_act_known_tr": float(acts[k_tr, TARGET].mean()),
        "mean_act_unknown_tr": float(acts[unk_tr, TARGET].mean()),
        "train_diff": float(diff[TARGET]),
        "train_diff_rank_of_65536": diff_rank,
        "mean_act_val_known": float(acts[k_va, TARGET].mean()),
        "min_act_val_known": float(acts[k_va, TARGET].min()),
        "cos_with_d_know28": float(prev["d_know28"] @ t_dir),
        "cos_with_W_dec_1007": float(prev["sel_dir"] @ t_dir),
    }
    print(json.dumps(char, indent=2), flush=True)

    # --- FRESH null, MAGNITUDE-matched (protocol note: the rate-matched pool is
    # exhausted — only ~tens of always-on features exist under BatchTopK k=100 and
    # the previous null consumed them; nearest-by-rate now yields rates 0.17-0.25
    # vs target 1.0, a too-weak control. Mean activation magnitude is the
    # mechanically relevant matching stat for ablation impact (the removed
    # component scales with the feature's activation), so the fresh null matches
    # on TRAIN mean activation. Decided BEFORE any fresh-null measurement. -------
    mean_act = acts[train_rows].mean(0)
    excluded = set(prev["null_feats"].tolist()) | {int(prev["selected"]), TARGET}
    cand = np.argsort(np.abs(mean_act - mean_act[TARGET]), kind="stable")
    fresh = [int(f) for f in cand if int(f) not in excluded][:N_NULL]
    fresh_acts = [float(mean_act[f]) for f in fresh]
    fresh_rates = [float(rates[f]) for f in fresh]
    null_dirs = np.stack([unit(W_dec[f]) for f in fresh])
    print(f"  fresh null mean-act [{min(fresh_acts):.2f}, {max(fresh_acts):.2f}] "
          f"(target {mean_act[TARGET]:.2f}); rates [{min(fresh_rates):.3f}, {max(fresh_rates):.3f}]")

    np.savez(CAP / "step_d2_directions.npz",
             target=np.int64(TARGET), target_dir=t_dir, target_dir_raw=W_dec[TARGET].astype(np.float32),
             null_feats=np.array(fresh), null_dirs=null_dirs)
    RES.mkdir(exist_ok=True)
    (RES / "step_d2_selection.json").write_text(json.dumps({
        "sae_id": SAE_ID, "target_feature": TARGET,
        "why": "max drop inside STEP D's null (post-hoc) — tested here against a FRESH null",
        "characterization": char,
        "fresh_null_spec": f"N={N_NULL} nearest-by-TRAIN-mean-activation (magnitude-matched), "
                           "excluding target, 1007, and all 50 previous null features",
        "matching_note": "rate-matched pool exhausted (nearest available rates 0.17-0.25 vs "
                         "target 1.0); switched to magnitude matching BEFORE any fresh-null "
                         "measurement — magnitude is the mechanically relevant stat for "
                         "projection-ablation impact",
        "fresh_null_mean_act_range": [min(fresh_acts), max(fresh_acts)],
        "target_mean_act": float(mean_act[TARGET]),
        "fresh_null_rate_range": [min(fresh_rates), max(fresh_rates)],
        "preregistered_gate": "drop > 0 AND percentile >= 95 in fresh null; magnitude carries "
                              "winner's curse (labeled), specificity is the tested claim",
    }, indent=2), encoding="utf-8")
    print("  wrote step_d2_directions.npz + results/step_d2_selection.json")
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
