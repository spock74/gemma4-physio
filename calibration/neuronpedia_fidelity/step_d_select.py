"""STEP D stage 2 — FEATURE SELECTION, held-out (ISOLATED sae venv, torch 2.7.1).

Loads the layer-28 SAE, encodes the captured base-E2B residuals, and selects the
"entity/factual-recall" feature on the TRAIN split ONLY (known_tr vs unknown_tr mean
activation diff) — the ablation is evaluated later on the disjoint VAL knowns
(selection/evaluation split, no circularity).

Null spec (agreed): N=50 OTHER SAE features, frequency-matched to the selected
feature's TRAIN firing rate (apples-to-apples for feature ablation — another
feature, not a dense Gaussian direction). Exports W_dec unit rows for the selected
+ null features, plus d_know@28 (diff-of-means on TRAIN residuals, base model) and
its cosine to the selected feature's decoder direction (the optional bonus).

Run (SAE VENV):  calibration/.venv-sae/bin/python calibration/neuronpedia_fidelity/step_d_select.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE

RELEASE = "decoderesearch/gemma-4-saes"
SAE_ID = "gemma-4-e2b/btk-mat-layer-28-k-100"
N_NULL = 50
HERE = Path(__file__).resolve().parent
CAP = HERE / "captures"
RES = HERE / "results"


def main() -> int:
    d = np.load(CAP / "step_d_data.npz")
    resids = torch.tensor(d["resids"], dtype=torch.float32)        # [80, 1536]
    labels = d["labels"]
    k_tr, k_va = d["k_tr"].tolist(), d["k_va"].tolist()
    u_tr = d["u_tr"].tolist()
    n_known = int((labels == 1).sum())
    unk_tr_global = [n_known + i for i in u_tr]                     # unknown idxs are offset

    print(f"Loading SAE {SAE_ID} ...", flush=True)
    res = SAE.from_pretrained(RELEASE, SAE_ID)
    sae = (res[0] if isinstance(res, tuple) else res).to("cpu").to(torch.float32)
    assert int(sae.cfg.d_in) == 1536

    with torch.no_grad():
        acts = sae.encode(resids)                                   # [80, d_sae]
    acts = (acts[0] if isinstance(acts, tuple) else acts).numpy()
    d_sae = acts.shape[1]

    # --- TRAIN-only selection: mean act diff (known_tr - unknown_tr) ---------
    diff = acts[k_tr].mean(0) - acts[unk_tr_global].mean(0)         # [d_sae]
    order = np.argsort(-diff)
    top5 = [{"feature": int(f), "train_diff": float(diff[f]),
             "mean_act_known_tr": float(acts[k_tr, f].mean()),
             "mean_act_unknown_tr": float(acts[unk_tr_global, f].mean())}
            for f in order[:5]]
    selected = int(order[0])
    print("  top-5 by train diff:", json.dumps(top5, indent=2), flush=True)

    # --- frequency-matched null: 50 nearest TRAIN firing rates ---------------
    train_rows = k_tr + unk_tr_global
    rates = (acts[train_rows] > 0).mean(0)                          # [d_sae]
    sel_rate = float(rates[selected])
    cand = np.argsort(np.abs(rates - sel_rate), kind="stable")
    null_feats = [int(f) for f in cand if f != selected][:N_NULL]
    null_rates = [float(rates[f]) for f in null_feats]

    # --- export unit decoder directions --------------------------------------
    W_dec = sae.W_dec.detach().float().cpu().numpy()                # [d_sae, 1536]
    def unit(v: np.ndarray) -> np.ndarray:
        return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    sel_dir = unit(W_dec[selected])
    null_dirs = np.stack([unit(W_dec[f]) for f in null_feats])      # [50, 1536]

    # --- bonus: d_know@28 on BASE (train diff-of-means) and cosine ----------
    d_know28 = resids[k_tr].mean(0).numpy() - resids[unk_tr_global].mean(0).numpy()
    d_know28 = unit(d_know28)
    cos_dknow = float(d_know28 @ sel_dir)

    out = CAP / "step_d_directions.npz"
    np.savez(out, selected=np.int64(selected), sel_dir=sel_dir,
             null_feats=np.array(null_feats), null_dirs=null_dirs,
             d_know28=d_know28, cos_dknow_seldir=np.float32(cos_dknow))
    RES.mkdir(exist_ok=True)
    (RES / "step_d_selection.json").write_text(json.dumps({
        "sae_id": SAE_ID, "d_sae": int(d_sae),
        "selection_split": "TRAIN only (seed 0, same as E1) — evaluation on disjoint VAL",
        "selected_feature": selected, "selected_train_rate": sel_rate,
        "top5_by_train_diff": top5,
        "null_spec": f"N={N_NULL} SAE features nearest in TRAIN firing rate (apples-to-apples)",
        "null_rate_range": [min(null_rates), max(null_rates)],
        "cos_dknow28_base_vs_selected_W_dec": round(cos_dknow, 4),
        "selected_mean_act_val_known": float(acts[k_va, selected].mean()),
    }, indent=2), encoding="utf-8")
    print(f"  selected feature {selected} (train rate {sel_rate:.3f}); "
          f"null rates [{min(null_rates):.3f}, {max(null_rates):.3f}]")
    print(f"  cos(d_know@28_base, W_dec[{selected}]) = {cos_dknow:.4f}")
    print(f"  mean act of feature {selected} on VAL knowns: {acts[k_va, selected].mean():.3f}")
    print(f"  wrote {out} + results/step_d_selection.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
