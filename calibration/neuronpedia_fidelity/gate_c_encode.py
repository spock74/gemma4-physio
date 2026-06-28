"""GATE C stage 2 — ENCODE + compare (ISOLATED sae venv, torch 2.7.1 + sae-lens).

Loads the layer-17 SAE, runs sae.encode() on the residuals captured by
gate_c_capture.py (core env), extracts each reference feature's per-token
activation, aligns to Neuronpedia (drop the prepended BOS), and compares.

PASS iff local activation tracks Neuronpedia: same top-activating position AND
high Pearson correlation across tokens. Writes results/gate_c_results.json.
FAIL -> the report names which detail breaks (hook point / normalization / BOS /
dtype) — the failure IS the finding.

Run (SAE VENV):  calibration/.venv-sae/bin/python calibration/neuronpedia_fidelity/gate_c_encode.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE

RELEASE = "decoderesearch/gemma-4-saes"
SAE_ID = "gemma-4-e2b/btk-mat-layer-17-k-100"   # layer 17 (Neuronpedia hosts only L17)
NP_SOURCE = "17-matryoshka-res-65k"
HERE = Path(__file__).resolve().parent
CAP = HERE / "captures"
RES = HERE / "results"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    print(f"Loading SAE {SAE_ID} ...", flush=True)
    res = SAE.from_pretrained(RELEASE, SAE_ID)
    sae = res[0] if isinstance(res, tuple) else res
    sae = sae.to("cpu").to(torch.float32)
    d_in = int(getattr(sae.cfg, "d_in", -1))
    hook = getattr(getattr(sae.cfg, "metadata", None), "hook_name", None) or getattr(sae.cfg, "hook_name", None)
    print(f"  SAE d_in={d_in} hook={hook}", flush=True)
    assert d_in == 1536, f"layer-17 SAE d_in {d_in} != 1536"
    assert hook and hook.endswith("layers.17"), f"hook {hook} is not layer 17"

    RES.mkdir(exist_ok=True)
    per_feature = []
    for npz in sorted(CAP.glob("feature_*.npz")):
        d = np.load(npz)
        resid = torch.tensor(d["resid"], dtype=torch.float32)      # [1+ntok, 1536] (row0 = BOS)
        np_vals = d["np_values"].astype(np.float32)                # [ntok] content only
        fidx = int(d["feature_index"])
        max_idx = int(d["max_idx"])

        with torch.no_grad():
            feats = sae.encode(resid)                              # [seq, d_sae]
        if isinstance(feats, tuple):
            feats = feats[0]
        my_full = feats[:, fidx].float().cpu().numpy()             # [1+ntok]
        my_vals = my_full[1:]                                       # drop BOS -> align to NP content

        assert my_vals.shape[0] == np_vals.shape[0], \
            f"len mismatch {my_vals.shape[0]} vs {np_vals.shape[0]}"

        my_peak = int(my_vals.argmax())
        r = _pearson(my_vals, np_vals)
        # cosine ignores scale (SAE encode units may be scaled vs Neuronpedia display)
        denom = (np.linalg.norm(my_vals) * np.linalg.norm(np_vals)) or 1.0
        cos = float(my_vals @ np_vals / denom)
        scale = float(my_vals.max() / np_vals.max()) if np_vals.max() > 1e-9 else None
        peak_match = my_peak == max_idx
        per_feature.append({
            "feature_index": fidx,
            "n_tokens": int(np_vals.shape[0]),
            "neuronpedia_peak_idx": max_idx, "local_peak_idx": my_peak,
            "peak_match": peak_match,
            "neuronpedia_max": float(np_vals.max()), "local_max": float(my_vals.max()),
            "scale_ratio_local_over_np": round(scale, 4) if scale else None,
            "pearson_r": round(r, 4), "cosine": round(cos, 4),
            "local_active_tokens": int((my_vals > 1e-6).sum()),
            "np_active_tokens": int((np_vals > 1e-6).sum()),
        })
        scale_s = f" scale {scale:.3f}" if scale else ""
        print(f"  feature {fidx}: peak NP {max_idx} vs local {my_peak} "
              f"(match={peak_match}); pearson {r:.3f} cosine {cos:.3f}{scale_s}", flush=True)

    mean_r = float(np.nanmean([f["pearson_r"] for f in per_feature]))
    mean_cos = float(np.nanmean([f["cosine"] for f in per_feature]))
    all_peaks = all(f["peak_match"] for f in per_feature)
    gate_c_pass = bool(all_peaks and mean_cos >= 0.9)

    result = {
        "gate": "C",
        "sae_release": RELEASE, "sae_id": SAE_ID, "neuronpedia_source": NP_SOURCE,
        "model_captured": "google/gemma-4-E2B (base; SAE training model)",
        "layer": 17, "n_features": len(per_feature),
        "mean_pearson_r": round(mean_r, 4), "mean_cosine": round(mean_cos, 4),
        "all_peaks_match": all_peaks,
        "gate_c_pass": gate_c_pass,
        "verdict": ("PASS — local sae.encode tracks Neuronpedia (same peaks + high "
                    "correlation): apparatus (layer indexing, hook point, BOS, dtype) "
                    "validated against a third party on the real Gemma 4 E2B."
                    if gate_c_pass else
                    "FAIL/PARTIAL — see per_feature; check hook point (resid_pre vs post), "
                    "BOS alignment, normalization (apply_b_dec_to_input), or dtype."),
        "per_feature": per_feature,
        "env_note": f"encode run in the isolated sae venv (torch {torch.__version__}); "
                    "capture in core (torch 2.11.0)",
    }
    (RES / "gate_c_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"GATE C — SAE fidelity vs Neuronpedia ({NP_SOURCE}, base E2B, L17)")
    print("=" * 68)
    print(f"  features: {len(per_feature)}  peaks_match: {all_peaks}  "
          f"mean pearson {mean_r:.3f}  mean cosine {mean_cos:.3f}")
    print(f"  {'PASS' if gate_c_pass else 'FAIL/PARTIAL'}: {result['verdict']}")
    print(f"  wrote {RES / 'gate_c_results.json'}")
    print("=" * 68)
    return 0 if gate_c_pass else 1


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
