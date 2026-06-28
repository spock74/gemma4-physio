"""GATE A — instrument: load the third-party SAE, read its cfg, pin EXACT values.

Runs in the ISOLATED sae venv (calibration/.venv-sae — torch 2.7.1 + sae-lens),
NOT the core conda env (that stays torch 2.11.0; see ../../pyproject.toml note).

We load ONLY the SAE weights via SAE.from_pretrained and read cfg. No
HookedSAETransformer, no model here. Writes MANIFEST.lock.json (the single source
of pinned numbers). d_in != 1536 -> exit 1 (STOP).

Run:  calibration/.venv-sae/bin/python calibration/neuronpedia_fidelity/gate_a_load_sae.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
from sae_lens import SAE

RELEASE = "decoderesearch/gemma-4-saes"
SAE_ID = "gemma-4-e2b/btk-mat-layer-28-k-100"
MODEL_ID = "google/gemma-4-E2B-it"
HERE = Path(__file__).resolve().parent


def _cfg_to_dict(cfg: object) -> dict:
    """Best-effort flatten of the sae-lens cfg object (dataclass or pydantic or plain)."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(cfg, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:  # noqa: BLE001 - introspection only, fall through
                pass
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {"_repr": repr(cfg)}


def _jsonable(v: object) -> object:
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def main() -> int:
    print(f"Loading SAE: release={RELEASE!r} sae_id={SAE_ID!r}", flush=True)
    try:
        res = SAE.from_pretrained(RELEASE, SAE_ID)
    except Exception as e:  # noqa: BLE001 - want the available ids on any failure
        print(f"\n!! SAE.from_pretrained FAILED: {type(e).__name__}: {e}", flush=True)
        try:
            from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
            d = get_pretrained_saes_directory()
            entry = d.get(RELEASE)
            ids = list(getattr(entry, "saes_map", {}) or {})
            print(f"  available sae_ids for {RELEASE} ({len(ids)}): {ids[:40]}")
        except Exception as e2:  # noqa: BLE001
            print(f"  (could not list sae_ids: {e2})")
        return 2

    # from_pretrained may return sae or (sae, cfg_dict, sparsity) across versions.
    cfg_returned = None
    if isinstance(res, tuple):
        sae = res[0]
        if len(res) > 1:
            cfg_returned = res[1]
    else:
        sae = res

    cfg_obj = getattr(sae, "cfg", None)
    cfg = _cfg_to_dict(cfg_obj) if cfg_obj is not None else {}
    if isinstance(cfg_returned, dict):
        cfg = {**cfg_returned, **cfg}  # prefer the live object's values
    meta = _cfg_to_dict(getattr(cfg_obj, "metadata", None)) if cfg_obj is not None else {}

    def pick(*names, src=None):
        for src_dict in ([src] if src else [cfg, meta]):
            for n in names:
                if n in src_dict and src_dict[n] is not None:
                    return src_dict[n]
        for n in names:  # fall back to attribute access on the cfg object
            v = getattr(cfg_obj, n, None)
            if v is not None:
                return v
        return None

    d_in = pick("d_in")
    d_sae = pick("d_sae")
    hook_name = pick("hook_name")
    dtype = pick("dtype")
    normalize = pick("normalize_activations")
    apply_b_dec = pick("apply_b_dec_to_input")
    model_name = pick("model_name")
    hook_layer = pick("hook_layer", "layer")

    # weight shapes (ground truth even if cfg is messy)
    W_enc = getattr(sae, "W_enc", None)
    W_dec = getattr(sae, "W_dec", None)
    w_enc_shape = list(W_enc.shape) if isinstance(W_enc, torch.Tensor) else None
    w_dec_shape = list(W_dec.shape) if isinstance(W_dec, torch.Tensor) else None

    print("\n--- pinned cfg ---")
    for k, v in [("d_in", d_in), ("d_sae", d_sae), ("hook_name", hook_name),
                 ("dtype", dtype), ("normalize_activations", normalize),
                 ("apply_b_dec_to_input", apply_b_dec), ("model_name", model_name),
                 ("hook_layer", hook_layer), ("W_enc.shape", w_enc_shape),
                 ("W_dec.shape", w_dec_shape)]:
        print(f"  {k:24} = {v}")

    # model revision SHA (pin the exact checkpoint the apparatus runs against)
    from huggingface_hub import HfApi
    sha = HfApi().model_info(MODEL_ID, token=os.getenv("HF_TOKEN") or None).sha

    manifest = {
        "gate": "A",
        "sae_release": RELEASE,
        "sae_id": SAE_ID,
        "model_id": MODEL_ID,
        "model_revision_sha": sha,
        "sae_repo_note": "decoderesearch/gemma-4-saes — public; E2B SAEs at layers 6/17/28",
        "cfg_pinned": {
            "d_in": d_in, "d_sae": d_sae, "hook_name": hook_name,
            "dtype": str(dtype), "normalize_activations": _jsonable(normalize),
            "apply_b_dec_to_input": _jsonable(apply_b_dec), "model_name": model_name,
            "hook_layer": hook_layer,
        },
        "weight_shapes": {"W_enc": w_enc_shape, "W_dec": w_dec_shape},
        "cfg_full": _jsonable(cfg),
        "cfg_metadata_full": _jsonable(meta),
        "env": {"sae_lens": __import__("sae_lens").__version__, "torch": torch.__version__,
                "note": "this MANIFEST written from the ISOLATED sae venv (torch 2.7.1); "
                        "capture runs in the core env (torch 2.11.0)"},
    }
    out = HERE / "MANIFEST.lock.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if d_in != 1536:
        print(f"\n!! GATE A FAIL: d_in = {d_in} != 1536 — STOP (wrong SAE / wrong model dim).")
        return 1
    print(f"\nGATE A PASS: d_in == 1536, d_sae == {d_sae}, hook_name == {hook_name!r}, "
          f"model sha {sha[:12]}.")
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    sys.exit(main())
