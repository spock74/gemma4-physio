"""O2 / GATE A — load the Gemma Scope 2 SAE, pin EXACT cfg values.

ISOLATED sae venv (calibration/.venv-sae — torch 2.7.1 + sae-lens 6.44.2).
Loads ONLY the SAE weights (no model), reads cfg, writes MANIFEST.lock.json.

Target (charter O2, 270m-it first):
    release = gemma-scope-2-270m-it-res
    sae_id  = layer_12_width_16k_l0_medium   (has a published Neuronpedia dashboard)

Gate: d_in == 640 (gemma-3-270m d_model). model_name in cfg should be the -it
checkpoint (the whole point of Gemma Scope 2 — SAEs on the instruction-tuned model).

Run: calibration/.venv-sae/bin/python calibration/gemmascope2_fidelity/gate_a_load_sae.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
from sae_lens import SAE

RELEASE = "gemma-scope-2-270m-it-res"
SAE_ID = "layer_12_width_16k_l0_medium"
EXPECT_D_IN = 640
EXPECT_LAYER = 12
HERE = Path(__file__).resolve().parent


def _cfg_to_dict(cfg: object) -> dict:
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(cfg, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        print(f"\n!! SAE.from_pretrained FAILED: {type(e).__name__}: {e}", flush=True)
        return 2

    sae = res[0] if isinstance(res, tuple) else res
    cfg_returned = res[1] if isinstance(res, tuple) and len(res) > 1 else None
    cfg_obj = getattr(sae, "cfg", None)
    cfg = _cfg_to_dict(cfg_obj) if cfg_obj is not None else {}
    if isinstance(cfg_returned, dict):
        cfg = {**cfg_returned, **cfg}
    meta = _cfg_to_dict(getattr(cfg_obj, "metadata", None)) if cfg_obj is not None else {}

    def pick(*names):
        for src in (cfg, meta):
            for n in names:
                if n in src and src[n] is not None:
                    return src[n]
        for n in names:
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
    prepend_bos = pick("prepend_bos")
    model_name = pick("model_name")
    hook_layer = pick("hook_layer", "layer")

    W_enc = getattr(sae, "W_enc", None)
    W_dec = getattr(sae, "W_dec", None)
    w_enc_shape = list(W_enc.shape) if isinstance(W_enc, torch.Tensor) else None
    w_dec_shape = list(W_dec.shape) if isinstance(W_dec, torch.Tensor) else None

    print("\n--- pinned cfg ---")
    for k, v in [("d_in", d_in), ("d_sae", d_sae), ("hook_name", hook_name),
                 ("hook_layer", hook_layer), ("dtype", dtype),
                 ("normalize_activations", normalize), ("apply_b_dec_to_input", apply_b_dec),
                 ("prepend_bos", prepend_bos), ("model_name", model_name),
                 ("W_enc.shape", w_enc_shape), ("W_dec.shape", w_dec_shape)]:
        print(f"  {k:24} = {v}")

    manifest = {
        "gate": "A", "objective": "O2",
        "sae_release": RELEASE, "sae_id": SAE_ID,
        "expect_d_in": EXPECT_D_IN, "expect_layer": EXPECT_LAYER,
        "cfg_pinned": {
            "d_in": d_in, "d_sae": d_sae, "hook_name": hook_name, "hook_layer": hook_layer,
            "dtype": str(dtype), "normalize_activations": _jsonable(normalize),
            "apply_b_dec_to_input": _jsonable(apply_b_dec), "prepend_bos": _jsonable(prepend_bos),
            "model_name": model_name,
        },
        "weight_shapes": {"W_enc": w_enc_shape, "W_dec": w_dec_shape},
        "cfg_full": _jsonable(cfg), "cfg_metadata_full": _jsonable(meta),
        "env": {"sae_lens": __import__("sae_lens").__version__, "torch": torch.__version__},
    }
    (HERE / "MANIFEST.lock.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {HERE / 'MANIFEST.lock.json'}")

    if d_in != EXPECT_D_IN:
        print(f"\n!! GATE A FAIL: d_in = {d_in} != {EXPECT_D_IN} — STOP.")
        return 1
    # Gemma Scope uses TransformerLens naming: blocks.N.hook_resid_post (resid-POST
    # of decoder layer N) — equivalently HF "...layers.N" forward-hook output.
    hn = str(hook_name or "")
    ok_hook = hn in (f"blocks.{EXPECT_LAYER}.hook_resid_post",) or hn.endswith(f"layers.{EXPECT_LAYER}")
    if not ok_hook:
        print(f"\n!! GATE A FAIL: hook_name {hook_name!r} is not layer-{EXPECT_LAYER} resid_post — STOP.")
        return 1
    if "resid_post" not in hn and "resid" not in hn:
        print(f"\n!! GATE A FAIL: hook_name {hook_name!r} is not a residual-stream hook — STOP.")
        return 1
    print(f"\nGATE A PASS: d_in == {d_in}, d_sae == {d_sae}, hook_name == {hook_name!r}, "
          f"model_name == {model_name!r}, prepend_bos == {prepend_bos}.")
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    sys.exit(main())
