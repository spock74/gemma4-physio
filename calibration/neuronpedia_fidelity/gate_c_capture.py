"""GATE B (inline) + GATE C stage 1 — CAPTURE (core conda env, torch 2.11.0).

Runs base google/gemma-4-E2B (the model the SAE was trained on, NOT -it) on the
EXACT token sequence of each frozen Neuronpedia reference snippet, and captures the
residual at layer 17 (forward-hook output of rec.layers[17] = resid-post L17 =
Neuronpedia hookName "...layers.17"). Saves residuals to captures/ for the sae-venv
encode stage. Neuronpedia protocol: raw text, prepend_bos=True (SAE cfg).

GATE B asserts (here, since the model is loaded): n_layers == 35, d_model == 1536.

Run (CORE env):  python calibration/neuronpedia_fidelity/gate_c_capture.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gemma4_lab.config import Settings
from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.interp.recorder import ActivationRecorder, tensor_health

BASE_MODEL = "google/gemma-4-E2B"  # SAE trained on BASE, not -it (GATE A finding)
LAYER = 17                          # Neuronpedia hosts only L17 of this SAE set
FEATURES = [0, 5, 19]
HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
CAP = HERE / "captures"


def main() -> int:
    import logfire

    from gemma4_lab import observability
    observability.setup()

    gemma = GemmaLocal(Settings())
    gemma.model_id = BASE_MODEL  # override -it -> base (calibration only; documented)
    rec = ActivationRecorder(gemma)
    tok = rec.tokenizer
    model = rec.model
    dev = gemma._device

    # --- GATE B: structure asserts (architecture is base/it-identical) ---
    assert rec.n_layers == 35, f"GATE B FAIL: n_layers {rec.n_layers} != 35"
    assert 0 <= LAYER < rec.n_layers
    print(f"GATE B: n_layers == 35 OK; capturing forward-output of rec.layers[{LAYER}] "
          f"(= resid-post L{LAYER} = Neuronpedia hookName ...layers.{LAYER}).", flush=True)

    CAP.mkdir(exist_ok=True)
    with logfire.span("calibration.gate_c.capture", model=BASE_MODEL, layer=LAYER):
        for idx in FEATURES:
            ref = json.loads((REF / f"np_feature_{idx}.json").read_text())
            snip = ref["activations"][0]               # the top-activating snippet
            tok_strs = snip["tokens"]
            np_values = np.asarray(snip["values"], dtype=np.float32)
            max_idx = int(snip["maxValueTokenIndex"])

            ids = tok.convert_tokens_to_ids(tok_strs)
            n_unk = sum(1 for i in ids if i is None or i == tok.unk_token_id)
            assert n_unk == 0, f"feature {idx}: {n_unk} UNK after token->id convert"
            input_ids = torch.tensor([[tok.bos_token_id, *ids]], device=dev)  # prepend_bos=True

            captured: dict[int, torch.Tensor] = {}

            def hook(_m, _i, output, _c=captured):
                h = output[0] if isinstance(output, tuple) else output
                _c["resid"] = h[0].detach().float().cpu()  # [seq, d_model]

            handle = rec.layers[LAYER].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    model(input_ids=input_ids,
                          attention_mask=torch.ones_like(input_ids))
            finally:
                handle.remove()

            resid = captured["resid"]                  # [1+ntok, d_model]
            assert resid.shape[-1] == 1536, f"GATE B FAIL: d_model {resid.shape[-1]} != 1536"
            bad = tensor_health(resid)
            if bad is not None:
                print(f"  [numerical-health] feature {idx}: non-finite resid {bad} — flagged")

            out = CAP / f"feature_{idx}.npz"
            np.savez(out,
                     resid=resid.numpy().astype(np.float32),  # row 0 = BOS, rows 1.. = content
                     np_values=np_values,                     # Neuronpedia per-token (content only)
                     max_idx=np.int64(max_idx),
                     feature_index=np.int64(idx),
                     n_content_tokens=np.int64(len(tok_strs)),
                     bos_prepended=np.int64(1))
            print(f"  feature {idx}: resid {tuple(resid.shape)} (BOS+{len(tok_strs)}), "
                  f"NP maxValue {snip['maxValue']:.3f} @ idx {max_idx} -> {out.name}", flush=True)

    print("\nGATE C capture done — residuals in captures/. Encode next in the sae venv:")
    print("  calibration/.venv-sae/bin/python calibration/neuronpedia_fidelity/gate_c_encode.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
