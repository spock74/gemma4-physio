"""STEP D stage 1 — CAPTURE (core env, torch 2.11.0): base E2B, raw clozes.

Method-level positive control at the LOGIT level, on-distribution for the SAE:
the layer-28 SAE was trained on base google/gemma-4-E2B, and on the BASE model the
raw-cloze readout is valid (base models complete clozes; the -it echo problem does
not apply). One forward per prompt captures BOTH the resid-post L28 last-token
residual (SAE input) and the next-token logits (gold readout).

Guard-rails (agreed 2026-06-11): base model only — the result is a METHOD/APPARATUS
positive, never a causal replication of d_know(-it); recall-precondition gate
(median gold rank) before any ablation is interpreted; numerical health on every
capture; Logfire span.

Outputs captures/step_d_data.npz for the sae-venv selection stage.

Run (CORE env):  python calibration/neuronpedia_fidelity/step_d_capture.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from gemma4_lab.config import Settings
from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.interp.entity_knowledge import _answer_token_id, _split_indices
from gemma4_lab.interp.recorder import ActivationRecorder, tensor_health

BASE_MODEL = "google/gemma-4-E2B"
LAYER = 28
HERE = Path(__file__).resolve().parent
CAP = HERE / "captures"


def main() -> int:
    import logfire

    from gemma4_lab import observability
    observability.setup()

    settings = Settings()
    corpus = json.loads((settings.data_dir / "eval" / "entity_knowledge_contrast.json").read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    prompts = [it["prompt"] for it in known] + [it["prompt"] for it in unknown]
    labels = np.array([1] * len(known) + [0] * len(unknown), dtype=np.int64)

    gemma = GemmaLocal(settings)
    gemma.model_id = BASE_MODEL  # base, not -it (SAE on-distribution; raw cloze valid)
    rec = ActivationRecorder(gemma)
    model = rec.model
    tok = rec.tokenizer
    assert rec.n_layers == 35

    gold_ids = np.array([_answer_token_id(tok, it["answer"]) for it in known], dtype=np.int64)

    resids = np.zeros((len(prompts), 1536), dtype=np.float32)
    gold_logits = np.full(len(known), np.nan, dtype=np.float32)
    gold_ranks = np.full(len(known), -1, dtype=np.int64)

    with logfire.span("calibration.step_d.capture", model=BASE_MODEL, layer=LAYER,
                      n_prompts=len(prompts)):
        for i, p in enumerate(prompts):
            captured: dict[str, torch.Tensor] = {}

            def hook(_m, _i, output, _c=captured):
                h = output[0] if isinstance(output, tuple) else output
                _c["h"] = h[0, -1, :].detach().float().cpu()   # last position, [1536]

            handle = rec.layers[LAYER].register_forward_hook(hook)
            try:
                inputs = rec.encode(p, templated=False)        # raw cloze; tokenizer adds BOS
                with torch.no_grad():
                    out = model(**inputs)
            finally:
                handle.remove()

            h = captured["h"]
            bad = tensor_health(h)
            assert bad is None, f"prompt {i}: non-finite L{LAYER} residual {bad}"
            resids[i] = h.numpy()

            if i < len(known):                                  # gold readout on knowns
                logits = out.logits[0, -1, :].detach().float().cpu()
                assert torch.isfinite(logits).all(), f"prompt {i}: non-finite logits"
                gid = int(gold_ids[i])
                gold_logits[i] = float(logits[gid])
                gold_ranks[i] = int((logits > logits[gid]).sum())
            if i % 20 == 0:
                print(f"  [capture] prompt {i}/{len(prompts)}", flush=True)

    k_tr, k_va = _split_indices(len(known), 0.5, 0)             # E1's exact split (seed 0)
    u_tr, u_va = _split_indices(len(unknown), 0.5, 1)

    med_rank = median(int(r) for r in gold_ranks)
    top5 = float((gold_ranks < 5).mean())
    print(f"\n  RECALL PRECONDITION (base E2B, raw cloze): median gold rank = {med_rank}, "
          f"top-5 = {top5:.0%}  ({'OK' if med_rank <= 10 else 'POOR — interpret drops with caution'})")

    CAP.mkdir(exist_ok=True)
    out_path = CAP / "step_d_data.npz"
    np.savez(out_path, resids=resids, labels=labels, gold_ids=gold_ids,
             gold_logits=gold_logits, gold_ranks=gold_ranks,
             k_tr=np.array(k_tr), k_va=np.array(k_va),
             u_tr=np.array(u_tr), u_va=np.array(u_va),
             median_gold_rank=np.int64(med_rank), layer=np.int64(LAYER))
    print(f"  wrote {out_path} (resids {resids.shape}, splits seed 0)")
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    raise SystemExit(main())
