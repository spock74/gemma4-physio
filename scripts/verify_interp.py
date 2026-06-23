"""Verify the interp (Phase 6) infrastructure against the REAL model before trusting
any E1 number. Loads the configured Gemma 4 variant once and gates four assumptions:

  Step 2a  resolve_text_layers returns a non-empty ModuleList (print depth).
  Step 2b  model(**inputs).logits exists with shape [1, seq, vocab].
  Step 2c  the tokenizer's leading-space convention yields sensible single answer
           tokens (" Paris" -> one id), eyeballed for a few corpus answers.
  Step 3   the ablation AND steering hooks actually change next_token_logits under
           accelerate's MPS+CPU split (a no-op hook ordering would leave them
           identical — the silent failure this whole script exists to catch).

The Step-3 bite test uses a SEEDED ARBITRARY direction, not d_know. That is
deliberate: it proves the *mechanism* (the hook rewrites the forward pass) without
entangling it with the *science* (whether the model uses d_know). A null E1
necessity result is only a real finding once we know the hook itself bites.

Run:  python -m scripts.verify_interp   (or:  python scripts/verify_interp.py)
Exits non-zero if any gate fails.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from gemma4_lab import observability
from gemma4_lab.config import Settings
from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.interp.directions import ablating, steering
from gemma4_lab.interp.entity_knowledge import _answer_token_id
from gemma4_lab.interp.recorder import ActivationRecorder

PROMPT = "The capital of France is"
ANSWERS = ["Paris", "Tokyo", "Au", "Na", "0", "seven", "yen", "Everest"]

ok = True


def check(name: str, passed: bool, detail: str) -> None:
    global ok
    ok = ok and passed
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {name}: {detail}")


def main() -> int:
    observability.setup()
    settings = Settings()
    rec = ActivationRecorder(GemmaLocal(settings))

    print(f"\n{'=' * 64}\nVERIFY INTERP — {settings.model_id} on {settings.device}\n{'=' * 64}")
    print("Loading model (split across MPS+CPU; first load is slow)...")

    # --- Step 2a: decoder layers ------------------------------------------
    layers = rec.layers  # triggers load + resolve_text_layers
    check(
        "2a resolve_text_layers",
        isinstance(layers, nn.ModuleList) and len(layers) > 0,
        f"{type(layers).__name__} with len = {len(layers)}",
    )

    # --- Step 2b: logits shape --------------------------------------------
    inputs = rec.encode(PROMPT)
    seq = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = rec.model(**inputs)
    has_logits = hasattr(out, "logits") and out.logits is not None
    shape = tuple(out.logits.shape) if has_logits else None
    vocab = int(shape[-1]) if has_logits else -1  # vocab dim is just for display
    check(
        "2b model(**inputs).logits",
        has_logits and len(shape) == 3 and shape[0] == 1 and shape[1] == seq,
        f"shape = {shape} (expected [1, {seq}, vocab≈{vocab}])",
    )

    # --- Step 2c: answer-token leading-space convention -------------------
    tok = rec.tokenizer
    rows = []
    for ans in ANSWERS:
        ids = tok(" " + ans.strip(), add_special_tokens=False).input_ids
        first = _answer_token_id(tok, ans)
        rows.append(f"{ans!r:>12} -> id {first:>6}  decode={tok.decode([first])!r:>12}  ntok={len(ids)}")
    single = sum(1 for ans in ANSWERS if len(tok(' ' + ans, add_special_tokens=False).input_ids) == 1)
    print("  -- Step 2c answer tokenization (eyeball leading-space) --")
    for r in rows:
        print("       " + r)
    check("2c single-token answers", single >= len(ANSWERS) // 2, f"{single}/{len(ANSWERS)} answers are single-token")

    # --- Step 3: interventions must bite ----------------------------------
    d_model = int(rec.last_token_residuals(PROMPT, [0])[0].shape[0])  # also exercises capture hook
    torch.manual_seed(0)
    d_test = torch.randn(d_model, dtype=torch.float32)
    d_test = d_test / d_test.norm()

    clean = rec.next_token_logits(PROMPT)
    with ablating(layers, d_test):
        ablated = rec.next_token_logits(PROMPT)
    abl_diff = float((clean - ablated).abs().max())
    check("3 ablation bites", not torch.allclose(clean, ablated), f"max|Δlogit| = {abl_diff:.4f} (must be > 0)")

    with steering(layers, d_test, coeff=8.0):
        steered = rec.next_token_logits(PROMPT)
    steer_diff = float((clean - steered).abs().max())
    check("3 steering bites", not torch.allclose(clean, steered), f"max|Δlogit| = {steer_diff:.4f} (must be > 0)")

    # --- Step 4 (GATE 0): position-localized ablation ---------------------
    # Ablate the LAST position at layer 0 only; capture a downstream layer at ALL
    # positions. Causal attention => earlier positions never see the last, so they
    # must be bit-identical to clean; the last position must change. Global ablation
    # (contrast) changes earlier positions too.
    mid = len(layers) // 2
    clean_mid = rec.all_token_residuals(PROMPT, [mid])[mid]            # [seq, d]
    with ablating(layers[0:1], d_test, positions="last"):
        loc_mid = rec.all_token_residuals(PROMPT, [mid])[mid]
    with ablating(layers[0:1], d_test, positions=None):
        all_mid = rec.all_token_residuals(PROMPT, [mid])[mid]
    early_intact = torch.allclose(clean_mid[:-1], loc_mid[:-1], atol=1e-3)
    last_moved = not torch.allclose(clean_mid[-1], loc_mid[-1], atol=1e-3)
    early_moved_global = not torch.allclose(clean_mid[:-1], all_mid[:-1], atol=1e-3)
    check("4 localized 'last' leaves earlier positions intact",
          early_intact and last_moved,
          f"early Δmax={float((clean_mid[:-1]-loc_mid[:-1]).abs().max()):.4f} (~0), "
          f"last Δmax={float((clean_mid[-1]-loc_mid[-1]).abs().max()):.4f} (>0)")
    check("4 global ablation moves earlier positions (contrast)", early_moved_global,
          f"early Δmax={float((clean_mid[:-1]-all_mid[:-1]).abs().max()):.4f} (>0)")
    with ablating(layers[-1:], d_test, positions="last"):
        loc_lg = rec.next_token_logits(PROMPT)
    check("4 localized last-layer/last-pos ablation bites logits",
          not torch.allclose(clean, loc_lg), f"max|Δlogit| = {float((clean-loc_lg).abs().max()):.4f} (>0)")

    print("=" * 64)
    print("RESULT:", "ALL GATES PASS — safe to run E1c." if ok else "GATE FAILED — fix before E1c.")
    print("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
