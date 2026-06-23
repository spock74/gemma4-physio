"""Smoke tests for the interp (Phase 6 probing) track — no model load, no network.

Verifies the public surface imports, the E2 placeholder still raises, and the two
pure-Python statistics that E1's causal claims are reported with (rank AUC, Cohen's
d, diff-of-means) behave correctly. None of this touches MPS or the HF weights.
"""

from __future__ import annotations

import pytest
import torch


def test_public_surface_importable() -> None:
    """The names E1 depends on import cleanly from the package root."""
    from gemma4_lab import interp

    assert callable(interp.ActivationRecorder)
    assert callable(interp.diff_of_means_direction)
    assert callable(interp.ablating)
    assert callable(interp.steering)
    # Also exported for loader-path debugging.
    assert callable(interp.resolve_text_layers)


def test_linear_cka_identity_and_invariance() -> None:
    """CKA(X, X) == 1; CKA is dimension-agnostic and rotation/scale invariant."""
    from gemma4_lab.interp import linear_cka

    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 8, generator=g)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-4)
    # Invariant to an orthogonal rotation and to isotropic scaling.
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=g))
    assert linear_cka(x, 3.0 * (x @ q)) == pytest.approx(1.0, abs=1e-4)
    # Dimension-agnostic: comparing a 8-d and a 5-d representation must not error
    # and must score lower than self-similarity.
    y = torch.randn(32, 5, generator=g)
    cka_xy = linear_cka(x, y)
    assert 0.0 <= cka_xy <= 1.0 and cka_xy < 0.99


def test_logistic_probe_separates() -> None:
    """The torch GD probe recovers a linearly separable known/unknown split."""
    from gemma4_lab.interp import fit_logistic_probe
    from gemma4_lab.interp.matformer_elastic import _auc, probe_scores

    g = torch.Generator().manual_seed(1)
    pos = torch.randn(20, 6, generator=g) + 3.0
    neg = torch.randn(20, 6, generator=g) - 3.0
    feats = torch.cat([pos, neg])
    labels = torch.tensor([1] * 20 + [0] * 20)
    probe = fit_logistic_probe(feats, labels, steps=200)
    auc = _auc(probe_scores(feats, probe), labels)
    assert auc is not None and auc > 0.95


def test_heldout_auc_exposes_overfit() -> None:
    """The core E2 validity fix: with d >> n and NO real signal, an in-sample probe
    is trivially AUC~1.0 (overfit) while k-fold held-out AUC is ~chance."""
    from gemma4_lab.interp.matformer_elastic import (
        _auc,
        fit_logistic_probe,
        heldout_auc,
        probe_scores,
    )

    g = torch.Generator().manual_seed(3)
    feats = torch.randn(40, 500, generator=g)  # d >> n, label-independent noise
    labels = torch.tensor([1] * 20 + [0] * 20)
    insample = _auc(probe_scores(feats, fit_logistic_probe(feats, labels, steps=300)), labels)
    held = heldout_auc(feats, labels, "diffmeans", k=5)
    assert insample is not None and insample > 0.9  # overfits in-sample
    assert held is not None and held < 0.75          # held-out reveals no signal


def test_heldout_auc_recovers_real_signal() -> None:
    """When a real low-dim signal is embedded in high-dim noise, held-out AUC finds it."""
    from gemma4_lab.interp.matformer_elastic import heldout_auc

    g = torch.Generator().manual_seed(4)
    feats = torch.randn(40, 500, generator=g)
    feats[:20, :5] += 2.0  # class-1 shifted in the first 5 dims
    labels = torch.tensor([1] * 20 + [0] * 20)
    assert (heldout_auc(feats, labels, "diffmeans", k=5) or 0) > 0.8


def test_random_slice_cka_bounded() -> None:
    """The nesting random-slice baseline returns a CKA in [0, 1]."""
    from gemma4_lab.interp.matformer_elastic import _random_slice_cka

    g = torch.Generator().manual_seed(5)
    x2 = torch.randn(60, 8, generator=g)
    x4 = torch.randn(60, 16, generator=g)
    v = _random_slice_cka(x2, x4, d2=8, n_slices=4, seed=5)
    assert 0.0 <= v <= 1.0


def test_aligned_matrices_pairs_by_position() -> None:
    """Multi-position alignment truncates per prompt to the shorter sequence and
    concatenates, so n_repr = sum of per-prompt min lengths."""
    from gemma4_lab.interp.matformer_elastic import _aligned_matrices

    ap2 = [{0: torch.randn(5, 3)}, {0: torch.randn(4, 3)}]
    ap4 = [{0: torch.randn(5, 6)}, {0: torch.randn(6, 6)}]  # 2nd prompt longer in E4B
    x2, x4 = _aligned_matrices(ap2, ap4, 0, 0)
    assert x2.shape == (5 + 4, 3) and x4.shape == (5 + 4, 6)  # min(5,5)+min(4,6)=9


def test_match_layers_endpoints() -> None:
    """Depth-fraction alignment keeps endpoints and is monotonic."""
    from gemma4_lab.interp import match_layers

    pairs = match_layers(35, 42)
    assert len(pairs) == 35
    assert pairs[0] == (0, 0) and pairs[-1] == (34, 41)
    larges = [b for _, b in pairs]
    assert larges == sorted(larges)  # monotonic non-decreasing


def test_matformer_run_is_callable() -> None:
    """E2 run() exists and is callable (no model load here — it would download/load)."""
    from gemma4_lab.interp import matformer_elastic

    assert callable(matformer_elastic.run)


def test_refusal_matcher_and_corpus() -> None:
    """is_refusal catches canonical refusals (head-of-text only) and passes
    compliance; the contrast corpus is index-paired."""
    import json
    from pathlib import Path

    from gemma4_lab.interp.refusal_control import REFUSAL_MARKERS, is_refusal

    assert is_refusal("I cannot help with that request.")
    assert is_refusal("I'm sorry, but I can't assist with this.")
    assert is_refusal("As an AI, I must decline.")
    assert not is_refusal("Sure! Here's a draft of the thank-you note you asked for.")
    assert not is_refusal("Here is a polite letter to your landlord about the repair.")
    # marker must be in the HEAD, not anywhere
    assert not is_refusal("Here's the letter. " + "x" * 100 + " i cannot stress enough...")
    assert all(m == m.lower() for m in REFUSAL_MARKERS)  # matcher lowercases first

    corpus = json.loads(Path("data/eval/refusal_contrast.json").read_text())
    assert len(corpus["harmful"]) == len(corpus["harmless"]) >= 8  # index-paired


def test_tensor_health_detects_nonfinite() -> None:
    """tensor_health: None when clean; counts + fraction when NaN/inf present."""
    from gemma4_lab.interp.recorder import tensor_health

    assert tensor_health(torch.ones(3, 4)) is None
    t = torch.ones(2, 5)
    t[0, 0] = float("nan")
    t[1, 4] = float("inf")
    h = tensor_health(t)
    assert h is not None and h["n_nonfinite"] == 2 and h["n_elements"] == 10
    assert h["frac"] == pytest.approx(0.2)


def test_math_guards_fail_loud_on_nonfinite() -> None:
    """Repo convention: pure-math helpers RAISE on non-finite input — NaN must never
    silently become a plausible number."""
    from gemma4_lab.interp import diff_of_means_direction, projection, rank_auc
    from gemma4_lab.interp.directions import require_finite
    from gemma4_lab.interp.matformer_elastic import fit_logistic_probe, heldout_auc, linear_cka

    nan_vec = torch.tensor([1.0, float("nan")])
    ok_vec = torch.ones(2)

    with pytest.raises(ValueError, match="non-finite"):
        require_finite("t", nan_vec)
    with pytest.raises(ValueError, match="non-finite"):
        diff_of_means_direction([nan_vec], [ok_vec])
    with pytest.raises(ValueError, match="non-finite"):
        projection(nan_vec, ok_vec)
    with pytest.raises(ValueError, match="non-finite"):
        rank_auc([1.0, float("nan")], ["yes", "no"])
    inf_mat = torch.ones(4, 3)
    inf_mat[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        linear_cka(inf_mat, torch.ones(4, 3))
    with pytest.raises(ValueError, match="non-finite"):
        fit_logistic_probe(inf_mat, torch.tensor([1.0, 0, 1, 0]))
    with pytest.raises(ValueError, match="non-finite"):
        heldout_auc(inf_mat, torch.tensor([1, 0, 1, 0]))
    # clean inputs still work
    assert projection(ok_vec, ok_vec) == pytest.approx(2.0)


def test_winsorize_residuals_explicit_path() -> None:
    """The documented ALTERNATIVE recovery (off by default): zero non-finite,
    clamp magnitudes; identity on clean input."""
    from gemma4_lab.interp.numerical_health import winsorize_residuals

    clean = torch.randn(4, 8)
    assert torch.equal(winsorize_residuals(clean), clean)  # untouched when clean
    t = torch.ones(2, 4)
    t[0, 0] = float("nan")
    t[1, 1] = float("inf")
    w = winsorize_residuals(t)
    assert torch.isfinite(w).all()
    assert w[0, 0] == 0.0  # non-finite replaced, not interpolated


def test_position_localized_ablation() -> None:
    """GATE 0: ablating(..., positions='last') must change ONLY the final position;
    positions=None changes all. Model-free — drives the hook on a tiny echo layer."""
    import torch.nn as nn

    from gemma4_lab.interp import ablating

    class EchoTensor(nn.Module):
        def forward(self, x):  # non-tuple output path
            return x

    class EchoTuple(nn.Module):
        def forward(self, x):  # tuple output path (like a decoder layer)
            return (x,)

    d = torch.zeros(8)
    d[0] = 1.0  # unit direction along dim 0
    h = torch.randn(1, 4, 8)

    # --- tensor-output layer, positions="last" ---
    layer = EchoTensor()
    with ablating(nn.ModuleList([layer]), d, positions="last"):
        out = layer(h)
    assert torch.allclose(out[:, :3, :], h[:, :3, :])          # earlier positions untouched
    assert out[0, 3, 0].abs() < 1e-5                            # component along d removed at last
    assert not torch.allclose(out[:, 3, :], h[:, 3, :])         # last position changed

    # --- positions=None changes EVERY position ---
    with ablating(nn.ModuleList([layer]), d, positions=None):
        out_all = layer(h)
    assert all(out_all[0, t, 0].abs() < 1e-5 for t in range(4))

    # --- explicit index list, tuple-output layer ---
    lt = EchoTuple()
    with ablating(nn.ModuleList([lt]), d, positions=[1]):
        out_t = lt(h)[0]
    assert out_t[0, 1, 0].abs() < 1e-5                          # index 1 ablated
    assert torch.allclose(out_t[:, 0, :], h[:, 0, :])           # index 0 untouched
    assert torch.allclose(out_t[:, 2:, :], h[:, 2:, :])         # 2,3 untouched


def test_diff_of_means_direction_is_unit_and_oriented() -> None:
    """mean(pos) - mean(neg), unit-normalized and pointing pos-ward."""
    from gemma4_lab.interp import diff_of_means_direction

    pos = [torch.tensor([2.0, 0.0]), torch.tensor([4.0, 0.0])]  # mean (3, 0)
    neg = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 3.0])]  # mean (0, 2)
    d = diff_of_means_direction(pos, neg)

    assert d.shape == (2,)
    assert d.norm().item() == pytest.approx(1.0, abs=1e-5)
    # Direction is (3, -2) normalized → positive x, negative y.
    assert d[0] > 0 and d[1] < 0


def test_rank_auc_perfect_and_chance() -> None:
    """AUC = 1.0 when positives strictly outrank negatives, 0.5 at chance."""
    from gemma4_lab.interp import rank_auc

    perfect = rank_auc([3.0, 2.0, 1.5], ["yes", "yes", "no"])
    assert perfect == pytest.approx(1.0)

    # Interleaved/tied → chance. One yes==one no (tie counts 0.5).
    chance = rank_auc([1.0, 1.0], ["yes", "no"])
    assert chance == pytest.approx(0.5)

    # Degenerate (single class) → None, not a crash.
    assert rank_auc([1.0, 2.0], ["yes", "yes"]) is None


def test_cohens_d_sign_and_zero_variance() -> None:
    """Positive separation gives positive d; zero pooled variance returns None."""
    from gemma4_lab.interp import cohens_d

    d = cohens_d([10.0, 11.0, 12.0], [0.0, 1.0, 2.0])
    assert d is not None and d > 0

    assert cohens_d([5.0, 5.0], [5.0, 5.0]) is None  # no variance
    assert cohens_d([1.0], [0.0]) is None  # too few samples
