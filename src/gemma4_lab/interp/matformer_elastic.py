"""E2 — Elastic interpretability across MatFormer granularities.

Anchor: Devvrit et al. 2023, MatFormer (arXiv:2310.07707). Gemma 4 ships E2B and
E4B as MatFormer granularities of one family. The original E2 hypothesis assumed
E2B is "nested in E4B sharing the residual stream", so a direction fit on E4B would
transfer to E2B at matched layers.

** That premise is falsified by the shipped configs ** (verified 2026-06-09):

    E2B-it : hidden_size 1536, 35 layers, 1 KV head
    E4B-it : hidden_size 2560, 42 layers, 2 KV heads

Canonical MatFormer (Gemma 3n) nests ONLY the FFN intermediate dimension, keeping
d_model and depth fixed — there, residual directions transfer by identity. Gemma 4
ALSO compresses d_model (1536 vs 2560) and depth (35 vs 42), so the two residual
streams live in DIFFERENT vector spaces. transformers 5.7 exposes no MatFormer
slicing API, so only the two shipped checkpoints exist. Literal "fit d_know on E4B
(2560-d), apply to E2B (1536-d)" is dimensionally ill-posed.

Reframed (sharper) question, scoped to the two shipped checkpoints: despite the
d_model + depth compression, is the known/unknown entity geometry still ALIGNED
across granularities?

Statistical validity (n>>d is the binding constraint — 80 prompts over 1536–2560
dims is n<<d, which inflates CKA and makes a same-data probe trivially AUC~1.0):
  (1) **Multi-position CKA**: capture EVERY token position (not just the last);
      E2B/E4B share the tokenizer so rows align by (prompt, position). This lifts
      n_repr to >1000 so linear CKA is in a usable regime. (Caveat: includes the
      shared chat-template positions, which can inflate similarity — reported.)
  (2) **Nesting test with a random baseline**: CKA of E2B vs the FIRST d2 dims of
      E4B (the Matryoshka inner-slice hypothesis) vs the TAIL d2 dims AND vs the
      mean over several RANDOM d2-wide slices. Nesting requires head > random, not
      merely head > tail.
  (3) **Held-out decodability**: k-fold AUC, never in-sample. Primary metric is the
      1-D diff-of-means projection (generalizes under d>>n); a strongly-L2 torch
      logistic probe is reported as secondary. Direct cross-model probe transfer is
      N/A (dims differ) and reported as such, not silently skipped.

All capture is wrapped in Logfire spans (observability first). One model is loaded
at a time (~10–16 GB each); the parent is freed before the child loads.

Numerical-health convention (repo-wide, see interp/numerical_health.py): the pure-
math helpers here (linear_cka, fit_logistic_probe, probe_scores, heldout_auc)
REQUIRE finite inputs and raise ValueError otherwise; run() excludes contaminated
layer pairs explicitly (reported as N/A-with-reason, never computed as if valid).
"""

from __future__ import annotations

import datetime
import gc
import json
from pathlib import Path

import torch

from ..config import Settings
from .directions import diff_of_means_direction, rank_auc, require_finite
from .entity_knowledge import RECALL_INSTRUCTION
from .recorder import ActivationRecorder, contaminated_layers_from_residuals

# -- representational similarity --------------------------------------------

def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019) between activation matrices
    X[n, d1] and Y[n, d2]. Columns are centered; the linear kernel makes this
    dimension-agnostic, so E2B (1536-d) and E4B (2560-d) residuals compare
    directly. Returns similarity in [0, 1] (1 = identical up to rotation/scale).
    NOTE: at n << d CKA is upward-biased — report n_repr alongside it.
    Raises on non-finite input (see module docstring)."""
    require_finite("linear_cka", x, y)
    x = _center(x.float())
    y = _center(y.float())
    # CKA = ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    xty_f2 = (x.t() @ y).pow(2).sum()
    xtx_f = (x.t() @ x).norm()
    yty_f = (y.t() @ y).norm()
    denom = xtx_f * yty_f
    if denom <= 0:
        return 0.0
    return float(xty_f2 / denom)


def _random_slice_cka(x2: torch.Tensor, x4: torch.Tensor, d2: int, n_slices: int, seed: int) -> float:
    """Mean CKA of X2 against several RANDOM d2-wide column subsets of X4 — the
    baseline the nesting (head-slice) claim must beat."""
    g = torch.Generator().manual_seed(seed)
    d4 = x4.shape[1]
    vals = [linear_cka(x2, x4[:, torch.randperm(d4, generator=g)[:d2]]) for _ in range(n_slices)]
    return sum(vals) / len(vals)


# -- torch GD logistic probe (no sklearn) -----------------------------------

def fit_logistic_probe(
    feats: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-batch gradient-descent logistic regression on standardized features,
    with STRONG L2 (weight_decay) since d >> n. feats[n, d] float, labels[n] in
    {0, 1}. Returns (w, b, mu, sd); apply with `probe_scores`. Standardization
    stats are returned so the SAME transform is used on held-out / transfer data.
    Raises on non-finite input."""
    require_finite("fit_logistic_probe", feats, labels)
    feats = feats.float()
    labels = labels.float()
    mu = feats.mean(0)
    sd = feats.std(0).clamp_min(1e-6)
    x = (feats - mu) / sd
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, labels)
        loss.backward()
        opt.step()
    return w.detach(), b.detach(), mu, sd


def probe_scores(
    feats: torch.Tensor, probe: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Decision scores X @ w + b under the probe's own standardization.
    Raises on non-finite input."""
    require_finite("probe_scores", feats)
    w, b, mu, sd = probe
    x = (feats.float() - mu) / sd
    return x @ w + b


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    s = [float(v) for v in scores]
    lab = ["yes" if int(t) == 1 else "no" for t in labels]
    a = rank_auc(s, lab)
    return None if a is None else max(a, 1 - a)


def _kfold(n: int, k: int, seed: int) -> list[list[int]]:
    """Round-robin k-fold index partition of range(n) (reproducible)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return [perm[i::k] for i in range(k)]


def heldout_auc(
    feats: torch.Tensor, labels: torch.Tensor, method: str = "diffmeans", k: int = 5, seed: int = 0
) -> float | None:
    """k-fold HELD-OUT AUC (never in-sample). method='diffmeans' scores the 1-D
    diff-of-means projection fit on train (robust under d>>n); method='probe' uses
    the strongly-L2 logistic probe. Returns pooled held-out AUC.
    Raises on non-finite input."""
    require_finite("heldout_auc", feats)
    n = feats.shape[0]
    if n < k:
        return None
    hs: list[float] = []
    hl: list[int] = []
    for fold in _kfold(n, k, seed):
        test = set(fold)
        tr = [i for i in range(n) if i not in test]
        te = [i for i in range(n) if i in test]
        if not te:
            continue
        pos = [feats[i] for i in tr if int(labels[i]) == 1]
        neg = [feats[i] for i in tr if int(labels[i]) == 0]
        if not pos or not neg:
            continue
        if method == "diffmeans":
            d = diff_of_means_direction(pos, neg)
            sc = [float(feats[i] @ d) for i in te]
        else:
            probe = fit_logistic_probe(feats[tr], labels[tr], steps=200)
            sc = [float(v) for v in probe_scores(feats[te], probe)]
        hs.extend(sc)
        hl.extend(int(labels[i]) for i in te)
    return _auc(torch.tensor(hs), torch.tensor(hl)) if hs else None


# -- layer matching ---------------------------------------------------------

def match_layers(n_small: int, n_large: int) -> list[tuple[int, int]]:
    """Depth-fraction alignment between a shallow (n_small) and deep (n_large)
    model: small layer i <-> large layer round(i/(n_small-1) * (n_large-1))."""
    if n_small < 2 or n_large < 2:
        return [(0, 0)]
    return [(i, round(i * (n_large - 1) / (n_small - 1))) for i in range(n_small)]


# -- capture helper ---------------------------------------------------------

def _capture_variant(
    variant: str, prompts: list[str], settings: Settings | None
) -> tuple[list[dict[int, torch.Tensor]], int, int]:
    """Load one variant, capture ALL-position residuals (assistant-prefill readout)
    at every layer for each prompt, then free the model. Returns (per_prompt, where
    per_prompt[p] = {layer: Tensor[seq, d]}, n_layers, d_model)."""
    from ..inference.hf_local import GemmaLocal

    s = (settings or Settings()).model_copy(update={"model_variant": variant})
    rec = ActivationRecorder(GemmaLocal(s))
    n_layers = rec.n_layers
    layers = list(range(n_layers))
    per_prompt = [rec.all_token_residuals(RECALL_INSTRUCTION, layers, True, p) for p in prompts]
    d_model = int(per_prompt[0][0].shape[-1])

    rec.gemma._model = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return per_prompt, n_layers, d_model


def _aligned_matrices(
    ap2: list[dict[int, torch.Tensor]], ap4: list[dict[int, torch.Tensor]], ls: int, ll: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack all-position residuals at (ls, ll), aligned per prompt by truncating to
    the shorter sequence (E2B/E4B share the tokenizer so prompts match; min() guards
    any off-by-one in chat-template specials). Returns (X2[n_repr, d2], X4[n_repr, d4])."""
    rows2, rows4 = [], []
    for p2, p4 in zip(ap2, ap4, strict=True):
        a, b = p2[ls], p4[ll]
        m = min(a.shape[0], b.shape[0])
        rows2.append(a[:m])
        rows4.append(b[:m])
    return torch.cat(rows2), torch.cat(rows4)


# -- experiment -------------------------------------------------------------

def run(
    settings: Settings | None = None,
    corpus_path: str | Path | None = None,
    n_random_slices: int = 5,
    out_dir: str | Path | None = None,
) -> dict:
    """E2: capture E2B and E4B all-position residuals on the same contrast and report
    per-layer multi-position CKA, the nesting test (head/tail/random slice), and
    held-out per-model decodability. Loads each checkpoint once, one at a time."""
    import logfire

    from .. import observability

    observability.setup()
    settings = settings or Settings()
    corpus_path = Path(corpus_path or (settings.data_dir / "eval" / "entity_knowledge_contrast.json"))
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    known, unknown = corpus["known"], corpus["unknown"]
    prompts = [it["prompt"] for it in known] + [it["prompt"] for it in unknown]
    labels = torch.tensor([1] * len(known) + [0] * len(unknown))

    with logfire.span("interp.matformer_elastic.run", n_prompts=len(prompts)):
        ap2, n2, d2 = _capture_variant("e2b", prompts, settings)
        ap4, n4, d4 = _capture_variant("e4b", prompts, settings)
        pairs = match_layers(n2, n4)

        # Numerical health: a pair with EITHER side contaminated is reported as
        # N/A-with-reason — never computed as if it were valid (fail-loud policy).
        cont2 = contaminated_layers_from_residuals(ap2)
        cont4 = contaminated_layers_from_residuals(ap4)
        numerical_health = {
            "contaminated_layers_e2b": sorted(cont2),
            "contaminated_layers_e4b": sorted(cont4),
            "n_prompts_affected_e2b": {str(k): v for k, v in cont2.items()},
            "n_prompts_affected_e4b": {str(k): v for k, v in cont4.items()},
            "policy": "layer pairs with either side contaminated are N/A-with-reason; "
                      "no winsorizing",
            "winsorize": False,
        }
        if cont2 or cont4:
            print(f"  [numerical-health] contaminated layers — E2B {sorted(cont2)}, "
                  f"E4B {sorted(cont4)}: affected pairs reported N/A, excluded from means")

        per_layer = []
        n_repr = 0
        for ls, ll in pairs:
            if ls in cont2 or ll in cont4:
                per_layer.append({
                    "e2b_layer": ls, "e4b_layer": ll,
                    "cka": None, "cka_head_slice": None, "cka_tail_slice": None,
                    "cka_random_slice": None,
                    "decode_auc_e2b_diffmeans": None, "decode_auc_e4b_diffmeans": None,
                    "decode_auc_e2b_probe": None, "decode_auc_e4b_probe": None,
                    "na_reason": "non-finite activations: "
                                 + ("e2b_layer " + str(ls) + " " if ls in cont2 else "")
                                 + ("e4b_layer " + str(ll) if ll in cont4 else ""),
                })
                continue
            X2, X4 = _aligned_matrices(ap2, ap4, ls, ll)          # [n_repr, d*]
            n_repr = X2.shape[0]
            cka = linear_cka(X2, X4)
            head = linear_cka(X2, X4[:, :d2]) if d4 >= d2 else None
            tail = linear_cka(X2, X4[:, -d2:]) if d4 >= d2 else None
            rand = _random_slice_cka(X2, X4, d2, n_random_slices, seed=ls) if d4 >= d2 else None
            # held-out decodability uses the LAST token per prompt (the entity signal)
            last2 = torch.stack([p[ls][-1] for p in ap2])         # [n_prompts, d2]
            last4 = torch.stack([p[ll][-1] for p in ap4])         # [n_prompts, d4]
            per_layer.append({
                "e2b_layer": ls, "e4b_layer": ll,
                "cka": round(cka, 4),
                "cka_head_slice": round(head, 4) if head is not None else None,
                "cka_tail_slice": round(tail, 4) if tail is not None else None,
                "cka_random_slice": round(rand, 4) if rand is not None else None,
                "decode_auc_e2b_diffmeans": _round(heldout_auc(last2, labels, "diffmeans")),
                "decode_auc_e4b_diffmeans": _round(heldout_auc(last4, labels, "diffmeans")),
                "decode_auc_e2b_probe": _round(heldout_auc(last2, labels, "probe")),
                "decode_auc_e4b_probe": _round(heldout_auc(last4, labels, "probe")),
            })

        def _mean(key: str) -> float | None:
            vals = [r[key] for r in per_layer if r[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        mean_head, mean_rand, mean_tail = _mean("cka_head_slice"), _mean("cka_random_slice"), _mean("cka_tail_slice")
        summary = {
            "d_model_e2b": d2, "d_model_e4b": d4, "n_layers_e2b": n2, "n_layers_e4b": n4,
            "dims_match": d2 == d4,
            "n_repr_cka": int(n_repr),
            "n_repr_note": ("OK (n_repr > 1000, CKA usable)" if n_repr > 1000 else
                            "LOW — CKA is upward-biased at this n_repr; treat as indicative only"),
            "probe_transfer": "N/A — E2B/E4B residual dims differ (1536 vs 2560); direct "
                              "vector/probe transfer is dimensionally ill-posed. Per-model "
                              "held-out decodability + dimension-agnostic CKA reported instead.",
            "mean_cka": _mean("cka"),
            "max_cka": _max("cka", per_layer),
            "n_pairs_na_contaminated": sum(1 for r in per_layer if r.get("na_reason")),
            "mean_cka_head_slice": mean_head,
            "mean_cka_tail_slice": mean_tail,
            "mean_cka_random_slice": mean_rand,
            # Nesting requires head > random (not just head > tail).
            "nesting_head_beats_random": (mean_head > mean_rand) if (mean_head and mean_rand) else None,
            "nesting_head_beats_tail": (mean_head > mean_tail) if (mean_head and mean_tail) else None,
            "decode_auc_e2b_diffmeans_max": _max("decode_auc_e2b_diffmeans", per_layer),
            "decode_auc_e4b_diffmeans_max": _max("decode_auc_e4b_diffmeans", per_layer),
        }

        result = {
            "experiment": "matformer_elastic",
            "n_prompts": len(prompts),
            "summary": summary,
            "numerical_health": numerical_health,
            "per_layer": per_layer,
        }
        out_dir = Path(out_dir or (settings.data_dir / "eval" / "results"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"matformer_elastic_{stamp}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = summary
    print("\n" + "=" * 68)
    print("E2 MATFORMER-ELASTIC — alignment E2B vs E4B (multi-position, held-out)")
    print("=" * 68)
    print(f"  E2B {s['d_model_e2b']}d/{s['n_layers_e2b']}L vs E4B {s['d_model_e4b']}d/{s['n_layers_e4b']}L"
          f"   n_repr={s['n_repr_cka']}  [{s['n_repr_note']}]")
    print(f"  mean CKA across matched layers = {s['mean_cka']} (max {s['max_cka']})")
    print(f"  nesting: head={s['mean_cka_head_slice']} vs random={s['mean_cka_random_slice']} "
          f"vs tail={s['mean_cka_tail_slice']}  -> head>random={s['nesting_head_beats_random']}")
    print(f"  held-out decodability (diff-of-means, max layer): "
          f"E2B={s['decode_auc_e2b_diffmeans_max']} E4B={s['decode_auc_e4b_diffmeans_max']}")
    print(f"  probe transfer: {s['probe_transfer']}")
    print(f"  wrote {out_path}")
    print("=" * 68)
    return result


def _round(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def _max(key: str, rows: list[dict]) -> float | None:
    vals = [r[key] for r in rows if r[key] is not None]
    return round(max(vals), 4) if vals else None


if __name__ == "__main__":
    run()
