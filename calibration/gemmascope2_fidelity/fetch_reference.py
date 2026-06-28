"""O2 / fetch — freeze Neuronpedia reference activations for the calibration features.

Pulls features from the public Neuronpedia API for the Gemma Scope 2 source
`gemma-3-270m-it/12-gemmascope-2-res-16k` and saves each feature's top-activating
snippet (tokens + per-token values + peak) to reference/np_feature_{idx}.json.

Stdlib only (urllib) so it runs in any env. Read endpoint is public (no key).

Run: python calibration/gemmascope2_fidelity/fetch_reference.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

MODEL = "gemma-3-270m-it"
SOURCE = "12-gemmascope-2-res-16k"
CANDIDATES = list(range(0, 10))   # fetch 10, keep the live ones
N_KEEP = 5
HERE = Path(__file__).resolve().parent
REF = HERE / "reference"


def fetch(idx: int) -> dict:
    url = f"https://www.neuronpedia.org/api/feature/{MODEL}/{SOURCE}/{idx}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    REF.mkdir(exist_ok=True)
    kept = []
    for idx in CANDIDATES:
        if len(kept) >= N_KEEP:
            break
        try:
            d = fetch(idx)
        except Exception as e:  # noqa: BLE001
            print(f"  feature {idx}: fetch failed ({e}) — skip")
            continue
        acts = d.get("activations") or []
        if not acts:
            print(f"  feature {idx}: no activations — skip (likely dead)")
            continue
        snip = acts[0]  # top-activating snippet
        max_val = float(snip.get("maxValue") or 0.0)
        if max_val < 1.0:
            print(f"  feature {idx}: maxValue {max_val:.3f} < 1 — skip (weak/dead)")
            continue
        ref = {
            "modelId": MODEL, "source": SOURCE, "feature_index": idx,
            "hookName": d.get("hookName"),
            "activations": [{
                "tokens": snip["tokens"],
                "values": snip["values"],
                "maxValue": max_val,
                "maxValueTokenIndex": int(snip["maxValueTokenIndex"]),
            }],
        }
        (REF / f"np_feature_{idx}.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
        kept.append(idx)
        print(f"  feature {idx}: KEPT — {len(snip['tokens'])} tokens, "
              f"maxValue {max_val:.3f} @ idx {ref['activations'][0]['maxValueTokenIndex']}, "
              f"hook {ref['hookName']}")
    print(f"\nfroze {len(kept)} reference features: {kept}")
    if len(kept) < 3:
        print("!! need >=3 live features for the O2 gate — STOP.")
        return 1
    return 0


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    sys.exit(main())
