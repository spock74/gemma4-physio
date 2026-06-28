"""Diagnostic script to verify text generation under rotational steering.
Generates greedy text completions under 4 conditions:
1. Clean baseline
2. Theta = 20° (Peak steering recovery)
3. Theta = 120° (Dead zone)
4. Theta = 200° (Orthogonal Attractor Translation)
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Any

sys.path.append("/Users/moraes/Documents/PROJETOS/interpretability/started-june-26/zero/src")

from gemma4_lab.config import hf_token_or_none
from gemma4_lab.interp.directions import diff_of_means_direction
from gemma4_lab.interp.entity_knowledge import RECALL_INSTRUCTION
from transformers import AutoModelForImageTextToText, AutoTokenizer

CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"
CORPUS_PATH = Path("data/eval/entity_knowledge_contrast.json")
TARGET_LAYER = 12

def locate_layers(model):
    for path in (("model", "layers"), ("model", "language_model", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("decoder layers not found")

def _apply_at(hidden: torch.Tensor, fn: Any, positions: str | list[int] | None) -> torch.Tensor:
    if positions is None:
        return fn(hidden)
    seq = hidden.shape[-2]
    idx = [seq - 1] if positions == "last" else [p if p >= 0 else seq + p for p in positions]
    sel = [slice(None)] * hidden.dim()
    sel[-2] = idx
    sel = tuple(sel)
    out = hidden.clone()
    out[sel] = fn(hidden[sel])
    return out

@contextmanager
def rotational_steering(layer: nn.Module, direction1: torch.Tensor, direction2: torch.Tensor, theta: float, R: float, positions: str = "last"):
    def make_hook():
        def rotate_slice(x):
            d1 = direction1.to(x.device, x.dtype)
            d2 = direction2.to(x.device, x.dtype)
            v_interv = R * (math.cos(theta) * d1 + math.sin(theta) * d2)
            return x + v_interv

        def hook(_m, _i, output):
            if isinstance(output, tuple):
                h = output[0]
                return (_apply_at(h, rotate_slice, positions), *output[1:])
            return _apply_at(output, rotate_slice, positions)
        return hook

    handle = layer.register_forward_hook(make_hook())
    try:
        yield
    finally:
        handle.remove()

def main():
    token = hf_token_or_none()
    if not token:
        print("HF_TOKEN missing!")
        return 1

    print("Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", cache_dir=CACHE_DIR, token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-3-4b-it", cache_dir=CACHE_DIR, token=token,
        dtype=torch.bfloat16, attn_implementation="eager"
    ).to(DEVICE)
    model.eval()
    layers = locate_layers(model)

    # Load corpus
    corpus = json.loads(CORPUS_PATH.read_text())
    known = corpus["known"]
    unknown = corpus["unknown"]

    # Select 3 test prompts
    test_prompts = known[:3]

    # Use a subset of 8 prompts for extracting the knowledge direction
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    k_subset = [known[i] for i in subset_indices]
    u_subset = [unknown[i] for i in subset_indices]

    # Pre-tokenize subset for direction extraction
    k_inputs = []
    for it in k_subset:
        p = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                    tokenize=False, add_generation_prompt=True) + it["prompt"]
        inputs = tok(p, return_tensors="pt").to(DEVICE)
        k_inputs.append(inputs)

    u_inputs = []
    for it in u_subset:
        p = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                    tokenize=False, add_generation_prompt=True) + it["prompt"]
        inputs = tok(p, return_tensors="pt").to(DEVICE)
        u_inputs.append(inputs)

    # Capture activations to build bases
    print("Capturing activations...")
    k_res, u_res = [], []
    def make_capture_hook(res_list):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            res_list.append(h[0, -1, :].detach().float().cpu())
        return hook

    for inputs in k_inputs:
        h = layers[TARGET_LAYER].register_forward_hook(make_capture_hook(k_res))
        with torch.no_grad():
            model(**inputs)
        h.remove()

    for inputs in u_inputs:
        h = layers[TARGET_LAYER].register_forward_hook(make_capture_hook(u_res))
        with torch.no_grad():
            model(**inputs)
        h.remove()

    # Calculate bases
    v1 = diff_of_means_direction(k_res, u_res).to(DEVICE).to(torch.bfloat16)
    u_mean = torch.stack(k_res).mean(0).to(DEVICE).to(torch.bfloat16)
    v2_sem = u_mean - torch.dot(u_mean, v1) * v1
    v2_sem = v2_sem / (v2_sem.norm() + 1e-8)

    print("\nStarting generation tests...")
    R = 10000.0
    conditions = {
        "Baseline": None,
        "Theta = 20° (Peak)": math.radians(20),
        "Theta = 120° (Dead)": math.radians(120),
        "Theta = 200° (Recovery)": math.radians(200),
    }

    for it in test_prompts:
        prompt_text = it["prompt"]
        gold_answer = it["answer"]
        print("-" * 80)
        print(f"Prompt: '{prompt_text}' (Expected: '{gold_answer}')")
        
        # Prepare generation input
        p = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                    tokenize=False, add_generation_prompt=True) + prompt_text
        inputs = tok(p, return_tensors="pt").to(DEVICE)
        
        for name, theta in conditions.items():
            if theta is None:
                # Baseline
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
            else:
                # Steered
                with rotational_steering(layers[TARGET_LAYER], v1, v2_sem, theta, R, positions="last"):
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
            
            gen_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            gen_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            print(f"  [{name:22s}]: '{gen_text}'")

    print("-" * 80)
    return 0

if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    sys.exit(main())
