"""O5.2 — Causal Steering (Sufficiency) of Truth on Gemma 3.
Steers the model on held-out statement prompts to test if the extracted d_truth
direction causally influences the model's factual belief (probabilities of "Yes" vs "No"),
evaluated against N=20 random controls.

Run:
    python calibration/o5_truth/o5_steering.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_lab.interp.directions import steering

MODEL_ID = "google/gemma-3-270m-it"
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def load():
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN absent — google/gemma-3-270m-it is gated")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
        dtype=torch.float32, attn_implementation="eager",
    ).to(DEVICE)
    model.eval()
    return tok, model


def first_token_id(tok, text: str) -> int:
    # Get first token ID without special tokens or space prefixes
    ids = tok(text.strip(), add_special_tokens=False)["input_ids"]
    return int(ids[0])


def get_logits_last(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def capture_resid_norm(model, layers, tokenizer, prompt: str, layer_idx: int) -> float:
    """Helper to capture the residual norm at the final position to scale steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    sink = {}

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        sink["norm"] = float(h[0, -1, :].detach().float().cpu().norm())

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return sink["norm"]


def run_steering():
    # Load probing results to locate best layer and direction
    probing_results_path = RESULTS_DIR / "o5_probing_results_270m.json"
    directions_path = RESULTS_DIR / "d_truth_270m.json"
    
    if not probing_results_path.exists() or not directions_path.exists():
        raise RuntimeError("Probing results not found. Please run o5_probing.py first.")
        
    with open(probing_results_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(directions_path, "r", encoding="utf-8") as f:
        directions = json.load(f)
        
    best_layer = int(meta["best_layer"])
    print(f"Using best layer {best_layer} (held-out AUC: {meta['best_auc']}) for steering.")
    
    # Load direction vector for best layer
    d_truth = torch.tensor(directions[str(best_layer)], dtype=torch.float32)
    
    tok, model = load()
    layers = model.model.layers
    
    # Token IDs for "Yes" and "No"
    yes_id = first_token_id(tok, "Yes")
    no_id = first_token_id(tok, "No")
    print(f"Answer tokens: 'Yes' (id {yes_id}), 'No' (id {no_id})")
    
    # Load evaluation dataset (Animals category statements)
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "truth_contrast.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    val_items = [x for x in data if x["category"] == "animals"]
    
    # Generate random control directions
    g = torch.Generator().manual_seed(42)
    random_dirs = []
    for _ in range(20):
        d_rand = torch.randn(d_truth.shape, generator=g)
        d_rand = d_rand / d_rand.norm()
        random_dirs.append(d_rand)
        
    results = []
    
    print("\nRunning causal steering experiment...")
    print(f"{'Statement':<35}{'Label':<8}{'Clean Prob':<12}{'Steered Prob':<15}{'Max Rand Prob':<15}")
    print("-" * 85)
    
    for item in val_items:
        # Format query prompt in chat format
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": f"Is this statement true? \"{item['statement']}\" Answer with only the word Yes or No."}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)
        
        # Capture resid norm at the target layer
        resid_norm = capture_resid_norm(model, layers, tok, prompt_text, best_layer)
        
        # Steering coefficient: scale relative to the residual stream norm at this layer
        # Marks & Tegmark / Rimsky scale steering based on activation magnitudes
        coeff = 1.0 * resid_norm
        
        # Clean run
        clean_logits = get_logits_last(model, inputs)
        clean_probs = torch.softmax(clean_logits, dim=-1)
        
        # Target token depends on whether the statement is True or False
        # If statement is True -> target is "Yes", steer with +d_truth
        # If statement is False -> target is "No", steer with -d_truth (making it believe False -> answer "No")
        # Let's test making the model believe the OPPOSITE (induce hallucination/error)
        # e.g., on True statements, steer with -d_truth to make it answer "No"
        # e.g., on False statements, steer with +d_truth to make it answer "Yes"
        if item["label"] == "true":
            target_id = yes_id
            steer_dir = -d_truth  # Make it believe False -> decrease Yes prob
            control_description = "steer true to false (decrease Yes prob)"
        else:
            target_id = no_id
            steer_dir = d_truth  # Make it believe True -> decrease No prob (increase Yes)
            control_description = "steer false to true (decrease No prob)"
            
        clean_target_prob = float(clean_probs[target_id])
        
        # Steer with d_truth
        with steering(layers[best_layer:best_layer+1], steer_dir, coeff=coeff, positions="last"):
            steered_logits = get_logits_last(model, inputs)
        steered_probs = torch.softmax(steered_logits, dim=-1)
        steered_target_prob = float(steered_probs[target_id])
        
        # Steer with random directions to record controls
        rand_probs = []
        for d_rand in random_dirs:
            # Match steering sign
            with steering(layers[best_layer:best_layer+1], -d_rand if item["label"] == "true" else d_rand, coeff=coeff, positions="last"):
                r_logits = get_logits_last(model, inputs)
            r_probs = torch.softmax(r_logits, dim=-1)
            rand_probs.append(float(r_probs[target_id]))
            
        max_rand_prob = max(rand_probs)
        min_rand_prob = min(rand_probs)
        
        # Ratio of drop (lower probability means successful steering of belief away from truth)
        # We want d_truth steering to drop target_prob significantly MORE than random controls
        clean_vs_steered_diff = clean_target_prob - steered_target_prob
        clean_vs_rand_diffs = [clean_target_prob - rp for rp in rand_probs]
        max_rand_diff = max(clean_vs_rand_diffs)
        
        # If the drop under d_truth is > 2x the max drop under random controls, it is highly specific
        specificity_ratio = clean_vs_steered_diff / max(max_rand_diff, 1e-5)
        
        print(f"{item['statement']:<35}{item['label']:<8}{clean_target_prob:<12.4f}{steered_target_prob:<15.4f}{max_rand_prob:<15.4f}")
        
        results.append({
            "statement": item["statement"],
            "label": item["label"],
            "clean_target_prob": clean_target_prob,
            "steered_target_prob": steered_target_prob,
            "max_rand_prob": max_rand_prob,
            "specificity_ratio": specificity_ratio
        })
        
    # Calculate summary metrics
    mean_clean = sum(r["clean_target_prob"] for r in results) / len(results)
    mean_steered = sum(r["steered_target_prob"] for r in results) / len(results)
    mean_rand = sum(r["max_rand_prob"] for r in results) / len(results)
    mean_ratio = sum(r["specificity_ratio"] for r in results) / len(results)
    
    passed = bool(mean_ratio > 2.0 or (mean_clean - mean_steered) > 2.0 * (mean_clean - mean_rand))
    
    print("-" * 85)
    print(f"Mean target probability: Clean {mean_clean:.4f} -> Steered {mean_steered:.4f} (Max Random: {mean_rand:.4f})")
    print(f"Mean specificity ratio: {mean_ratio:.2f}x")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    
    output_meta = {
        "objective": "O5.2",
        "status": "done",
        "model_id": MODEL_ID,
        "best_layer": best_layer,
        "mean_clean": round(mean_clean, 4),
        "mean_steered": round(mean_steered, 4),
        "mean_max_random": round(mean_rand, 4),
        "mean_specificity_ratio": round(mean_ratio, 2),
        "pass": passed,
        "per_item": results
    }
    
    results_path = RESULTS_DIR / "o5_steering_results_270m.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)
    print(f"Saved steering results to {results_path}")


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    run_steering()
