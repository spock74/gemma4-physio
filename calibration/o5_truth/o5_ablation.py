"""O5.3 — Causal Ablation (Necessity) of Truth on Gemma 3.
Ablates the d_truth direction at the best layer to test if the model's factual
accuracy (probability of answering "Yes" to True statements) decreases,
evaluated against N=20 random controls.

Run:
    python calibration/o5_truth/o5_ablation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_lab.interp.directions import ablating

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
    ids = tok(text.strip(), add_special_tokens=False)["input_ids"]
    return int(ids[0])


def get_logits_last(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def run_ablation():
    probing_results_path = RESULTS_DIR / "o5_probing_results_270m.json"
    directions_path = RESULTS_DIR / "d_truth_270m.json"
    
    if not probing_results_path.exists() or not directions_path.exists():
        raise RuntimeError("Probing results not found. Please run o5_probing.py first.")
        
    with open(probing_results_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(directions_path, "r", encoding="utf-8") as f:
        directions = json.load(f)
        
    best_layer = int(meta["best_layer"])
    print(f"Using best layer {best_layer} (held-out AUC: {meta['best_auc']}) for ablation.")
    
    d_truth = torch.tensor(directions[str(best_layer)], dtype=torch.float32)
    
    tok, model = load()
    layers = model.model.layers
    
    yes_id = first_token_id(tok, "Yes")
    print(f"Target token: 'Yes' (id {yes_id})")
    
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "truth_contrast.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # We only test ablation on TRUE statements (where the model should answer "Yes")
    true_items = [x for x in data if x["category"] == "animals" and x["label"] == "true"]
    
    # Generate random controls
    g = torch.Generator().manual_seed(42)
    random_dirs = []
    for _ in range(20):
        d_rand = torch.randn(d_truth.shape, generator=g)
        d_rand = d_rand / d_rand.norm()
        random_dirs.append(d_rand)
        
    results = []
    
    print("\nRunning causal ablation experiment...")
    print(f"{'Statement':<35}{'Clean Prob':<12}{'Ablated Prob':<15}{'Min Rand Prob':<15}")
    print("-" * 80)
    
    for item in true_items:
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": f"Is this statement true? \"{item['statement']}\" Answer with only the word Yes or No."}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)
        
        # Clean run
        clean_logits = get_logits_last(model, inputs)
        clean_probs = torch.softmax(clean_logits, dim=-1)
        clean_target_prob = float(clean_probs[yes_id])
        
        # Ablated run (project out d_truth at last position)
        with ablating(layers[best_layer:best_layer+1], d_truth, positions="last"):
            ablated_logits = get_logits_last(model, inputs)
        ablated_probs = torch.softmax(ablated_logits, dim=-1)
        ablated_target_prob = float(ablated_probs[yes_id])
        
        # Random controls runs
        rand_probs = []
        for d_rand in random_dirs:
            with ablating(layers[best_layer:best_layer+1], d_rand, positions="last"):
                r_logits = get_logits_last(model, inputs)
            r_probs = torch.softmax(r_logits, dim=-1)
            rand_probs.append(float(r_probs[yes_id]))
            
        max_rand_prob = max(rand_probs)
        min_rand_prob = min(rand_probs)
        
        # Calculate drops (clean - ablated)
        clean_vs_ablated_diff = clean_target_prob - ablated_target_prob
        clean_vs_rand_diffs = [clean_target_prob - rp for rp in rand_probs]
        max_rand_diff = max(clean_vs_rand_diffs)
        
        specificity_ratio = clean_vs_ablated_diff / max(max_rand_diff, 1e-5)
        
        print(f"{item['statement']:<35}{clean_target_prob:<12.4f}{ablated_target_prob:<15.4f}{min_rand_prob:<15.4f}")
        
        results.append({
            "statement": item["statement"],
            "clean_target_prob": clean_target_prob,
            "ablated_target_prob": ablated_target_prob,
            "min_rand_prob": min_rand_prob,
            "specificity_ratio": specificity_ratio
        })
        
    mean_clean = sum(r["clean_target_prob"] for r in results) / len(results)
    mean_ablated = sum(r["ablated_target_prob"] for r in results) / len(results)
    mean_rand_min = sum(r["min_rand_prob"] for r in results) / len(results)
    mean_ratio = sum(r["specificity_ratio"] for r in results) / len(results)
    
    passed = bool(mean_ratio > 2.0 or (mean_clean - mean_ablated) > 2.0 * (mean_clean - mean_rand_min))
    
    print("-" * 80)
    print(f"Mean target probability: Clean {mean_clean:.4f} -> Ablated {mean_ablated:.4f} (Min Random: {mean_rand_min:.4f})")
    print(f"Mean specificity ratio: {mean_ratio:.2f}x")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    
    output_meta = {
        "objective": "O5.3",
        "status": "done",
        "model_id": MODEL_ID,
        "best_layer": best_layer,
        "mean_clean": round(mean_clean, 4),
        "mean_ablated": round(mean_ablated, 4),
        "mean_min_random": round(mean_rand_min, 4),
        "mean_specificity_ratio": round(mean_ratio, 2),
        "pass": passed,
        "per_item": results
    }
    
    results_path = RESULTS_DIR / "o5_ablation_results_270m.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)
    print(f"Saved ablation results to {results_path}")


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    run_ablation()
