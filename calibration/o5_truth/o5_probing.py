"""O5.1 — Probing & Linear Separability of Truth on Gemma 3.
Fits the truth direction (d_truth) using difference-of-means on training categories
and evaluates held-out classification AUC on a disjoint validation category across all layers.

Run:
    python calibration/o5_truth/o5_probing.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_lab.interp.directions import diff_of_means_direction, rank_auc

MODEL_ID = "google/gemma-3-270m-it"
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def capture_resid(model, layers, tokenizer, statement: str, layer_idx: int) -> torch.Tensor:
    """Capture the activation of the final token of the chat prompt at the specified layer."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Is this statement true? \"{statement}\" Answer with only the word Yes or No."}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    sink = {}

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        # Get activation at final position of the sequence
        sink["h"] = h[0, -1, :].detach().float().cpu()

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return sink["h"]


def run_probing():
    tok, model = load()
    layers = model.model.layers
    num_layers = len(layers)
    
    # Load dataset
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "eval" / "truth_contrast.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Loaded dataset from {dataset_path} containing {len(data)} statements.")
    
    # Train-Val split by category to test generalizability
    # Train = geography & arithmetic (60 statements)
    # Val = animals (30 statements)
    train_items = [x for x in data if x["category"] in ("geography", "arithmetic")]
    val_items = [x for x in data if x["category"] == "animals"]
    
    print(f"Train split size: {len(train_items)} (geography, arithmetic)")
    print(f"Val split size: {len(val_items)} (animals)")
    
    results = []
    best_auc = 0.0
    best_layer = -1
    
    # We will save the calculated directions for steering/ablation later
    directions_by_layer = {}
    
    print("\nStarting per-layer probing sweep...")
    print(f"{'Layer':<8}{'Train AUC':<12}{'Val AUC (Held-out)':<20}")
    print("-" * 40)
    
    for L in range(num_layers):
        # Capture training activations
        train_pos_acts = []
        train_neg_acts = []
        for x in train_items:
            act = capture_resid(model, layers, tok, x["statement"], L)
            if x["label"] == "true":
                train_pos_acts.append(act)
            else:
                train_neg_acts.append(act)
                
        # Calculate truth direction on Train split
        d_truth = diff_of_means_direction(train_pos_acts, train_neg_acts)
        directions_by_layer[L] = d_truth.tolist()
        
        # Calculate train AUC (sanity check)
        train_scores = [float(v @ d_truth) for v in train_pos_acts] + [float(v @ d_truth) for v in train_neg_acts]
        train_labels = ["yes"] * len(train_pos_acts) + ["no"] * len(train_neg_acts)
        train_auc = rank_auc(train_scores, train_labels)
        train_auc = max(train_auc, 1.0 - train_auc)
        
        # Capture validation activations & project onto d_truth
        val_pos_acts = []
        val_neg_acts = []
        for x in val_items:
            act = capture_resid(model, layers, tok, x["statement"], L)
            if x["label"] == "true":
                val_pos_acts.append(act)
            else:
                val_neg_acts.append(act)
                
        val_scores = [float(v @ d_truth) for v in val_pos_acts] + [float(v @ d_truth) for v in val_neg_acts]
        val_labels = ["yes"] * len(val_pos_acts) + ["no"] * len(val_neg_acts)
        val_auc = rank_auc(val_scores, val_labels)
        val_auc = max(val_auc, 1.0 - val_auc)
        
        print(f"{L:<8}{train_auc:<12.4f}{val_auc:<20.4f}")
        
        results.append({
            "layer": L,
            "train_auc": round(train_auc, 4),
            "val_auc": round(val_auc, 4)
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_layer = L
            
    print("-" * 40)
    print(f"Probing complete. Best held-out Val AUC: {best_auc:.4f} at Layer {best_layer}")
    
    # Save results summary
    output_meta = {
        "objective": "O5.1",
        "status": "done",
        "model_id": MODEL_ID,
        "best_layer": best_layer,
        "best_auc": round(best_auc, 4),
        "layers": results
    }
    
    results_path = RESULTS_DIR / "o5_probing_results_270m.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)
    print(f"Saved results summary to {results_path}")
    
    # Save the fitted directions to disk so we don't have to re-compute them in steering/ablation
    directions_path = RESULTS_DIR / "d_truth_270m.json"
    with open(directions_path, "w", encoding="utf-8") as f:
        json.dump(directions_by_layer, f)
    print(f"Saved fitted truth directions to {directions_path}")


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    run_probing()
