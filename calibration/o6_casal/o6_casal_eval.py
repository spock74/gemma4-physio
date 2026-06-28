"""O6.2 — Evaluation of Amortized Activation Steering (CASAL).
Loads the trained Layer 7 MLP weights and compares the post-CASAL model
against the baseline on held-out validation items.

Run:
    python calibration/o6_casal/o6_casal_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma4_lab.interp.directions import diff_of_means_direction, projection, rank_auc

MODEL_ID = "google/gemma-3-270m-it"
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CORPUS = Path("data/eval/entity_knowledge_contrast.json")
RECALL_INSTRUCTION = "Answer with the fact, continuing the sentence."


def load():
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN absent — google/gemma-3-270m-it is gated")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, token=os.getenv("HF_TOKEN"),
        dtype=torch.float32, attn_implementation="eager",
    ).to(DEVICE)
    return tok, model


def recall_inputs(tok, stem: str):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": RECALL_INSTRUCTION}],
        tokenize=False, add_generation_prompt=True) + stem
    return tok(prompt, return_tensors="pt").to(DEVICE)


def categorize(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "capital of" in prompt_lower:
        return "geography"
    elif "chemical symbol" in prompt_lower:
        return "chemistry"
    elif "planet" in prompt_lower or "solar system" in prompt_lower or "sun" in prompt_lower:
        return "astronomy"
    elif "currency" in prompt_lower:
        return "currency"
    else:
        return "other"


def stratified_split(items: list[dict], seed: int, val_frac=0.5) -> tuple[list[int], list[int]]:
    categories = {}
    for idx, item in enumerate(items):
        cat = categorize(item["prompt"])
        categories.setdefault(cat, []).append(idx)
        
    g = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    
    for cat, idxs in sorted(categories.items()):
        n = len(idxs)
        perm = torch.randperm(n, generator=g).tolist()
        shuffled_idxs = [idxs[p] for p in perm]
        nv = max(1, round(n * val_frac))
        val_indices.extend(shuffled_idxs[:nv])
        train_indices.extend(shuffled_idxs[nv:])
        
    return sorted(train_indices), sorted(val_indices)


def capture_resid(model, layers, tokenizer, stem: str, layer_idx: int) -> torch.Tensor:
    inputs = recall_inputs(tokenizer, stem)
    sink = []
    
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        sink.append(h[0, -1, :].detach().float().cpu())
        
    handle = layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return sink[0]


def logits_last(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return out.logits[0, -1, :].detach().float().cpu()


def main():
    tok, model = load()
    layers = model.model.layers
    L_target = 7
    
    weights_path = RESULTS_DIR / "casal_mlp_weights_270m.pt"
    if not weights_path.exists():
        raise RuntimeError(f"Trained CASAL weights not found at {weights_path}. Run training first.")
        
    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    k_tr, k_va = stratified_split(known, 0)
    u_tr, u_va = stratified_split(unknown, 1)
    
    # --- 1. Fit d_know on Train split using Baseline weights ---
    print("Fitting baseline d_know at Layer 7...")
    k_res_tr = [capture_resid(model, layers, tok, known[i]["prompt"], L_target) for i in k_tr]
    u_res_tr = [capture_resid(model, layers, tok, unknown[i]["prompt"], L_target) for i in u_tr]
    d_know = diff_of_means_direction(k_res_tr, u_res_tr)
    
    # --- 2. Evaluate Baseline model on Val split ---
    print("\nEvaluating baseline model on Val split...")
    base_k_res_va = [capture_resid(model, layers, tok, known[i]["prompt"], L_target) for i in k_va]
    base_u_res_va = [capture_resid(model, layers, tok, unknown[i]["prompt"], L_target) for i in u_va]
    
    base_scores = [projection(r, d_know) for r in base_k_res_va] + [projection(r, d_know) for r in base_u_res_va]
    base_labels = ["yes"] * len(k_va) + ["no"] * len(u_va)
    base_auc = rank_auc(base_scores, base_labels)
    base_auc = max(base_auc, 1.0 - base_auc)
    
    # Measure factual recall ranks under baseline
    base_ranks = []
    for i in k_va:
        it = known[i]
        inputs = recall_inputs(tok, it["prompt"])
        gold_id = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        logits = logits_last(model, inputs)
        rank = int((logits > logits[gold_id]).sum())
        base_ranks.append(rank)
        
    mean_base_rank = sum(base_ranks) / len(base_ranks)
    median_base_rank = sorted(base_ranks)[len(base_ranks) // 2]
    
    # --- 3. Load CASAL weights and Evaluate ---
    print("\nLoading trained CASAL FFN weights into Layer 7...")
    mlp = layers[L_target].mlp
    mlp.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    print("Evaluating CASAL-trained model on Val split...")
    casal_k_res_va = [capture_resid(model, layers, tok, known[i]["prompt"], L_target) for i in k_va]
    casal_u_res_va = [capture_resid(model, layers, tok, unknown[i]["prompt"], L_target) for i in u_va]
    
    casal_scores = [projection(r, d_know) for r in casal_k_res_va] + [projection(r, d_know) for r in casal_u_res_va]
    casal_auc = rank_auc(casal_scores, base_labels)
    casal_auc = max(casal_auc, 1.0 - casal_auc)
    
    # Measure factual recall ranks under CASAL
    casal_ranks = []
    for i in k_va:
        it = known[i]
        inputs = recall_inputs(tok, it["prompt"])
        gold_id = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        logits = logits_last(model, inputs)
        rank = int((logits > logits[gold_id]).sum())
        casal_ranks.append(rank)
        
    mean_casal_rank = sum(casal_ranks) / len(casal_ranks)
    median_casal_rank = sorted(casal_ranks)[len(casal_ranks) // 2]
    
    # --- 4. Print Summary Comparison ---
    print("\n" + "=" * 70)
    print("CASAL POST-TRAINING EVALUATION SUMMARY (Gemma 3 270m)")
    print("=" * 70)
    print(f"{'Metric':<30}{'Baseline':<20}{'CASAL-Trained':<20}")
    print("-" * 70)
    print(f"{'Held-out Val AUC':<30}{base_auc:<20.4f}{casal_auc:<20.4f}")
    print(f"{'Median Recall Rank (known)':<30}{median_base_rank:<20}{median_casal_rank:<20}")
    print(f"{'Mean Recall Rank (known)':<30}{mean_base_rank:<20.2f}{mean_casal_rank:<20.2f}")
    print("=" * 70)
    
    # Let's inspect projection shift details
    base_mean_known_proj = sum(projection(r, d_know) for r in base_k_res_va) / len(base_k_res_va)
    base_mean_unknown_proj = sum(projection(r, d_know) for r in base_u_res_va) / len(base_u_res_va)
    
    casal_mean_known_proj = sum(projection(r, d_know) for r in casal_k_res_va) / len(casal_k_res_va)
    casal_mean_unknown_proj = sum(projection(r, d_know) for r in casal_u_res_va) / len(casal_u_res_va)
    
    print("\nProjection details on d_know axis:")
    print(f"  Mean Known Projection   : Baseline {base_mean_known_proj:+.4f} -> CASAL {casal_mean_known_proj:+.4f}")
    print(f"  Mean Unknown Projection : Baseline {base_mean_unknown_proj:+.4f} -> CASAL {casal_mean_unknown_proj:+.4f}")
    
    # Verdict
    # If CASAL trained model keeps median recall rank <= 1 (retaining recall) and held-out Val AUC remains high, it works!
    passed = bool(median_casal_rank <= 1 and casal_auc >= 0.85)
    print(f"\nVerdict: {'PASS (Steering baked successfully)' if passed else 'FAIL / UNCONVERGED'}")
    
    # Save output metadata
    output_meta = {
        "objective": "O6.2",
        "status": "done",
        "model_id": MODEL_ID,
        "best_layer": L_target,
        "baseline": {
            "val_auc": round(base_auc, 4),
            "median_recall_rank": median_base_rank,
            "mean_recall_rank": round(mean_base_rank, 2),
            "mean_known_proj": round(base_mean_known_proj, 4),
            "mean_unknown_proj": round(base_mean_unknown_proj, 4)
        },
        "casal_trained": {
            "val_auc": round(casal_auc, 4),
            "median_recall_rank": median_casal_rank,
            "mean_recall_rank": round(mean_casal_rank, 2),
            "mean_known_proj": round(casal_mean_known_proj, 4),
            "mean_unknown_proj": round(casal_mean_unknown_proj, 4)
        },
        "pass": passed
    }
    
    results_path = RESULTS_DIR / "o6_casal_results_270m.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_meta, f, indent=2)
    print(f"Saved CASAL eval results to {results_path}")


if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    main()
