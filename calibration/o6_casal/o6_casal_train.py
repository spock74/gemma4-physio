"""O6.1 — Amortized Activation Steering (CASAL on Gemma 3).
Trains the MLP weights of Layer 7 to natively bake the known/unknown steering effect
into the model weights, using a joint MSE representation loss and LM cross-entropy loss.

Run:
    python calibration/o6_casal/o6_casal_train.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


from gemma4_lab import observability
from gemma4_lab.interp.directions import diff_of_means_direction

MODEL_ID = "google/gemma-3-270m-it"
CACHE_DIR = "/Users/moraes/Documents/PROJETOS/main-projects/gemma4-lab/models/hf-cache"
DEVICE = "mps"

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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


def main():
    observability.setup()
    
    # 1. Load Model & Data
    tok, model = load()
    layers = model.model.layers
    L_target = 7  # Peak necessity layer for 270m model
    
    corpus = json.loads(CORPUS.read_text())
    known, unknown = corpus["known"], corpus["unknown"]
    k_tr, k_va = stratified_split(known, 0)
    u_tr, u_va = stratified_split(unknown, 1)
    
    print(f"Loaded {MODEL_ID}. Target layer for CASAL training: Layer {L_target}")
    
    # 2. Extract baseline d_know at Layer 7 on Train split (using hook)
    # We do a standard capture pass first to fit d_know
    print("Extracting baseline d_know at Layer 7...")
    k_res_tr = []
    u_res_tr = []
    
    def make_hook(sink):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            sink.append(h[0, -1, :].detach().float().cpu())
        return hook
        
    # Capture known train activations
    for i in k_tr:
        sink = []
        inputs = recall_inputs(tok, known[i]["prompt"])
        handle = layers[L_target].register_forward_hook(make_hook(sink))
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        k_res_tr.append(sink[0])
        
    # Capture unknown train activations
    for i in u_tr:
        sink = []
        inputs = recall_inputs(tok, unknown[i]["prompt"])
        handle = layers[L_target].register_forward_hook(make_hook(sink))
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        u_res_tr.append(sink[0])
        
    d_know = diff_of_means_direction(k_res_tr, u_res_tr).to(DEVICE)
    print("Baseline d_know extracted successfully.")
    
    # 3. Freeze all parameters except Layer 7 MLP
    print("Freezing model parameters...")
    for p in model.parameters():
        p.requires_grad = False
        
    mlp = layers[L_target].mlp
    for p in mlp.parameters():
        p.requires_grad = True
        
    trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"Trainable parameters in Layer 7 MLP: {trainable_params:,}")
    
    # 4. Prepare training inputs and target steered activations
    # We pre-compute the target steered residual activation h_target for each training prompt
    print("Pre-computing steered activation targets...")
    train_prompts = []
    
    alpha = 0.2  # Steering strength
    
    # Process known train items
    for i in k_tr:
        it = known[i]
        inputs = recall_inputs(tok, it["prompt"])
        gold_id = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        
        # Get clean activation
        sink = []
        handle = layers[L_target].register_forward_hook(make_hook(sink))
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        h_clean = sink[0].to(DEVICE)
        
        # Steer towards "known" (+d_know)
        h_target = h_clean + alpha * d_know * h_clean.norm()
        
        train_prompts.append({
            "inputs": inputs,
            "gold_id": gold_id,
            "h_target": h_target,
            "type": "known"
        })
        
    # Process unknown train items
    for i in u_tr:
        it = unknown[i]
        inputs = recall_inputs(tok, it["prompt"])
        
        # Get clean activation
        sink = []
        handle = layers[L_target].register_forward_hook(make_hook(sink))
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        h_clean = sink[0].to(DEVICE)
        
        # Steer towards "unknown" (-d_know) to suppress hallucination
        h_target = h_clean - alpha * d_know * h_clean.norm()
        
        train_prompts.append({
            "inputs": inputs,
            "gold_id": None,
            "h_target": h_target,
            "type": "unknown"
        })
        
    # 5. Training Loop
    optimizer = AdamW(mlp.parameters(), lr=1.5e-5, weight_decay=0.01)
    
    epochs = 35
    beta = 0.5  # LM regularization strength
    
    print(f"\nStarting CASAL training loop for {epochs} epochs...")
    print(f"{'Epoch':<8}{'MSE Loss':<12}{'LM Loss':<12}{'Total Loss':<12}")
    print("-" * 45)
    
    for epoch in range(epochs):
        epoch_mse = 0.0
        epoch_lm = 0.0
        epoch_total = 0.0
        
        model.train()
        
        for item in train_prompts:
            # We hook the trainable layer to capture the actual activation
            sink_trainable = {}
            
            def hook_trainable(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                sink_trainable["h"] = h[0, -1, :]
                
            handle = layers[L_target].register_forward_hook(hook_trainable)
            
            # Forward pass
            out = model(**item["inputs"])
            handle.remove()
            
            # Calculate MSE loss
            h_act = sink_trainable["h"]
            loss_mse = torch.mean((h_act - item["h_target"]) ** 2)
            
            # Calculate LM loss if known prompt (where we have gold target)
            loss_lm = torch.tensor(0.0, device=DEVICE)
            if item["type"] == "known" and item["gold_id"] is not None:
                logits = out.logits[0, -1, :]
                loss_lm = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([item["gold_id"]], device=DEVICE))
                
            loss = loss_mse + beta * loss_lm
            
            # Backward & Step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_mse += float(loss_mse)
            epoch_lm += float(loss_lm)
            epoch_total += float(loss)
            
        mean_mse = epoch_mse / len(train_prompts)
        mean_lm = epoch_lm / sum(1 for x in train_prompts if x["type"] == "known")
        mean_total = epoch_total / len(train_prompts)
        
        print(f"{epoch+1:<8}{mean_mse:<12.5f}{mean_lm:<12.5f}{mean_total:<12.5f}")
        
    print("-" * 45)
    print("Training complete.")
    
    # Save the trained FFN weights to disk
    weights_path = RESULTS_DIR / "casal_mlp_weights_270m.pt"
    torch.save(mlp.state_dict(), weights_path)
    print(f"Saved trained Layer 7 MLP weights to {weights_path}")


if __name__ == "__main__":
    main()
