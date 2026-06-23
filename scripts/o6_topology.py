"""TDA Sweep on Gemma 3 4b-it: Persistent Homology of Residual Stream activations.
This script implements a 2D rotational sweep in Layer 12, captures Layer 13 activations
autoregressively for newly generated tokens, and computes Betti-0 and H1 persistence.
"""

import sys
import os
import json
import math
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy.spatial.distance import pdist, squareform
import ripser

# Suppress ripser columns vs rows warning
warnings.filterwarnings("ignore", category=UserWarning, module="ripser")

# Ensure local path is in sys.path
sys.path.append("/Users/moraes/Documents/PROJETOS/interpretability/started-june-26/zero/src")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def compute_kl_divergence(baseline_logits, steered_logits):
    """Computes KL divergence between baseline and steered logits for a single token."""
    p_log = torch.log_softmax(baseline_logits, dim=-1)
    q_log = torch.log_softmax(steered_logits, dim=-1)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum().item()

@contextmanager
def rotational_intervention(layer: nn.Module, v1: torch.Tensor, v2_k: torch.Tensor, theta: float, R: float):
    """Hooks a layer to perform a SINGLE-SHOT 2D rotational sweep projection (pre-fill only).
    
    FIX: The previous version applied R=15000 at EVERY forward pass, including during
    autoregressive KV-cache generation. This caused a death spiral where the model was
    ejected from the grammatical manifold 40 times in a row, collapsing into repetitive
    gibberish loops and inflating H1 persistence artificially.
    
    This version:
    - Returns h UNMODIFIED when h.shape[1] == 1 (autoregressive KV-cache phase)
    - Only perturbs the LAST TOKEN during the pre-fill phase (h.shape[1] > 1)
    - Leaves all prior context tokens untouched to preserve grammatical reading
    """
    def make_hook():
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            
            # 1. BYPASS AUTOREGRESSIVE PHASE (THE DEATH SPIRAL FIX)
            # If h.shape[1] == 1, the model is generating new tokens via KV Cache.
            # Turn off the steering to let natural syntax operate.
            if h.shape[1] == 1:
                return output
            
            # 2. PRE-FILL PHASE (Processing the prompt)
            # Apply perturbation ONLY to the last token (where factual prediction occurs)
            d1 = v1.to(h.device, h.dtype)
            d2 = v2_k.to(h.device, h.dtype)
            
            # Extract the last token's activation
            h_last = h[:, -1:, :]  # [batch, 1, d_model]
            
            # Record its clean norm for RMSNorm preservation
            clean_norm = h_last.norm(dim=-1, keepdim=True)
            
            # Orthogonal projection on the last token only
            proj_v1 = torch.einsum('bsd,d->bs', h_last, d1).unsqueeze(-1) * d1
            proj_v2 = torch.einsum('bsd,d->bs', h_last, d2).unsqueeze(-1) * d2
            h_last_perp = h_last - (proj_v1 + proj_v2)
            
            # Rotational perturbation
            perturbation = R * (math.cos(theta) * d1 + math.sin(theta) * d2)
            patched_raw = h_last_perp + perturbation
            
            # Rescale to match clean norm (RMSNorm bypass)
            patched_norm = patched_raw.norm(dim=-1, keepdim=True) + 1e-8
            h_last_patched = patched_raw * (clean_norm / patched_norm)
            
            # Reconstruct the full tensor: context tokens unchanged, last token patched
            h_out = h.clone()
            h_out[:, -1:, :] = h_last_patched
            
            if isinstance(output, tuple):
                return (h_out, *output[1:])
            return h_out
        return hook

    handle = layer.register_forward_hook(make_hook())
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def capture_activations(layer: nn.Module, sink: list[torch.Tensor]):
    """Hooks a layer to capture activation hidden states [batch, 1, d_model] only for newly generated tokens (seq_len == 1)."""
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:
            sink.append(h.detach().cpu())
    
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

def compute_tda_metrics(point_cloud, threshold):
    """Computes Betti-0 and Total H1 Persistence using ripser."""
    result = ripser.ripser(point_cloud, maxdim=1)
    dgms = result['dgms']
    
    # Betti-0 count: number of components alive past threshold
    h0_dgms = dgms[0]
    betti_0 = sum(1 for birth, death in h0_dgms if death > threshold)
    
    # Total H1 persistence: sum of finite lifespans of 1D cycles
    h1_dgms = dgms[1]
    total_h1_persistence = 0.0
    for birth, death in h1_dgms:
        if np.isfinite(death):
            total_h1_persistence += (death - birth)
            
    return betti_0, total_h1_persistence, h1_dgms

def plot_h1_barcodes(h1_dgms, title, save_path):
    """Plots the persistence barcode for the 1D cycles."""
    plt.figure(figsize=(9, 4.5))
    
    finite_cycles = [c for c in h1_dgms if np.isfinite(c[1])]
    # Sort cycles by lifetime in descending order for clean presentation
    finite_cycles = sorted(finite_cycles, key=lambda c: c[1] - c[0], reverse=True)
    
    if len(finite_cycles) == 0:
        plt.text(0.5, 0.5, "No finite H1 cycles detected", ha='center', va='center', fontsize=12)
    else:
        for i, (birth, death) in enumerate(finite_cycles):
            plt.plot([birth, death], [i, i], color='royalblue', lw=2.5)
            
    plt.yticks([])
    plt.xlabel("Filtration Parameter (Euclidean Distance)", fontsize=11)
    plt.ylabel("H1 Topological Cycles", fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    # Setup dry-run flag
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("Running in DRY-RUN mode (K=2, 3 angles).", flush=True)
        K = 2
        angles_deg = [0, 120, 200]
    else:
        print("Running in FULL mode (K=30, 10 angles).", flush=True)
        K = 30
        angles_deg = list(range(0, 361, 40))
        
    token = hf_token_or_none()
    if not token:
        print("HF_TOKEN missing!", flush=True)
        return 1

    print("Loading model and tokenizer...", flush=True)
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

    # Use 8 prompts for extracting knowledge direction
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    k_subset = [known[i] for i in subset_indices]
    u_subset = [unknown[i] for i in subset_indices]

    # Pre-tokenize
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

    # Capture baseline activations on layer 12
    print("Capturing baseline activations on layer 12...", flush=True)
    k_res = []
    u_res = []

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

    # Calculate v1 (Primary knowledge direction)
    v1 = diff_of_means_direction(k_res, u_res).to(DEVICE).to(torch.bfloat16)

    # Generate K orthogonal random control vectors v2_k
    print(f"Generating {K} orthogonal random vectors...", flush=True)
    v2_rands = []
    g = torch.Generator(device=DEVICE).manual_seed(42)
    for k in range(K):
        w_rand = torch.randn(v1.shape, generator=g, device=DEVICE, dtype=v1.dtype)
        v2_rand = w_rand - torch.dot(w_rand, v1) * v1
        v2_rand = v2_rand / (v2_rand.norm() + 1e-8)
        v2_rands.append(v2_rand)

    # We run the TDA analysis on prompt 0: "The capital of France is"
    test_idx = 0
    it_test = k_subset[test_idx]
    prompt_text = it_test["prompt"]
    print(f"Running sweep on prompt: '{prompt_text}'", flush=True)
    
    p_test = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                     tokenize=False, add_generation_prompt=True) + prompt_text
    test_inputs = tok(p_test, return_tensors="pt").to(DEVICE)
    
    # 1. Clean Baseline Generation & Point Cloud Capture
    print("Capturing baseline generation point cloud and logits...", flush=True)
    baseline_sink = []
    with capture_activations(layers[13], baseline_sink):
        with torch.no_grad():
            baseline_outputs = model.generate(
                **test_inputs,
                max_new_tokens=40,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True
            )
            
    baseline_text = tok.decode(baseline_outputs.sequences[0][test_inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    # Extract the very first token's logits [vocab_size]
    baseline_first_logits = baseline_outputs.logits[0][0].detach().float().cpu()
    
    # Concatenate the [batch, 1, d_model] tensors along dim=1 and squeeze to form [seq_len, d_model] point cloud
    baseline_point_cloud = torch.cat(baseline_sink, dim=1).squeeze(0).float().numpy()
    
    print(f"Baseline Text: '{baseline_text}'", flush=True)
    print(f"Baseline Point Cloud shape: {baseline_point_cloud.shape}", flush=True)
    
    # Compute baseline pairwise Euclidean distance matrix for Betti-0 thresholding
    dists_matrix = squareform(pdist(baseline_point_cloud, metric='euclidean'))
    mean_baseline_dist = float(np.mean(dists_matrix[dists_matrix > 0]))
    betti_0_threshold = 0.5 * mean_baseline_dist
    print(f"Mean baseline pairwise distance: {mean_baseline_dist:.4f}", flush=True)
    print(f"Betti-0 threshold set to: {betti_0_threshold:.4f}", flush=True)

    # Save baseline Betti-0 and H1 metrics
    b0_base, h1_base, _ = compute_tda_metrics(baseline_point_cloud, betti_0_threshold)
    print(f"Baseline: Betti-0 = {b0_base}, Total H1 Persistence = {h1_base:.4f}", flush=True)

    # Setup grid search parameters
    R_steer = 15000.0
    grid_results = []
    
    # We will save the barcodes for K=0 across all angles to illustrate the sweep
    visual_k = 0

    print("Starting grid sweep...", flush=True)
    for k_idx in range(K):
        v2_k = v2_rands[k_idx]
        for theta_deg in angles_deg:
            theta_rad = math.radians(theta_deg)
            
            # Capture activations & logits under intervention
            steered_sink = []
            with rotational_intervention(layers[TARGET_LAYER], v1, v2_k, theta_rad, R_steer):
                with capture_activations(layers[13], steered_sink):
                    with torch.no_grad():
                        steered_outputs = model.generate(
                            **test_inputs,
                            max_new_tokens=40,
                            do_sample=False,
                            use_cache=True,
                            return_dict_in_generate=True,
                            output_logits=True
                        )
            
            # Decode output text
            steered_text = tok.decode(steered_outputs.sequences[0][test_inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            # Extract the very first token's logits [vocab_size]
            steered_first_logits = steered_outputs.logits[0][0].detach().float().cpu()
            
            # Concatenate [batch, 1, d_model] tensors along dim=1 and squeeze to form [seq_len, d_model] point cloud
            steered_point_cloud = torch.cat(steered_sink, dim=1).squeeze(0).float().numpy()
            
            # Compute KL divergence ONLY on the very first generated token's logits
            kl_div = compute_kl_divergence(baseline_first_logits, steered_first_logits)
            
            # Compute topological metrics
            b0, h1_pers, h1_dgms = compute_tda_metrics(steered_point_cloud, betti_0_threshold)
            
            # Save results
            grid_results.append({
                "theta_deg": float(theta_deg),
                "R": R_steer,
                "k_index": k_idx,
                "betti_0": int(b0),
                "total_h1_persistence": float(h1_pers),
                "kl_divergence": float(kl_div),
                "output_text": steered_text
            })
            
            print(f"  [K={k_idx:02d}] Theta={theta_deg:3d}° | Betti-0={b0} | H1 Pers={h1_pers:.4f} | KL={kl_div:.4f} | Text: '{steered_text[:40]}...'", flush=True)
            
            # Plot persistence barcode for visual K
            if k_idx == visual_k:
                title = f"Gemma 3 4b-it H1 Barcode | Theta = {theta_deg}° | R = {R_steer:.0f}"
                save_path = f"docs/antigr_reports/h1_barcodes_theta_{theta_deg}.png"
                plot_h1_barcodes(h1_dgms, title, save_path)
            
            # Release cached memory on MPS to prevent OOM
            if DEVICE == "mps":
                torch.mps.empty_cache()
                
        # Incremental Save to prevent progress loss on crash/interruption
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "topology_sweep_v2_singleshot.json"
        out_json.write_text(json.dumps(grid_results, indent=2))
        
        artifact_json = Path("/Users/moraes/.gemini/antigravity/brain/e7cc9ba5-760c-4701-b604-315a148d9942/results/topology_sweep_v2_singleshot.json")
        artifact_json.parent.mkdir(parents=True, exist_ok=True)
        artifact_json.write_text(json.dumps(grid_results, indent=2))
        
    print(f"\nSaved final grid results to {out_json}", flush=True)
    print(f"Saved final duplicate to artifacts folder: {artifact_json}", flush=True)

    return 0

if __name__ == "__main__":
    sys.exit(main())
