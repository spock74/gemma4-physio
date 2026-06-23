"""Topological Analysis and Advanced Controls on Gemma 3 4b-it.
Implements:
1. Persistent Homology (H0 birth/death merges and Betti-0 complexity over the sweep)
2. Norm-Preserving Householder Reflection Steering
3. Completely Random Steering Baseline
4. Layer-wise Ablation Sweep (Layer 2 vs. Layer 12 vs. Layer 30)
"""

import sys
import os
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Any
from scipy.sparse.csgraph import minimum_spanning_tree

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

def kl(p_logits, q_logits):
    logp = torch.log_softmax(p_logits, -1)
    logq = torch.log_softmax(q_logits, -1)
    p = logp.exp()
    return float((p * (logp - logq)).sum())

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
def householder_intervention(layer: nn.Module, v1: torch.Tensor, v2: torch.Tensor, theta: float, positions: str = "last"):
    """Hooks target layer to perform a norm-preserving Householder reflection.
    Reflects the subspace projection of the activation vector onto the rotated direction
    v(theta) = cos(theta)*v1 + sin(theta)*v2.
    """
    def make_hook():
        def reflect_slice(x):
            d1 = v1.to(x.device, x.dtype)
            d2 = v2.to(x.device, x.dtype)
            
            # Subspace projection components
            c1 = (x @ d1).unsqueeze(-1)
            c2 = (x @ d2).unsqueeze(-1)
            x_perp = x - c1 * d1 - c2 * d2
            
            # Subspace vector before reflection
            v_sub = c1 * d1 + c2 * d2
            v_sub_norm = v_sub.norm(dim=-1, keepdim=True) + 1e-8
            
            # Target direction vector
            v_target = math.cos(theta) * d1 + math.sin(theta) * d2
            
            # Bisector vector for Householder reflection
            # w = (u - v) / ||u - v|| where u is the current direction and v is the target direction
            u = v_sub / v_sub_norm
            w = u - v_target
            w_norm = w.norm(dim=-1, keepdim=True) + 1e-8
            w_unit = w / w_norm
            
            # Reflect only the subspace component to preserve its norm exactly
            v_reflected = v_sub - 2 * (v_sub @ w_unit.transpose(-1, -2)) * w_unit
            return x_perp + v_reflected

        def hook(_m, _i, output):
            if isinstance(output, tuple):
                h = output[0]
                return (_apply_at(h, reflect_slice, positions), *output[1:])
            return _apply_at(output, reflect_slice, positions)
        return hook

    handle = layer.register_forward_hook(make_hook())
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def additive_intervention(layer: nn.Module, steer_vec: torch.Tensor, positions: str = "last"):
    """Hooks target layer to perform additive steering along a single vector."""
    def make_hook():
        def steer_slice(x):
            sv = steer_vec.to(x.device, x.dtype)
            return x + sv

        def hook(_m, _i, output):
            if isinstance(output, tuple):
                h = output[0]
                return (_apply_at(h, steer_slice, positions), *output[1:])
            return _apply_at(output, steer_slice, positions)
        return hook

    handle = layer.register_forward_hook(make_hook())
    try:
        yield
    finally:
        handle.remove()

def compute_h0_persistence(dist_matrix: np.ndarray):
    """Computes the 0-dimensional persistent homology from a distance matrix
    using the Minimum Spanning Tree (MST) equivalence to single-linkage clustering.
    Returns list of (birth, death) pairs.
    """
    n = dist_matrix.shape[0]
    mst = minimum_spanning_tree(dist_matrix).toarray()
    merge_distances = np.sort(mst[mst > 0])
    
    # Each merge represents the death of a component
    # All components are born at 0.0
    pairs = [(0.0, float(d)) for d in merge_distances]
    # One component survives forever (death = infinity)
    pairs.append((0.0, float('inf')))
    return pairs

def get_betti_0(pairs, threshold: float) -> int:
    """Returns the Betti-0 number (number of active connected components) at a given filtration scale."""
    active = 0
    for birth, death in pairs:
        if birth <= threshold < death:
            active += 1
    return active

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

    # Use 8 prompts for subset evaluation
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    k_subset = [known[i] for i in subset_indices]
    u_subset = [unknown[i] for i in subset_indices]

    # Pre-tokenize
    k_inputs = []
    for it in k_subset:
        p = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                    tokenize=False, add_generation_prompt=True) + it["prompt"]
        inputs = tok(p, return_tensors="pt").to(DEVICE)
        gold = int(tok(" " + it["answer"].strip(), add_special_tokens=False)["input_ids"][0])
        k_inputs.append((inputs, gold, it["prompt"]))

    u_inputs = []
    for it in u_subset:
        p = tok.apply_chat_template([{"role": "user", "content": RECALL_INSTRUCTION}],
                                    tokenize=False, add_generation_prompt=True) + it["prompt"]
        inputs = tok(p, return_tensors="pt").to(DEVICE)
        u_inputs.append(inputs)

    # Capture baseline residuals at layer 12
    print("Capturing baseline activations on layer 12...")
    k_res = []
    u_res = []

    def make_capture_hook(res_list):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            res_list.append(h[0, -1, :].detach().float().cpu())
        return hook

    for inputs, _, _ in k_inputs:
        h = layers[12].register_forward_hook(make_capture_hook(k_res))
        with torch.no_grad():
            model(**inputs)
        h.remove()

    for inputs in u_inputs:
        h = layers[12].register_forward_hook(make_capture_hook(u_res))
        with torch.no_grad():
            model(**inputs)
        h.remove()

    # Calculate v1 and v2_semantic
    v1 = diff_of_means_direction(k_res, u_res).to(DEVICE).to(torch.bfloat16)
    u_mean = torch.stack(k_res).mean(0).to(DEVICE).to(torch.bfloat16)
    v2_sem = u_mean - torch.dot(u_mean, v1) * v1
    v2_sem = v2_sem / (v2_sem.norm() + 1e-8)

    # 1. Topological Sweep (H0 Persistence on Layer 13)
    print("\n--- 1. Running Topological Sweep (Layer 13) ---")
    topo_angles = list(range(0, 361, 40)) # 10 steps to run fast
    betti_numbers = []
    mean_pair_distances = []
    
    # Calculate baseline pairwise distances at Layer 13
    baseline_l13_acts = []
    def make_l13_capture(res_list):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            res_list.append(h[0, -1, :].detach().float().cpu().numpy())
        return hook

    for inputs, _, _ in k_inputs:
        h = layers[13].register_forward_hook(make_l13_capture(baseline_l13_acts))
        with torch.no_grad():
            model(**inputs)
        h.remove()
        
    baseline_l13_acts = np.array(baseline_l13_acts)
    # Compute pairwise Euclidean distance matrix
    baseline_dists = np.sqrt(np.sum((baseline_l13_acts[:, None, :] - baseline_l13_acts[None, :, :])**2, axis=-1))
    mean_baseline_dist = float(np.mean(baseline_dists[baseline_dists > 0]))
    # Filtration threshold set to 0.5 * mean_baseline_dist
    threshold = 0.5 * mean_baseline_dist
    print(f"Mean baseline pairwise distance at Layer 13: {mean_baseline_dist:.4f}")
    print(f"Filtration threshold set to: {threshold:.4f}")

    R_steer = 15000.0
    for theta_deg in topo_angles:
        theta_rad = math.radians(theta_deg)
        steer_vec = R_steer * (math.cos(theta_rad) * v1 + math.sin(theta_rad) * v2_sem)
        
        # Capture Layer 13 activations under Layer 12 intervention
        steered_acts = []
        for inputs, _, _ in k_inputs:
            h_interv = layers[12].register_forward_hook(make_capture_hook([])) # Dummy hook to allow context
            with additive_intervention(layers[12], steer_vec, positions="last"):
                h_l13 = layers[13].register_forward_hook(make_l13_capture(steered_acts))
                with torch.no_grad():
                    model(**inputs)
                h_l13.remove()
            h_interv.remove()
            
        steered_acts = np.array(steered_acts)
        dists = np.sqrt(np.sum((steered_acts[:, None, :] - steered_acts[None, :, :])**2, axis=-1))
        mean_dist = float(np.mean(dists[dists > 0]))
        mean_pair_distances.append(mean_dist)
        
        # Compute H0 persistence pairs
        pairs = compute_h0_persistence(dists)
        b0 = get_betti_0(pairs, threshold)
        betti_numbers.append(b0)
        print(f"  Angle {theta_deg:3d}°: Mean Distance = {mean_dist:8.2f}, Betti-0 (at scale {threshold:.1f}) = {b0}")

    # 2. Householder Norm-Preserving Rotation Sweep vs Additive Steering
    print("\n--- 2. Running Householder Reflection Sweep (Layer 12) ---")
    householder_probs = []
    householder_kls = []
    
    for theta_deg in topo_angles:
        theta_rad = math.radians(theta_deg)
        probs = []
        kl_divs = []
        
        for inputs, gold, _ in k_inputs:
            with householder_intervention(layers[12], v1, v2_sem, theta_rad, positions="last"):
                with torch.no_grad():
                    steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().float()
            steered_prob = float(torch.softmax(steered_logits, -1)[gold])
            probs.append(steered_prob)
            
        for inputs in u_inputs:
            with torch.no_grad():
                clean_out = model(**inputs)
            clean_logits = clean_out.logits[0, -1, :].detach().float()
            
            with householder_intervention(layers[12], v1, v2_sem, theta_rad, positions="last"):
                with torch.no_grad():
                    steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().float()
            kl_divs.append(kl(clean_logits, steered_logits))
            
        householder_probs.append(float(np.mean(probs)))
        householder_kls.append(float(np.mean(kl_divs)))
        print(f"  Angle {theta_deg:3d}°: Prob = {householder_probs[-1]:.4f}, KL = {householder_kls[-1]:.4f}")

    # 3. Completely Random Steering Baseline
    print("\n--- 3. Running Completely Random Steering Baseline ---")
    random_probs = []
    random_kls = []
    g_rand = torch.Generator(device=DEVICE).manual_seed(101)
    
    # Draw a completely random direction (not orthogonal to v1)
    w_rand_dir = torch.randn(v1.shape, generator=g_rand, device=DEVICE, dtype=v1.dtype)
    w_rand_dir = w_rand_dir / (w_rand_dir.norm() + 1e-8)
    
    for theta_deg in topo_angles:
        theta_rad = math.radians(theta_deg)
        # Random steering has no angular dependency on semantic direction; we steer along w_rand_dir
        # with magnitude R_steer * cos(theta) to project a 1D sweep along a random axis
        steer_vec_rand = R_steer * math.cos(theta_rad) * w_rand_dir
        
        probs = []
        kl_divs = []
        for inputs, gold, _ in k_inputs:
            with additive_intervention(layers[12], steer_vec_rand, positions="last"):
                with torch.no_grad():
                    steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().float()
            steered_prob = float(torch.softmax(steered_logits, -1)[gold])
            probs.append(steered_prob)
            
        for inputs in u_inputs:
            with torch.no_grad():
                clean_out = model(**inputs)
            clean_logits = clean_out.logits[0, -1, :].detach().float()
            
            with additive_intervention(layers[12], steer_vec_rand, positions="last"):
                with torch.no_grad():
                    steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().float()
            kl_divs.append(kl(clean_logits, steered_logits))
            
        random_probs.append(float(np.mean(probs)))
        random_kls.append(float(np.mean(kl_divs)))
        print(f"  Angle {theta_deg:3d}°: Prob = {random_probs[-1]:.4f}, KL = {random_kls[-1]:.4f}")

    # 4. Layer-wise Ablation (Layer 2 and Layer 30)
    print("\n--- 4. Running Layer-wise Ablation (Layer 2 vs. Layer 30) ---")
    layer_ablation_results = {}
    
    for ablation_layer in [2, 30]:
        print(f"  Ablating Layer {ablation_layer}...")
        l_res_k = []
        l_res_u = []
        
        # Capture residuals at target layer
        for inputs, _, _ in k_inputs:
            h = layers[ablation_layer].register_forward_hook(make_capture_hook(l_res_k))
            with torch.no_grad():
                model(**inputs)
            h.remove()

        for inputs in u_inputs:
            h = layers[ablation_layer].register_forward_hook(make_capture_hook(l_res_u))
            with torch.no_grad():
                model(**inputs)
            h.remove()
            
        l_v1 = diff_of_means_direction(l_res_k, l_res_u).to(DEVICE).to(torch.bfloat16)
        l_u_mean = torch.stack(l_res_k).mean(0).to(DEVICE).to(torch.bfloat16)
        l_v2_sem = l_u_mean - torch.dot(l_u_mean, l_v1) * l_v1
        l_v2_sem = l_v2_sem / (l_v2_sem.norm() + 1e-8)
        
        # Measure Layer norm average to scale R accordingly
        l_norm = float(torch.stack(l_res_k).mean(0).norm())
        l_R = 0.7 * l_norm
        print(f"    Layer {ablation_layer} average L2 norm: {l_norm:.2f}, R used: {l_R:.2f}")
        
        l_probs = []
        for theta_deg in topo_angles:
            theta_rad = math.radians(theta_deg)
            l_steer = l_R * (math.cos(theta_rad) * l_v1 + math.sin(theta_rad) * l_v2_sem)
            
            probs = []
            for inputs, gold, _ in k_inputs:
                with additive_intervention(layers[ablation_layer], l_steer, positions="last"):
                    with torch.no_grad():
                        steered_out = model(**inputs)
                steered_logits = steered_out.logits[0, -1, :].detach().float()
                steered_prob = float(torch.softmax(steered_logits, -1)[gold])
                probs.append(steered_prob)
            l_probs.append(float(np.mean(probs)))
            
        layer_ablation_results[str(ablation_layer)] = l_probs
        print(f"    Layer {ablation_layer} Probs: {[round(p, 4) for p in l_probs]}")

    # Save results
    results_payload = {
        "topo_angles": topo_angles,
        "betti_numbers": betti_numbers,
        "mean_pair_distances": mean_pair_distances,
        "householder_probs": householder_probs,
        "householder_kls": householder_kls,
        "random_probs": random_probs,
        "random_kls": random_kls,
        "layer_ablation": layer_ablation_results
    }
    
    # Save both to local workspace scratch and mirror to artifacts
    out_path = Path("scratch/topological_controls_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = json.dumps(results_payload, indent=2)
    out_path.write_text(out_payload)
    print(f"\nResults saved to {out_path}")
    
    artifacts_dir = Path("/Users/moraes/.gemini/antigravity/brain/e7cc9ba5-760c-4701-b604-315a148d9942")
    if artifacts_dir.exists():
        art_scratch = artifacts_dir / "scratch"
        art_scratch.mkdir(parents=True, exist_ok=True)
        (art_scratch / "topological_controls_results.json").write_text(out_payload)
        print("Also saved copy to Antigravity artifacts directory.")

    # Plot results
    print("Generating topological and control plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Betti-0 and mean distance (Topological Collapse)
    ax1 = axes[0, 0]
    ax1.plot(topo_angles, betti_numbers, 'o-', color='crimson', label='Betti-0 count')
    ax1.set_xlabel('Steering Angle (degrees)')
    ax1.set_ylabel('Betti-0 at scale epsilon', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.set_title('Topological Simplification (H0 Persistence) on L13')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(topo_angles, mean_pair_distances, 's--', color='navy', label='Mean Pair Distance')
    ax1_twin.set_ylabel('Mean Pairwise Euclidean Distance', color='navy')
    ax1_twin.tick_params(axis='y', labelcolor='navy')
    
    # Plot 2: Householder reflection (Norm-preserving) vs Additive
    ax2 = axes[0, 1]
    ax2.plot(topo_angles, householder_probs, 'o-', label='Householder Reflection (Norm-Preserving)', color='forestgreen')
    ax2.set_xlabel('Steering Angle (degrees)')
    ax2.set_ylabel('Target Answer Probability')
    ax2.set_title('Norm-Preserving Householder vs Additive Steering')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Random Steering Baseline
    ax3 = axes[1, 0]
    ax3.plot(topo_angles, random_probs, 'd-', label='Random Axis Steering', color='purple')
    ax3.set_xlabel('Steering Angle (degrees)')
    ax3.set_ylabel('Target Answer Probability')
    ax3.set_title('Completely Random Axis Steering Control')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Layer-wise Ablation (Layer 2 vs Layer 12 vs Layer 30)
    ax4 = axes[1, 1]
    ax4.plot(topo_angles, layer_ablation_results["2"], 's--', label='Layer 2 (Early Layer)', color='teal')
    ax4.plot(topo_angles, layer_ablation_results["30"], 'x--', label='Layer 30 (Late Layer)', color='goldenrod')
    ax4.set_xlabel('Steering Angle (degrees)')
    ax4.set_ylabel('Target Answer Probability')
    ax4.set_title('Layer-wise Ablation (Depth Analysis)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/topological_analysis.png", dpi=150)
    
    # Also save copy to the Antigravity artifacts directory
    artifacts_dir = Path("/Users/moraes/.gemini/antigravity/brain/e7cc9ba5-760c-4701-b604-315a148d9942")
    if artifacts_dir.exists():
        art_plots = artifacts_dir / "plots"
        art_plots.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(art_plots / "topological_analysis.png"), dpi=150)
        print("Also saved plot copy to Antigravity artifacts directory.")
        
    plt.close()
    print("Plots saved to plots/topological_analysis.png")
    return 0

if __name__ == "__main__":
    sys.exit(main())
