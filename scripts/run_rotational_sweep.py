"""2D Rotational Sweep: Attractor Stability and Phase Transitions under Orthogonal Perturbations on Gemma 3 4b-it.
Measures the stability of local attractors under extreme orthogonal perturbations in Layer 12
using two methods:
1. Pure Subspace Rotation (SO(2) rotation of natural activations, scale-free)
2. High-Scale Additive Steering (R = 10000, 20000, matching the 21k activation norm)
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
def subspace_intervention(layer: nn.Module, direction1: torch.Tensor, direction2: torch.Tensor, theta: float, R: float | None = None, positions: str = "last"):
    """Hooks target layer to either:
    - Rotate hidden states within the 2D plane by theta (if R is None)
    - Perform high-scale additive steering by R * (cos(theta)*v1 + sin(theta)*v2) (if R is not None)
    """
    def make_hook():
        def rotate_slice(x):
            d1 = direction1.to(x.device, x.dtype)
            d2 = direction2.to(x.device, x.dtype)
            
            if R is None:
                # 1. Pure Subspace Rotation
                c1 = (x @ d1).unsqueeze(-1)
                c2 = (x @ d2).unsqueeze(-1)
                x_perp = x - c1 * d1 - c2 * d2
                c1_rot = c1 * math.cos(theta) - c2 * math.sin(theta)
                c2_rot = c1 * math.sin(theta) + c2 * math.cos(theta)
                return x_perp + c1_rot * d1 + c2_rot * d2
            else:
                # 2. Additive Steering
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

    # Use a representative subset of 8 prompts
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

    # 1. Capture residuals at layer TARGET_LAYER to form 2D rotation plane
    print(f"Capturing activations on layer {TARGET_LAYER}...")
    k_res = []
    u_res = []

    def make_capture_hook(res_list):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            res_list.append(h[0, -1, :].detach().float().cpu())
        return hook

    # Capture Known
    for inputs, _, _ in k_inputs:
        h = layers[TARGET_LAYER].register_forward_hook(make_capture_hook(k_res))
        with torch.no_grad():
            model(**inputs)
        h.remove()

    # Capture Unknown
    for inputs in u_inputs:
        h = layers[TARGET_LAYER].register_forward_hook(make_capture_hook(u_res))
        with torch.no_grad():
            model(**inputs)
        h.remove()

    # Calculate v1 (Primary direction of factual knowledge)
    print("Calculating orthogonal bases...")
    v1 = diff_of_means_direction(k_res, u_res).to(DEVICE).to(torch.bfloat16)

    # Calculate v2_semantic (Known baseline context orthogonal to v1)
    u_mean = torch.stack(k_res).mean(0).to(DEVICE).to(torch.bfloat16)
    v2_sem = u_mean - torch.dot(u_mean, v1) * v1
    v2_sem = v2_sem / (v2_sem.norm() + 1e-8)

    # Calculate K random control vectors orthogonal to v1 (Robust baseline control)
    K = 5
    v2_rands = []
    g = torch.Generator(device=DEVICE).manual_seed(42)
    for k in range(K):
        w_rand = torch.randn(v1.shape, generator=g, device=DEVICE, dtype=v1.dtype)
        v2_rand = w_rand - torch.dot(w_rand, v1) * v1
        v2_rand = v2_rand / (v2_rand.norm() + 1e-8)
        v2_rands.append(v2_rand)

    print(f"  v1 norm: {float(v1.norm()):.4f}")
    print(f"  v2_semantic norm: {float(v2_sem.norm()):.4f}, dot(v1, v2_sem): {float(torch.dot(v1, v2_sem)):.2e}")
    for k in range(K):
        print(f"  v2_random[{k}] norm: {float(v2_rands[k].norm()):.4f}, dot(v1, v2_random[{k}]): {float(torch.dot(v1, v2_rands[k])):.2e}")

    # 2. Run the sweeps
    angles_deg = list(range(0, 361, 20))
    
    # Structure results
    results = {
        "rotation": {
            "semantic": [],
            "control": [[] for _ in range(K)]
        },
        "additive": {
            "semantic": {10000.0: [], 20000.0: []},
            "control": {
                10000.0: [[] for _ in range(K)],
                20000.0: [[] for _ in range(K)]
            }
        }
    }

    print("Running pure rotation sweep...")
    # Semantic mode
    print("  Mode: semantic")
    for theta_deg in angles_deg:
        theta_rad = math.radians(theta_deg)
        probs = []
        kl_divs = []
        
        for inputs, gold, _ in k_inputs:
            with subspace_intervention(layers[TARGET_LAYER], v1, v2_sem, theta_rad, R=None, positions="last"):
                with torch.no_grad():
                    steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().float()
            steered_prob = float(torch.softmax(steered_logits, -1)[gold])
            probs.append(steered_prob)
        
        for inputs in u_inputs:
            with torch.no_grad():
                clean_out = model(**inputs)
            clean_logits = clean_out.logits[0, -1, :].detach().float()
            
            with subspace_intervention(layers[TARGET_LAYER], v1, v2_sem, theta_rad, R=None, positions="last"):
                with torch.no_grad():
                    steered_out = model(**inputs)
            steered_logits = steered_out.logits[0, -1, :].detach().float()
            kl_divs.append(kl(clean_logits, steered_logits))
            
        results["rotation"]["semantic"].append({
            "angle_deg": theta_deg,
            "mean_prob": float(np.mean(probs)),
            "mean_kl": float(np.mean(kl_divs))
        })

    # Control mode (multiple random orthogonal vectors)
    print("  Mode: control (baseline)")
    for k in range(K):
        print(f"    Random vector {k+1}/{K}")
        v2 = v2_rands[k]
        for theta_deg in angles_deg:
            theta_rad = math.radians(theta_deg)
            probs = []
            kl_divs = []
            
            for inputs, gold, _ in k_inputs:
                with subspace_intervention(layers[TARGET_LAYER], v1, v2, theta_rad, R=None, positions="last"):
                    with torch.no_grad():
                        steered_out = model(**inputs)
                steered_logits = steered_out.logits[0, -1, :].detach().float()
                steered_prob = float(torch.softmax(steered_logits, -1)[gold])
                probs.append(steered_prob)
            
            for inputs in u_inputs:
                with torch.no_grad():
                    clean_out = model(**inputs)
                clean_logits = clean_out.logits[0, -1, :].detach().float()
                
                with subspace_intervention(layers[TARGET_LAYER], v1, v2, theta_rad, R=None, positions="last"):
                    with torch.no_grad():
                        steered_out = model(**inputs)
                steered_logits = steered_out.logits[0, -1, :].detach().float()
                kl_divs.append(kl(clean_logits, steered_logits))
                
            results["rotation"]["control"][k].append({
                "angle_deg": theta_deg,
                "mean_prob": float(np.mean(probs)),
                "mean_kl": float(np.mean(kl_divs))
            })

    print("Running high-scale additive steering sweep...")
    # Semantic mode
    print("  Mode: semantic")
    for R in [10000.0, 20000.0]:
        print(f"    R = {R:.1f}")
        for theta_deg in angles_deg:
            theta_rad = math.radians(theta_deg)
            probs = []
            kl_divs = []
            
            for inputs, gold, _ in k_inputs:
                with subspace_intervention(layers[TARGET_LAYER], v1, v2_sem, theta_rad, R=R, positions="last"):
                    with torch.no_grad():
                        steered_out = model(**inputs)
                steered_logits = steered_out.logits[0, -1, :].detach().float()
                steered_prob = float(torch.softmax(steered_logits, -1)[gold])
                probs.append(steered_prob)
            
            for inputs in u_inputs:
                with torch.no_grad():
                    clean_out = model(**inputs)
                clean_logits = clean_out.logits[0, -1, :].detach().float()
                
                with subspace_intervention(layers[TARGET_LAYER], v1, v2_sem, theta_rad, R=R, positions="last"):
                    with torch.no_grad():
                        steered_out = model(**inputs)
                steered_logits = steered_out.logits[0, -1, :].detach().float()
                kl_divs.append(kl(clean_logits, steered_logits))
                
            results["additive"]["semantic"][R].append({
                "angle_deg": theta_deg,
                "mean_prob": float(np.mean(probs)),
                "mean_kl": float(np.mean(kl_divs))
            })

    # Control mode
    print("  Mode: control (baseline)")
    for R in [10000.0, 20000.0]:
        print(f"    R = {R:.1f}")
        for k in range(K):
            print(f"      Random vector {k+1}/{K}")
            v2 = v2_rands[k]
            for theta_deg in angles_deg:
                theta_rad = math.radians(theta_deg)
                probs = []
                kl_divs = []
                
                for inputs, gold, _ in k_inputs:
                    with subspace_intervention(layers[TARGET_LAYER], v1, v2, theta_rad, R=R, positions="last"):
                        with torch.no_grad():
                            steered_out = model(**inputs)
                    steered_logits = steered_out.logits[0, -1, :].detach().float()
                    steered_prob = float(torch.softmax(steered_logits, -1)[gold])
                    probs.append(steered_prob)
                
                for inputs in u_inputs:
                    with torch.no_grad():
                        clean_out = model(**inputs)
                    clean_logits = clean_out.logits[0, -1, :].detach().float()
                    
                    with subspace_intervention(layers[TARGET_LAYER], v1, v2, theta_rad, R=R, positions="last"):
                        with torch.no_grad():
                            steered_out = model(**inputs)
                    steered_logits = steered_out.logits[0, -1, :].detach().float()
                    kl_divs.append(kl(clean_logits, steered_logits))
                    
                results["additive"]["control"][R][k].append({
                    "angle_deg": theta_deg,
                    "mean_prob": float(np.mean(probs)),
                    "mean_kl": float(np.mean(kl_divs))
                })

    # Save raw results
    out_dir = Path("/Users/moraes/.gemini/antigravity/brain/e7cc9ba5-760c-4701-b604-315a148d9942/scratch")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "rotational_sweep_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved raw results to {json_path}")

    # 3. Generate beautiful polar plots
    print("Generating polar plots...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Helvetica', 'Arial']
    plt.rcParams['text.color'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#1e293b'

    # Create 2x2 grid of polar subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 11), subplot_kw={'projection': 'polar'}, dpi=300)
    
    theta_rad = np.radians(angles_deg)

    # Subplot [0, 0]: Pure Rotation Probability
    ax = axs[0, 0]
    ax.set_title("Probabilidade do Alvo ($P_{target}$)\nRotação de Subespaço Pura (SO(2))", fontsize=11, fontweight='bold', pad=15)
    ax.plot(theta_rad, [pt["mean_prob"] for pt in results["rotation"]["semantic"]], color='#10b981', linewidth=2.5, label="Semântico v2", marker='o', markersize=4)
    ax.fill(theta_rad, [pt["mean_prob"] for pt in results["rotation"]["semantic"]], color='#10b981', alpha=0.1)
    
    # Process multiple controls
    control_probs_rot = np.array([[pt["mean_prob"] for pt in run] for run in results["rotation"]["control"]])
    mean_ctrl_probs_rot = np.mean(control_probs_rot, axis=0)
    std_ctrl_probs_rot = np.std(control_probs_rot, axis=0)
    
    ax.plot(theta_rad, mean_ctrl_probs_rot, color='#64748b', linewidth=2, label="Controle v2 (Média)", marker='x', markersize=4, linestyle=':')
    ax.fill_between(theta_rad, np.clip(mean_ctrl_probs_rot - std_ctrl_probs_rot, 0, 1), np.clip(mean_ctrl_probs_rot + std_ctrl_probs_rot, 0, 1), color='#64748b', alpha=0.15, label="Controle v2 (±1 DP)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', facecolor='white', edgecolor='#e2e8f0', fontsize=8)

    # Subplot [0, 1]: Additive Steering Probability
    ax = axs[0, 1]
    ax.set_title("Probabilidade do Alvo ($P_{target}$)\nDirecionamento Aditivo de Larga Escala", fontsize=11, fontweight='bold', pad=15)
    colors_add = {10000.0: '#3b82f6', 20000.0: '#ef4444'}
    for R in [10000.0, 20000.0]:
        ax.plot(theta_rad, [pt["mean_prob"] for pt in results["additive"]["semantic"][R]], color=colors_add[R], linewidth=2, label=f"Semântico R={int(R/1000)}k", marker='o', markersize=3)
        
        control_probs_add = np.array([[pt["mean_prob"] for pt in run] for run in results["additive"]["control"][R]])
        mean_ctrl_probs_add = np.mean(control_probs_add, axis=0)
        std_ctrl_probs_add = np.std(control_probs_add, axis=0)
        
        ax.plot(theta_rad, mean_ctrl_probs_add, color=colors_add[R], linewidth=1.5, label=f"Controle R={int(R/1000)}k (Média)", linestyle=':', marker='x', markersize=3)
        ax.fill_between(theta_rad, np.clip(mean_ctrl_probs_add - std_ctrl_probs_add, 0, 1), np.clip(mean_ctrl_probs_add + std_ctrl_probs_add, 0, 1), color=colors_add[R], alpha=0.08)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', facecolor='white', edgecolor='#e2e8f0', fontsize=8)

    # Subplot [1, 0]: Pure Rotation KL Divergence
    ax = axs[1, 0]
    ax.set_title("Divergência KL ($D_{KL}$)\nRotação de Subespaço Pura (SO(2))", fontsize=11, fontweight='bold', pad=15)
    ax.plot(theta_rad, [pt["mean_kl"] for pt in results["rotation"]["semantic"]], color='#10b981', linewidth=2.5, label="Semântico v2", marker='o', markersize=4)
    ax.fill(theta_rad, [pt["mean_kl"] for pt in results["rotation"]["semantic"]], color='#10b981', alpha=0.1)
    
    control_kls_rot = np.array([[pt["mean_kl"] for pt in run] for run in results["rotation"]["control"]])
    mean_ctrl_kls_rot = np.mean(control_kls_rot, axis=0)
    std_ctrl_kls_rot = np.std(control_kls_rot, axis=0)
    
    ax.plot(theta_rad, mean_ctrl_kls_rot, color='#64748b', linewidth=2, label="Controle v2 (Média)", marker='x', markersize=4, linestyle=':')
    ax.fill_between(theta_rad, np.maximum(mean_ctrl_kls_rot - std_ctrl_kls_rot, 0), mean_ctrl_kls_rot + std_ctrl_kls_rot, color='#64748b', alpha=0.15, label="Controle v2 (±1 DP)")
    ax.legend(loc='lower right', facecolor='white', edgecolor='#e2e8f0', fontsize=8)

    # Subplot [1, 1]: Additive Steering KL Divergence
    ax = axs[1, 1]
    ax.set_title("Divergência KL ($D_{KL}$)\nDirecionamento Aditivo de Larga Escala", fontsize=11, fontweight='bold', pad=15)
    for R in [10000.0, 20000.0]:
        ax.plot(theta_rad, [pt["mean_kl"] for pt in results["additive"]["semantic"][R]], color=colors_add[R], linewidth=2, label=f"Semântico R={int(R/1000)}k", marker='o', markersize=3)
        
        control_kls_add = np.array([[pt["mean_kl"] for pt in run] for run in results["additive"]["control"][R]])
        mean_ctrl_kls_add = np.mean(control_kls_add, axis=0)
        std_ctrl_kls_add = np.std(control_kls_add, axis=0)
        
        ax.plot(theta_rad, mean_ctrl_kls_add, color=colors_add[R], linewidth=1.5, label=f"Controle R={int(R/1000)}k (Média)", linestyle=':', marker='x', markersize=3)
        ax.fill_between(theta_rad, np.maximum(mean_ctrl_kls_add - std_ctrl_kls_add, 0), mean_ctrl_kls_add + std_ctrl_kls_add, color=colors_add[R], alpha=0.08)
    ax.legend(loc='lower right', facecolor='white', edgecolor='#e2e8f0', fontsize=8)

    # Style polar grids
    for ax in axs.flat:
        ax.set_xticks(np.radians(list(range(0, 360, 45))))
        ax.set_xticklabels([f"{d}°" for d in range(0, 360, 45)], fontsize=9)
        ax.grid(True, linestyle=':', color='#cbd5e1')
        ax.set_rlabel_position(45)

    plt.suptitle("Varredura Rotacional 2D na Camada L12 do Gemma 3\nEstabilidade de Atratores e Controle de Linha de Base Ortogonal", 
                 fontsize=14, fontweight='bold', color='#0f172a', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plots
    plot_path1 = Path("/Users/moraes/.gemini/antigravity/brain/e7cc9ba5-760c-4701-b604-315a148d9942/plots/rotational_sweep.png")
    plot_path1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path1, dpi=300)
    
    plot_path2 = Path("/Users/moraes/Documents/PROJETOS/interpretability/started-june-26/zero/docs/figures/fig7_rotational_sweep.png")
    plot_path2.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path2, dpi=300)
    
    plt.close()
    
    print(f"Saved polar plot to:\n  - {plot_path1}\n  - {plot_path2}")
    return 0

if __name__ == "__main__":
    from gemma4_lab import observability
    observability.setup()
    sys.exit(main())
