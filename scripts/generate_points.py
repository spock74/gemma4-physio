import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gemma4_physio.observability import setup_logfire

setup_logfire()

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Garantir visibilidade do pacote local
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gemma4_physio.config import PipelineConfig
from gemma4_physio.data_loader import load_and_stratify_popqa
from gemma4_physio.spps_hooks import spps_rotational_hook

def generate_plane(config: PipelineConfig, plane_idx: int):
    sweep_cfg = config.pipelines.get("topological_sweep")
    if not sweep_cfg or not sweep_cfg.get("enabled"):
        print("Topological sweep pipeline disabled in configuration.", flush=True)
        return
        
    K = sweep_cfg["control_planes_K"]
    if plane_idx < 0 or plane_idx >= K:
        print(f"Error: Plane index {plane_idx} is out of bounds (0 to {K-1})")
        return

    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_file = checkpoint_dir / f"plane_{plane_idx}.npy"
    
    if out_file.exists():
        print(f"Skipping plane {plane_idx}: {out_file} already exists.", flush=True)
        return

    print(f"\n--- Generating points for Plane {plane_idx}/{K-1} ---", flush=True)
    device = config.model_settings.device
    dtype = config.model_settings.torch_dtype
    
    print("Loading model and tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_settings.model_id, 
        cache_dir=config.model_settings.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_settings.model_id,
        cache_dir=config.model_settings.cache_dir,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=config.model_settings.attn_implementation
    ).eval()
    
    d_model = model.config.text_config.hidden_size if hasattr(model.config, 'text_config') else model.config.hidden_size
    
    # CRÍTICO: Fixar a semente ANTES de gerar v1 para que TODOS os subprocessos
    # (plane_0, plane_1, etc.) gerem o exato mesmo vetor v1 e a mesma base v2!
    torch.manual_seed(42)
    v1 = torch.randn(d_model, device=device, dtype=dtype)
    v1 = v1 / (v1.norm() + 1e-8)
    
    # Gerar vetores K com a mesma semente garante consistência entre processos
    g = torch.Generator(device=device).manual_seed(42)
    v2_rands = []
    for _ in range(K):
        w_rand = torch.randn(d_model, generator=g, device=device, dtype=dtype)
        v2_rand = w_rand - torch.dot(w_rand, v1) * v1
        v2_rand = v2_rand / (v2_rand.norm() + 1e-8)
        v2_rands.append(v2_rand)
        
    v2_k = v2_rands[plane_idx]
    
    angles = sweep_cfg["angles_deg"]
    R = sweep_cfg["magnitude_R"]
    layer_idx = sweep_cfg["layer_intervention"]
    capture_idx = sweep_cfg["layer_capture"]
    
    target_prompt = "The capital of France is"
    inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
    
    phase_activations = []
    
    def capture_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:
            phase_activations.append(h[0, 0, :].detach().float().cpu().numpy())
            
    layers = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    capture_handle = layers[capture_idx].register_forward_hook(capture_hook)
    
    print(f"Running sweep over plane {plane_idx} with {len(angles)} angles...", flush=True)
    for theta_deg in angles:
        theta_rad = math.radians(theta_deg)
        target_layer = layers[layer_idx]
        with spps_rotational_hook(target_layer, v1, v2_k, theta_rad, R):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
                
        if device == "mps":
            torch.mps.empty_cache()
            
    capture_handle.remove()
    
    point_cloud = np.array(phase_activations)
    print(f"Saving checkpoint to {out_file}", flush=True)
    np.save(out_file, point_cloud)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plane", type=int, required=True, help="Index of the control plane to run")
    args = parser.parse_args()

    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    
    config = PipelineConfig.load_from_yaml(config_path)
    generate_plane(config, args.plane)

if __name__ == "__main__":
    main()
