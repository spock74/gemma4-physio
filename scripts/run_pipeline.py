import sys
import os
import math
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Garantir visibilidade do pacote local
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gemma4_physio.config import PipelineConfig
from gemma4_physio.data_loader import load_and_stratify_popqa
from gemma4_physio.spps_hooks import spps_rotational_hook
from gemma4_physio.tda_engine import compute_phase_space_tda

def run_topological_sweep(config: PipelineConfig):
    sweep_cfg = config.pipelines.get("topological_sweep")
    if not sweep_cfg or not sweep_cfg.get("enabled"):
        print("Topological sweep pipeline disabled in configuration.", flush=True)
        return
        
    print(f"\n--- Starting Pipeline: {config.experiment_name} ---", flush=True)
    device = config.model_settings.device
    dtype = config.model_settings.torch_dtype
    
    # 1. Carregar Modelo e Tokenizer
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
    
    # 2. Carregar e estratificar dados do PopQA
    popqa_path = Path("data/popqa/popqa_subset.json")
    if not popqa_path.exists():
        print(f"Error: PopQA file missing at {popqa_path}", flush=True)
        return
    known_set, unknown_set = load_and_stratify_popqa(popqa_path)
    print(f"PopQA Data: {len(known_set)} Known, {len(unknown_set)} Unknown entities.", flush=True)
    
    # 3. Simular vetor de direção factual DoM (v1)
    # Em produção, você carregará o d_know.pt gerado pelo pipeline A.
    # Aqui inicializamos um vetor simulado na dimensão de embedding do Gemma para fins de pipeline.
    d_model = model.config.text_config.hidden_size if hasattr(model.config, 'text_config') else model.config.hidden_size
    v1 = torch.randn(d_model, device=device, dtype=dtype)
    v1 = v1 / (v1.norm() + 1e-8)
    
    # 4. Gerar K vetores ortogonais de controle via Gram-Schmidt
    K = sweep_cfg["control_planes_K"]
    angles = sweep_cfg["angles_deg"]
    R = sweep_cfg["magnitude_R"]
    layer_idx = sweep_cfg["layer_intervention"]
    capture_idx = sweep_cfg["layer_capture"]
    
    print(f"Generating {K} orthogonal control vectors...", flush=True)
    v2_rands = []
    g = torch.Generator(device=device).manual_seed(42)
    for _ in range(K):
        w_rand = torch.randn(d_model, generator=g, device=device, dtype=dtype)
        v2_rand = w_rand - torch.dot(w_rand, v1) * v1
        v2_rand = v2_rand / (v2_rand.norm() + 1e-8)
        v2_rands.append(v2_rand)
        
    # 5. Executar a Varredura Rotacional Coletando as Ativações do Primeiro Token
    # Vamos usar a primeira pergunta do PopQA conhecido como nosso prompt teste
    target_prompt = "The capital of France is"  # Exemplo de entrada
    inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
    
    phase_activations = []
    
    # Hook temporário para capturar o primeiro token na camada de captura (L13)
    def capture_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1: # Primeiro passo de decodificação
            phase_activations.append(h[0, 0, :].detach().float().cpu().numpy())
            
    layers = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    capture_handle = layers[capture_idx].register_forward_hook(capture_hook)
    
    print(f"Running sweep over {K} planes and {len(angles)} angles...", flush=True)
    for k in range(K):
        v2_k = v2_rands[k]
        for theta_deg in angles:
            theta_rad = math.radians(theta_deg)
            
            # Aplica a SPPS de fase única na Camada 12
            target_layer = layers[layer_idx]
            with spps_rotational_hook(target_layer, v1, v2_k, theta_rad, R):
                with torch.no_grad():
                    # Geramos exatamente 1 novo token para capturar seu estado
                    model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
                    
            if device == "mps":
                torch.mps.empty_cache()
                
    capture_handle.remove()
    
    # 6. Processar a Nuvem de Pontos de Fase no Motor TDA
    point_cloud = np.array(phase_activations) # [K * len(angles), d_model]
    print(f"\nPhase Point Cloud generated with shape: {point_cloud.shape}", flush=True)
    
    ratio = sweep_cfg["tda_config"]["betti_0_threshold_ratio"]
    betti_0, h1_persistence, _ = compute_phase_space_tda(point_cloud, threshold_ratio=ratio)
    
    # 7. Salvar e Logar Métricas
    results = {
        "experiment": config.experiment_name,
        "betti_0": betti_0,
        "total_h1_persistence": h1_persistence,
        "point_cloud_dimensions": list(point_cloud.shape)
    }
    
    out_path = Path(sweep_cfg["tda_config"]["output_json_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n--- Pipeline Completed Successfully ---", flush=True)
    print(f"Results: Betti-0 = {betti_0} | H1 Persistence = {h1_persistence:.4f}", flush=True)
    print(f"JSON metrics written to: {out_path}", flush=True)

def main():
    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    
    config = PipelineConfig.load_from_yaml(config_path)
    run_topological_sweep(config)

if __name__ == "__main__":
    main()
