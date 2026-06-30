"""
================================================================================
Copyright (c) 2026 Jose E Moraes. All rights reserved.

NOME CANÔNICO: Estabilização de Freeman (Repair Heads Topology)
INTENÇÃO: Monitorar a persistência topológica nas cabeças de reparo (camadas superiores) em resposta a perturbações polares.
DATASET: PopQA
DATA DE CRIAÇÃO: Junho de 2026
ÚLTIMA MODIFICAÇÃO: 30 de Junho de 2026
MUDANÇA PRINCIPAL: Adição de barras de progresso (tqdm) e logs coloridos (typer).
================================================================================
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ripser import ripser
from persim import plot_diagrams
from tqdm import tqdm
import typer
from persim import plot_diagrams

from gemma4_physio.data_loader import PopQASampler
from gemma4_physio.spps_hooks import spps_rotational_hook

def run_freeman_stabilization(config_dict: dict, model, tokenizer, device: str):
    start_time = time.time()
    typer.secho("\n" + "="*50, fg=typer.colors.MAGENTA, bold=True)
    typer.secho("🚀 INICIANDO PIPELINE: ESTABILIZAÇÃO DE FREEMAN", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("="*50 + "\n", fg=typer.colors.MAGENTA, bold=True)
    
    json_path = Path("data/popqa/popqa_full.json")
    sampler = PopQASampler(json_path)
    
    classes = config_dict["target_classes"]
    seed = config_dict.get("seed", 42)
    layer_intervention = config_dict.get("layer_intervention", 12)
    l_start = config_dict.get("layer_capture_start", 20)
    l_end = config_dict.get("layer_capture_end", 26)
    
    data = sampler.sample_5x5_representatives(classes, seed=seed)
    items = [item for c_items in data.values() for item in c_items]
    
    dtype = model.dtype
    layers = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    
    d_model = model.config.text_config.hidden_size if hasattr(model.config, 'text_config') else model.config.hidden_size
    R = config_dict.get("magnitude_R", 15000.0)
    
    contrast_path = Path("data/eval/entity_knowledge_contrast.json")
    if contrast_path.exists():
        import json
        with open(contrast_path, "r") as f:
            contrast_data = json.load(f)
        known_items = contrast_data["known"][:15]
        unknown_items = contrast_data["unknown"][:15]
        known_prompts = [f"Answer the following question succinctly.\nQuestion: {it['prompt']}?\nAnswer:" for it in known_items]
        unknown_prompts = [f"Answer the following question succinctly.\nQuestion: {it['prompt']}?\nAnswer:" for it in unknown_items]
        
        from gemma4_physio.directions import extract_difference_of_means
        typer.secho("🔹 Extraindo direção factual real via Difference-of-Means...", fg=typer.colors.CYAN)
        v1 = extract_difference_of_means(model, tokenizer, known_prompts, unknown_prompts, layers[layer_intervention], device)
        
        # Gera v2 ortogonal a v1 usando a semente especificada
        torch.manual_seed(seed)
        w_rand = torch.randn(d_model, device=device, dtype=dtype)
        proj = torch.dot(w_rand, v1) * v1
        v2 = w_rand - proj
        v2 = v2 / (v2.norm() + 1e-8)
    else:
        typer.secho("⚠️ Dataset contrastivo ausente. Usando direções aleatórias estáveis...", fg=typer.colors.YELLOW)
        torch.manual_seed(seed)
        v1 = torch.randn(d_model, device=device, dtype=dtype)
        v1 = v1 / (v1.norm() + 1e-8)
        w_rand = torch.randn(d_model, device=device, dtype=dtype)
        proj = torch.dot(w_rand, v1) * v1
        v2 = w_rand - proj
        v2 = v2 / (v2.norm() + 1e-8)
    
    repair_acts = []
    handles = []
    
    def capture_repair_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            repair_acts.append(h[:, -1:, :].detach().cpu().float().numpy())
            
    for l in range(l_start, l_end + 1):
        mod = layers[l].self_attn.o_proj
        handles.append(mod.register_forward_hook(capture_repair_hook))
        
    typer.secho(f"🔹 Injetando SPPS na Camada {layer_intervention} (R={R}) e monitorando Reparadores {l_start}-{l_end}...", fg=typer.colors.CYAN)
    with spps_rotational_hook(layers[layer_intervention], v1, v2, theta=np.pi/2, R=R):
        for item in tqdm(items, desc="Coletando ativações", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            inputs = tokenizer(f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:", return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
                
    for h in handles:
        h.remove()
        
    acts = np.array(repair_acts).squeeze()
    from scipy.spatial.distance import pdist, squareform
    X = acts.reshape(-1, acts.shape[-1])
    
    typer.secho("\n🔹 Calculando matriz de distância e rodando TDA (Ripser)...", fg=typer.colors.CYAN)
    D = squareform(pdist(X, metric='euclidean'))
    res = ripser(D, maxdim=1, distance_matrix=True)
    dgms = res['dgms']
    
    plt.figure(figsize=(6, 6))
    plot_diagrams(dgms, show=False)
    plt.title(f"Persistência: Cabeças de Reparo (Layers {l_start}-{l_end})")
    
    timestamp = int(time.time())
    out_path = Path(f"docs/antigr_reports/freeman_stabilization_{timestamp}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    
    elapsed = time.time() - start_time
    typer.secho(f"\n✅ Pipeline Concluído em {elapsed:.1f} segundos.", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"📁 Gráfico salvo em: {out_path}\n", fg=typer.colors.BLUE)
