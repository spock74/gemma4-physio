"""
================================================================================
Copyright (c) 2026 Jose E Moraes. All rights reserved.

NOME CANÔNICO: Multi-Layer Causal Scrubbing (Ablação Multicamada em Série)
INTENÇÃO: Avaliar a resiliência do modelo bloqueando o Self-Repair via ablação
           simultânea na Camada 12 e nas camadas subsequentes (20-26).
================================================================================
"""
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path
from tqdm import tqdm
import typer
from contextlib import nullcontext

from gemma4_physio.data_loader import PopQASampler
from gemma4_physio.spps_hooks import spps_ablation_hook, spps_multi_ablation_hook

def get_multi_accuracy_and_acts(model, tokenizer, items, layer_indices, layers, v1_dict, v2_dict, device, do_ablation=True):
    # Vamos capturar as ativações da primeira camada de intervenção (geralmente Camada 12) para o plot UMAP
    monitor_layer_idx = layer_indices[0]
    monitor_layer = layers[monitor_layer_idx]
    
    acts = []
    correct = 0
    
    capture_handle = monitor_layer.register_forward_hook(
        lambda m, i, o: acts.append(o[0][:, -1:, :].detach().cpu().float().numpy() if isinstance(o, tuple) else o[:, -1:, :].detach().cpu().float().numpy())
    )
    
    desc_str = "Aplicando Multi-Layer Scrubbing" if do_ablation else "Linha de Base (Sem Intervenção)"
    
    # Prepara a lista de tuplas (layer_module, v1, v2) para o hook multi-camada
    layers_with_dirs = [(layers[idx], v1_dict[idx], v2_dict[idx]) for idx in layer_indices]
    
    for item in tqdm(items, desc=desc_str, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        prompt = f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        if do_ablation:
            ctx = spps_multi_ablation_hook(layers_with_dirs)
        else:
            ctx = nullcontext()
            
        with ctx:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        import json
        try:
            allowed_answers = json.loads(item['answer'])
            if not isinstance(allowed_answers, list):
                allowed_answers = [str(allowed_answers)]
        except Exception:
            allowed_answers = [item['answer']]
            
        if any(ans.lower() in text.lower() for ans in allowed_answers):
            correct += 1
            
    capture_handle.remove()
    return np.array(acts).squeeze(), correct / len(items)

def run_multi_causal_scrubbing(config_dict: dict, model, tokenizer, device: str):
    start_time = time.time()
    typer.secho("\n" + "="*50, fg=typer.colors.MAGENTA, bold=True)
    typer.secho("🚀 INICIANDO PIPELINE: MULTI-LAYER CAUSAL SCRUBBING", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("="*50 + "\n", fg=typer.colors.MAGENTA, bold=True)
    
    json_path = Path("data/popqa/popqa_full.json")
    sampler = PopQASampler(json_path)
    
    classes = config_dict["target_classes"]
    seed = config_dict.get("seed", 100)
    layer_indices = config_dict.get("layers_target", [12, 20, 21, 22, 23, 24, 25, 26])
    
    if config_dict.get("purified", False):
        data = sampler.sample_purified_representatives(classes, seed=seed)
        typer.secho("🔹 Amostragem Purificada Ativada (Removendo atalhos nominais/ortográficos).", fg=typer.colors.YELLOW)
    else:
        data = sampler.sample_5x5_representatives(classes, seed=seed)
    items = [item for c_items in data.values() for item in c_items]
    
    dtype = model.dtype
    layers = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    
    # Mapeia as camadas alvo
    target_layers = [(idx, layers[idx]) for idx in layer_indices]
    
    typer.secho(f"🔹 Alvos: Camadas {layer_indices} | Amostras: {len(items)}", fg=typer.colors.CYAN)
    
    contrast_path = Path("data/eval/entity_knowledge_contrast.json")
    if contrast_path.exists():
        import json
        with open(contrast_path, "r") as f:
            contrast_data = json.load(f)
        known_items = contrast_data["known"][:15]
        unknown_items = contrast_data["unknown"][:15]
        known_prompts = [f"Answer the following question succinctly.\nQuestion: {it['prompt']}?\nAnswer:" for it in known_items]
        unknown_prompts = [f"Answer the following question succinctly.\nQuestion: {it['prompt']}?\nAnswer:" for it in unknown_items]
        
        from gemma4_physio.directions import extract_multi_difference_of_means
        typer.secho("🔹 Etapa 1: Extraindo direções factuais (DOM) para todas as camadas...", fg=typer.colors.CYAN)
        v1_dict = extract_multi_difference_of_means(model, tokenizer, known_prompts, unknown_prompts, target_layers, device)
        
        # Extração otimizada das ativações de base de uma única vez
        base_acts_dict = {idx: [] for idx in layer_indices}
        
        def make_base_hook(idx):
            def capture_hook(_m, _i, o):
                h = o[0] if isinstance(o, tuple) else o
                base_acts_dict[idx].append(h[0, -1, :].detach().float().cpu().numpy())
            return capture_hook
            
        handles = [module.register_forward_hook(make_base_hook(idx)) for idx, module in target_layers]
        for item in tqdm(items, desc="Extraindo ativações base para PCA"):
            inputs = tokenizer(f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:", return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
        for h in handles:
            h.remove()
            
    else:
        typer.secho("❌ Erro: Dataset contrastivo 'entity_knowledge_contrast.json' necessário para o Multi-Layer.", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    # Calcula v2 (projeção ortogonal PCA) para cada camada
    v2_dict = {}
    for idx in layer_indices:
        acts = np.array(base_acts_dict[idx]).squeeze()
        base_acts_t = torch.tensor(acts, device="cpu", dtype=torch.float32)
        base_acts_centered = base_acts_t - base_acts_t.mean(dim=0)
        _, _, V = torch.pca_lowrank(base_acts_centered, q=1)
        v2 = V[:, 0].to(device).to(dtype)
        
        v1 = v1_dict[idx]
        proj = torch.dot(v2, v1) * v1
        v2 = v2 - proj
        v2 = v2 / (v2.norm() + 1e-8)
        v2_dict[idx] = v2
        
    typer.secho("🔹 Etapa 2: Rodando Baseline (Sem Ablação)...", fg=typer.colors.CYAN)
    acts_clean, acc_clean = get_multi_accuracy_and_acts(model, tokenizer, items, layer_indices, layers, v1_dict, v2_dict, device, do_ablation=False)
    
    typer.secho("🔹 Etapa 3: Rodando Causal Scrubbing Multicamada (Knockout em Série)...", fg=typer.colors.CYAN)
    acts_ablated, acc_ablated = get_multi_accuracy_and_acts(model, tokenizer, items, layer_indices, layers, v1_dict, v2_dict, device, do_ablation=True)
    
    typer.secho(f"\n📊 RESULTADOS:", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"  ➜ Acurácia Limpa: {acc_clean*100:.1f}%", fg=typer.colors.GREEN)
    typer.secho(f"  ➜ Acurácia Ablacionada: {acc_ablated*100:.1f}%", fg=typer.colors.RED)
    
    # UMAP Plot das ativações da camada de monitoramento
    X = np.vstack([acts_clean, acts_ablated])
    labels = np.array([0]*len(acts_clean) + [1]*len(acts_ablated))
    
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    umap_emb = reducer.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_emb[labels==0, 0], umap_emb[labels==0, 1], c='green', label=f'Baseline (Acc: {acc_clean*100:.1f}%)', alpha=0.7)
    plt.scatter(umap_emb[labels==1, 0], umap_emb[labels==1, 1], c='red', label=f'Ablacionado Multicamada (Acc: {acc_ablated*100:.1f}%)', alpha=0.7)
    for i in range(len(acts_clean)):
        plt.plot([umap_emb[i, 0], umap_emb[i + len(acts_clean), 0]], 
                 [umap_emb[i, 1], umap_emb[i + len(acts_clean), 1]], 
                 'k-', alpha=0.1)
                 
    plt.title(f"Trajetória da Ativação (Layer {layer_indices[0]}) após Causal Scrubbing Multicamada")
    plt.legend()
    
    timestamp = int(time.time())
    out_path = Path(f"docs/antigr_reports/multi_causal_scrubbing_{timestamp}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    
    elapsed = time.time() - start_time
    typer.secho(f"\n✅ Pipeline Concluído em {elapsed:.1f} segundos.", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"📁 Gráfico salvo em: {out_path}\n", fg=typer.colors.BLUE)
