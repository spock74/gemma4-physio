"""
================================================================================
Copyright (c) 2026 Jose E Moraes. All rights reserved.

NOME CANÔNICO: Causal Scrubbing (Ablação de Atrator Semântico)
INTENÇÃO: Avaliar a resiliência causal do modelo ao remover direções factuais do pre-fill.
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
import umap
from pathlib import Path
from tqdm import tqdm
import typer

from gemma4_physio.data_loader import PopQASampler
from gemma4_physio.spps_hooks import spps_ablation_hook, spps_rotational_hook

def get_activations(model, tokenizer, items, target_layer, device):
    acts = []
    capture_handle = target_layer.register_forward_hook(
        lambda m, i, o: acts.append(o[0][:, -1:, :].detach().cpu().float().numpy() if isinstance(o, tuple) else o[:, -1:, :].detach().cpu().float().numpy())
    )
    for item in tqdm(items, desc="Extraindo ativações originais", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        inputs = tokenizer(f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:", return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
    capture_handle.remove()
    return np.array(acts).squeeze()

def get_accuracy_and_acts(model, tokenizer, items, target_layer, v1, v2, device, do_ablation=True):
    acts = []
    correct = 0
    capture_handle = target_layer.register_forward_hook(
        lambda m, i, o: acts.append(o[0][:, -1:, :].detach().cpu().float().numpy() if isinstance(o, tuple) else o[:, -1:, :].detach().cpu().float().numpy())
    )
    
    desc_str = "Aplicando Causal Scrubbing" if do_ablation else "Linha de Base (Sem Intervenção)"
    for item in tqdm(items, desc=desc_str, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        prompt = f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        if do_ablation:
            ctx = spps_ablation_hook(target_layer, v1, v2)
        else:
            from contextlib import nullcontext
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

def run_causal_scrubbing(config_dict: dict, model, tokenizer, device: str):
    start_time = time.time()
    typer.secho("\n" + "="*50, fg=typer.colors.MAGENTA, bold=True)
    typer.secho("🚀 INICIANDO PIPELINE: CAUSAL SCRUBBING", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("="*50 + "\n", fg=typer.colors.MAGENTA, bold=True)
    
    json_path = Path("data/popqa/popqa_full.json")
    sampler = PopQASampler(json_path)
    
    classes = config_dict["target_classes"]
    seed = config_dict.get("seed", 100)
    layer_idx = config_dict.get("layer_target", 12)
    
    if config_dict.get("purified", False):
        data = sampler.sample_purified_representatives(classes, seed=seed)
        typer.secho("🔹 Amostragem Purificada Ativada (Removendo atalhos nominais/ortográficos).", fg=typer.colors.YELLOW)
    else:
        data = sampler.sample_5x5_representatives(classes, seed=seed)
    items = [item for c_items in data.values() for item in c_items]
    
    dtype = model.dtype
    layers = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    target_layer = layers[layer_idx]
    
    typer.secho(f"🔹 Alvo: Camada {layer_idx} | Amostras: {len(items)}", fg=typer.colors.CYAN)
    
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
        typer.secho("🔹 Etapa 1: Extraindo direção factual real via Difference-of-Means...", fg=typer.colors.CYAN)
        v1 = extract_difference_of_means(model, tokenizer, known_prompts, unknown_prompts, target_layer, device)
        base_acts = get_activations(model, tokenizer, items, target_layer, device)
    else:
        typer.secho("⚠️ Dataset contrastivo ausente. Usando centróide como fallback...", fg=typer.colors.YELLOW)
        base_acts = get_activations(model, tokenizer, items, target_layer, device)
        v1_np = np.mean(base_acts, axis=0)
        v1 = torch.tensor(v1_np, device=device, dtype=dtype)
        v1 = v1 / (v1.norm() + 1e-8)
        
    base_acts_t = torch.tensor(base_acts, device="cpu", dtype=torch.float32)
    base_acts_centered = base_acts_t - base_acts_t.mean(dim=0)
    _, _, V = torch.pca_lowrank(base_acts_centered, q=1)
    v2 = V[:, 0].to(device).to(dtype)
    
    proj = torch.dot(v2, v1) * v1
    v2 = v2 - proj
    v2 = v2 / (v2.norm() + 1e-8)
    
    typer.secho("🔹 Etapa 2: Rodando Baseline (Sem Ablação)...", fg=typer.colors.CYAN)
    acts_clean, acc_clean = get_accuracy_and_acts(model, tokenizer, items, target_layer, v1, v2, device, do_ablation=False)
    
    typer.secho("🔹 Etapa 3: Rodando Causal Scrubbing (Knockout do Plano Semântico)...", fg=typer.colors.CYAN)
    acts_ablated, acc_ablated = get_accuracy_and_acts(model, tokenizer, items, target_layer, v1, v2, device, do_ablation=True)
    
    typer.secho(f"\n📊 RESULTADOS:", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"  ➜ Acurácia Limpa: {acc_clean*100:.1f}%", fg=typer.colors.GREEN)
    typer.secho(f"  ➜ Acurácia Ablacionada: {acc_ablated*100:.1f}%", fg=typer.colors.RED)
    
    X = np.vstack([acts_clean, acts_ablated])
    labels = np.array([0]*len(acts_clean) + [1]*len(acts_ablated))
    
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    umap_emb = reducer.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_emb[labels==0, 0], umap_emb[labels==0, 1], c='green', label=f'Baseline (Acc: {acc_clean*100:.1f}%)', alpha=0.7)
    plt.scatter(umap_emb[labels==1, 0], umap_emb[labels==1, 1], c='red', label=f'Ablacionado (Acc: {acc_ablated*100:.1f}%)', alpha=0.7)
    for i in range(len(acts_clean)):
        plt.plot([umap_emb[i, 0], umap_emb[i + len(acts_clean), 0]], 
                 [umap_emb[i, 1], umap_emb[i + len(acts_clean), 1]], 
                 'k-', alpha=0.1)
                 
    plt.title("Trajetória do Atrator após Causal Scrubbing")
    plt.legend()
    
    timestamp = int(time.time())
    out_path = Path(f"docs/antigr_reports/causal_scrubbing_{timestamp}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    
    elapsed = time.time() - start_time
    typer.secho(f"\n✅ Pipeline Concluído em {elapsed:.1f} segundos.", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"📁 Gráfico salvo em: {out_path}\n", fg=typer.colors.BLUE)
