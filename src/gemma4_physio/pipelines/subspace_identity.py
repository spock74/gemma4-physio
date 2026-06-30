"""
================================================================================
Copyright (c) 2026 Jose E Moraes. All rights reserved.

NOME CANÔNICO: Identidade de Subespaço (Subspace Identity PCA/UMAP)
INTENÇÃO: Verificar a separação topológica e linear de conceitos ortogonais no espaço latente.
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

def get_clean_activations(model, tokenizer, items, target_layer, device):
    activations = []
    def capture_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:
            activations.append(h[0, 0, :].detach().float().cpu().numpy())
    capture_handle = target_layer.register_forward_hook(capture_hook)
    for item in tqdm(items, desc="Coletando ativações", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        prompt = f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=2, do_sample=False, use_cache=True)
    capture_handle.remove()
    return np.array(activations)

def run_subspace_identity(config_dict: dict, model, tokenizer, device: str):
    start_time = time.time()
    typer.secho("\n" + "="*50, fg=typer.colors.MAGENTA, bold=True)
    typer.secho("🚀 INICIANDO PIPELINE: IDENTIDADE DE SUBESPAÇO", fg=typer.colors.MAGENTA, bold=True)
    typer.secho("="*50 + "\n", fg=typer.colors.MAGENTA, bold=True)
    
    json_path = Path("data/popqa/popqa_full.json")
    sampler = PopQASampler(json_path)
    
    g1_classes = config_dict["classes_g1"]
    g2_classes = config_dict["classes_g2"]
    seed = config_dict.get("seed", 42)
    layer_idx = config_dict.get("layer_target", 12)
    
    g1_data = sampler.sample_5x5_representatives(g1_classes, seed=seed)
    g2_data = sampler.sample_5x5_representatives(g2_classes, seed=seed)
    g1_items = [item for items in g1_data.values() for item in items]
    g2_items = [item for items in g2_data.values() for item in items]
    
    layers = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    target_layer = layers[layer_idx]
    
    typer.secho(f"🔹 Coletando ativações para o Grupo 1 (n={len(g1_items)})...", fg=typer.colors.CYAN)
    acts_g1 = get_clean_activations(model, tokenizer, g1_items, target_layer, device)
    typer.secho(f"🔹 Coletando ativações para o Grupo 2 (n={len(g2_items)})...", fg=typer.colors.CYAN)
    acts_g2 = get_clean_activations(model, tokenizer, g2_items, target_layer, device)
    
    X = np.vstack([acts_g1, acts_g2])
    labels = np.array([0]*len(acts_g1) + [1]*len(acts_g2))
    
    typer.secho("\n🔹 Executando PCA e UMAP nas matrizes de distância...", fg=typer.colors.CYAN)
    
    X_t = torch.tensor(X, device="cpu", dtype=torch.float32)
    X_centered = X_t - X_t.mean(dim=0)
    _, _, V = torch.pca_lowrank(X_centered, q=2)
    pca = torch.matmul(X_centered, V).numpy()
    
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    umap_emb = reducer.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(pca[labels==0, 0], pca[labels==0, 1], c='blue', label='Grupo 1', alpha=0.7)
    ax1.scatter(pca[labels==1, 0], pca[labels==1, 1], c='red', label='Grupo 2', alpha=0.7)
    ax1.set_title("PCA Linear: Separação Parcial")
    ax1.legend()
    
    ax2.scatter(umap_emb[labels==0, 0], umap_emb[labels==0, 1], c='blue', label='Grupo 1', alpha=0.7)
    ax2.scatter(umap_emb[labels==1, 0], umap_emb[labels==1, 1], c='red', label='Grupo 2', alpha=0.7)
    ax2.set_title("UMAP Não-Linear: Separação de Atratores")
    ax2.legend()
    
    timestamp = int(time.time())
    out_path = Path(f"docs/antigr_reports/subspace_identity_{timestamp}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    
    elapsed = time.time() - start_time
    typer.secho(f"\n✅ Pipeline Concluído em {elapsed:.1f} segundos.", fg=typer.colors.GREEN, bold=True)
    typer.secho(f"📁 Gráfico salvo em: {out_path}\n", fg=typer.colors.BLUE)
