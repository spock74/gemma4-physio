import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import umap

# --- CONFIGURAÇÃO DE ESTILO CIENTÍFICO ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['text.usetex'] = False # Mantenha False se não tiver LaTeX instalado localmente

def load_point_cloud() -> np.ndarray:
    """Carrega as ativações salvas em formato .npy do orchestrator."""
    checkpoint_dir = "data/checkpoints"
    all_points = []
    
    if not os.path.exists(checkpoint_dir):
        print(f"Aviso: {checkpoint_dir} não encontrado. Gerando dados de teste...")
        np.random.seed(42)
        theta = np.linspace(0, 2*np.pi, 300)
        circle_2d = np.stack([np.cos(theta), np.sin(theta)], axis=1) 
        projection_matrix = np.random.randn(2, 3072)
        noise = np.random.normal(0, 0.1, (300, 3072))
        return (circle_2d @ projection_matrix) + noise
        
    for k in range(30):
        plane_file = os.path.join(checkpoint_dir, f"plane_{k}.npy")
        if os.path.exists(plane_file):
            all_points.append(np.load(plane_file))
            
    if not all_points:
        print("Nenhum dado .npy encontrado. Gerando dados de teste...")
        np.random.seed(42)
        return np.random.randn(300, 2560)
        
    return np.concatenate(all_points, axis=0)

def main():
    output_dir = "docs/antigr_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    X = load_point_cloud()
    M, d = X.shape
    print(f"Nuvem de pontos carregada com sucesso: {M} amostras, {d} dimensões.")
    
    # Vetor de coloração baseado no índice de geração (0 a 300) para rastrear a fase
    colors = np.arange(M)
    
    # ----------------------------------------------------
    # EXECUÇÃO DOS ALGORITMOS DE REDUÇÃO
    # ----------------------------------------------------
    print("Executando PCA linear clássico...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    
    print("Executando Kernel PCA (RBF)...")
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=1.0/d)
    X_kpca = kpca.fit_transform(X)
    
    print("Executando Isomap...")
    isomap = Isomap(n_neighbors=15, n_components=2, eigen_solver="dense")
    X_isomap = isomap.fit_transform(X)
    
    print("Executando UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, metric="euclidean")
    X_umap = reducer.fit_transform(X)
    
    # ----------------------------------------------------
    # PLOTAGEM DO GRID ACADÊMICO 2x2
    # ----------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    cmap = "viridis"
    
    # Subplot 1: PCA Linear
    sc1 = axs[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap=cmap, alpha=0.7, edgecolors='none', s=35)
    axs[0, 0].set_title("A. Linear PCA (Mapeamento de Projeção Reta)", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel(f"PC1 ({pc1_var:.1f}% var.)", fontsize=10)
    axs[0, 0].set_ylabel(f"PC2 ({pc2_var:.1f}% var.)", fontsize=10)
    axs[0, 0].grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 2: Kernel PCA (RBF)
    sc2 = axs[0, 1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=colors, cmap=cmap, alpha=0.7, edgecolors='none', s=35)
    axs[0, 1].set_title("B. Kernel PCA (RBF - Projeção em Hiperespaço Hilberteano)", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("RBF Component 1", fontsize=10)
    axs[0, 1].set_ylabel("RBF Component 2", fontsize=10)
    axs[0, 1].grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 3: Isomap
    sc3 = axs[1, 0].scatter(X_isomap[:, 0], X_isomap[:, 1], c=colors, cmap=cmap, alpha=0.7, edgecolors='none', s=35)
    axs[1, 0].set_title("C. Isomap (Preservação de Distância Geodésica)", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("Isomap Dimension 1", fontsize=10)
    axs[1, 0].set_ylabel("Isomap Dimension 2", fontsize=10)
    axs[1, 0].grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 4: UMAP
    sc4 = axs[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=colors, cmap=cmap, alpha=0.7, edgecolors='none', s=35)
    axs[1, 1].set_title("D. UMAP (Aproximação de Manifolds Riemannianos)", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("UMAP Dimension 1", fontsize=10)
    axs[1, 1].set_ylabel("UMAP Dimension 2", fontsize=10)
    axs[1, 1].grid(True, linestyle=':', alpha=0.5)
    
    # Barra de Cores Unificada Lateral
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Índice de Fase da Varredura (Ordem de Geração θ)", fontsize=11, fontweight='bold')
    
    # Ajustes finais de layout
    plt.subplots_adjust(right=0.9, hspace=0.25, wspace=0.2)
    fig.suptitle("Análise Comparativa de Variedades do Espaço de Fase SPPS\n(Gemma 3 4b-it | Camada 12 Intervention | Camada 13 Capture | R=15k)", 
                 fontsize=13, fontweight='bold', y=0.96)
    
    save_path = f"{output_dir}/dimension_reduction_comparison.png"
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    
    print(f"\nSucesso! Gráfico comparativo de alta resolução salvo em: {save_path}")

if __name__ == "__main__":
    main()
