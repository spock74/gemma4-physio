import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
from persim import plot_diagrams

def main():
    # Carregar os 300 pontos (30 planos x 10 ângulos)
    checkpoint_dir = Path("data/checkpoints")
    all_points = []
    
    if not checkpoint_dir.exists():
        print("Erro: Pasta de checkpoints não encontrada.")
        sys.exit(1)
        
    for k in range(30):
        plane_file = checkpoint_dir / f"plane_{k}.npy"
        if plane_file.exists():
            all_points.append(np.load(plane_file))
            
    if not all_points:
        print("Nenhum dado encontrado para plotar.")
        sys.exit(1)
        
    point_cloud = np.concatenate(all_points, axis=0)
    print(f"Dados carregados: {point_cloud.shape}")
    
    # Criar diretório de relatórios
    out_dir = Path("docs/antigr_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Plot PCA 2D dos Ativadores de Fase
    print("Gerando PCA 2D...")
    pca = PCA(n_components=2)
    pts_2d = pca.fit_transform(point_cloud)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=np.arange(len(pts_2d)), cmap='viridis', alpha=0.7)
    plt.colorbar(label='Índice do Ponto (Ordem de Geração)')
    plt.title('Projeção PCA 2D do Espaço de Fase SPPS')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(alpha=0.3)
    pca_path = out_dir / "pca_projection.png"
    plt.savefig(pca_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Matriz de Distância
    print("Gerando Heatmap da Matriz de Distâncias...")
    dists = squareform(pdist(point_cloud, metric='euclidean'))
    plt.figure(figsize=(10, 8))
    plt.imshow(dists, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distância Euclidiana')
    plt.title('Matriz de Distâncias do Espaço de Fase')
    dist_path = out_dir / "distance_matrix.png"
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Diagrama de Persistência (Ripser)
    print("Calculando TDA para o Diagrama de Persistência...")
    # Ripser requires max_dim=1 para Betti-0 e Betti-1
    res = ripser(point_cloud, maxdim=1)
    dgms = res['dgms']
    
    plt.figure(figsize=(8, 8))
    plot_diagrams(dgms, show=False, title='Diagrama de Persistência Topológica')
    pers_path = out_dir / "persistence_diagram.png"
    plt.savefig(pers_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Todos os gráficos foram gerados e salvos na pasta docs/antigr_reports/")

if __name__ == "__main__":
    main()
