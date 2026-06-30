import numpy as np
import ripser
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Any

def compute_phase_space_tda(point_cloud: np.ndarray, threshold_ratio: float = 0.5) -> Tuple[int, float, Any]:
    """
    Computa Betti-0 e a persistência H1 sobre a nuvem de pontos de fase.
    point_cloud: numpy array de dimensão [K * len(angles), d_model]
    threshold_ratio: proporção da distância média para o limiar de Betti-0
    """
    # 1. Computar matriz de distâncias euclidianas par a par
    dists_matrix = squareform(pdist(point_cloud, metric='euclidean'))
    mean_dist = float(np.mean(dists_matrix[dists_matrix > 0])) if np.any(dists_matrix > 0) else 1e-8
    betti_0_threshold = threshold_ratio * mean_dist
    
    # 2. Executar filtração de Rips via ripser
    result = ripser.ripser(point_cloud, maxdim=1)
    dgms = result['dgms']
    
    # 3. Calcular Betti-0 com base no limiar dinâmico
    h0_dgms = dgms[0]
    betti_0 = sum(1 for birth, death in h0_dgms if death > betti_0_threshold)
    
    # 4. Calcular persistência H1 acumulada de ciclos finitos
    h1_dgms = dgms[1]
    total_h1_persistence = 0.0
    for birth, death in h1_dgms:
        if np.isfinite(death):
            total_h1_persistence += (death - birth)
            
    return betti_0, total_h1_persistence, h1_dgms
