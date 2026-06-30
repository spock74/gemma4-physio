import sys
import json
import numpy as np
from pathlib import Path

# Garantir visibilidade do pacote local
sys.path.append(str(Path(__file__).parent.parent / "src"))
from gemma4_physio.config import PipelineConfig
from gemma4_physio.tda_engine import compute_phase_space_tda

def main():
    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    
    config = PipelineConfig.load_from_yaml(config_path)
    sweep_cfg = config.pipelines.get("topological_sweep")
    if not sweep_cfg or not sweep_cfg.get("enabled"):
        print("Topological sweep pipeline disabled in configuration.", flush=True)
        return
        
    K = sweep_cfg["control_planes_K"]
    angles = sweep_cfg["angles_deg"]
    
    checkpoint_dir = Path("data/checkpoints")
    all_points = []
    
    print("Aggregating phase vectors from checkpoints...")
    for k in range(K):
        plane_file = checkpoint_dir / f"plane_{k}.npy"
        if not plane_file.exists():
            print(f"Error: Missing {plane_file}. You must run the orchestrator completely first.")
            sys.exit(1)
            
        plane_pts = np.load(plane_file)
        all_points.append(plane_pts)
        
    point_cloud = np.concatenate(all_points, axis=0)
    print(f"Aggregated Point Cloud shape: {point_cloud.shape}")
    
    # Validação do shape
    expected_pts = K * len(angles)
    if point_cloud.shape[0] != expected_pts:
        print(f"Warning: Expected {expected_pts} points, but got {point_cloud.shape[0]}")
        
    ratio = sweep_cfg["tda_config"]["betti_0_threshold_ratio"]
    print("Computing TDA (Ripser)...")
    betti_0, h1_persistence, _ = compute_phase_space_tda(point_cloud, threshold_ratio=ratio)
    
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
        
    print(f"\n--- TDA Computed Successfully ---")
    print(f"Results: Betti-0 = {betti_0} | H1 Persistence = {h1_persistence:.4f}")
    print(f"JSON metrics written to: {out_path}")

if __name__ == "__main__":
    main()
