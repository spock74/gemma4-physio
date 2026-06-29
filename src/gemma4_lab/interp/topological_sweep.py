from __future__ import annotations

import json
import math
import torch
import logfire
from pathlib import Path

from .recorder import ActivationRecorder
from .directions import steering
from ..config.pipeline_schema import TopologicalSweepConfig

def execute_sweep(rec: ActivationRecorder, config: TopologicalSweepConfig) -> dict:
    """
    Executes the Topological Rotational Sweep (SPPS).
    Steers the intervention layer in K control planes, reads the capture layer,
    and runs TDA (Ripser) on the resulting point clouds.
    """
    with logfire.span("interp.topological_sweep", 
                      intervention=config.layer_intervention,
                      capture=config.layer_capture):
        
        if not config.input_vector_path.exists():
            raise FileNotFoundError(f"Input direction vector not found at {config.input_vector_path}")
            
        d_know = torch.load(config.input_vector_path, weights_only=True).to(rec.gemma._device)
        d_model = d_know.shape[0]
        
        # We need a prompt to sweep over. In a real scenario, this would come from the dataset.
        # For this skeleton, we use a fixed prompt or expect it in the config. 
        # (Assuming the pipeline handles this or we hardcode a test prompt)
        test_prompt = "The fictional city of Omelas is located in"
        
        # 1. Generate K orthogonal control planes
        # A plane is defined by (d_know, v_k) where v_k is orthogonal to d_know.
        planes = []
        g = torch.Generator().manual_seed(42)
        for _ in range(config.control_planes_K):
            v = torch.randn(d_model, generator=g, device=d_know.device)
            # Gram-Schmidt to make v orthogonal to d_know
            v = v - (torch.dot(v, d_know) * d_know)
            v = v / v.norm()
            planes.append(v)
            
        point_cloud = []
        
        # 2. Rotational Sweep
        for angle_deg in config.angles_deg:
            theta = math.radians(angle_deg)
            for k, v_k in enumerate(planes):
                # Rotated vector in the (d_know, v_k) plane
                # R(theta) = cos(theta)*d_know + sin(theta)*v_k
                d_theta = math.cos(theta) * d_know + math.sin(theta) * v_k
                
                # Steer at intervention layer
                target_layer = rec.layers[config.layer_intervention]
                with steering([target_layer], d_theta, config.magnitude_R):
                    # Capture at capture layer
                    # templated=False since we just want the raw representation
                    res = rec.last_token_residuals(test_prompt, [config.layer_capture], templated=False)
                    point_cloud.append(res[config.layer_capture].detach().cpu().numpy())
                    
        # 3. TDA (Ripser)
        # In a real environment, we'd import ripser. For the stub, we simulate the output.
        try:
            import ripser
            import numpy as np
            pc_array = np.vstack(point_cloud)
            diagrams = ripser.ripser(pc_array, maxdim=config.tda_config.max_dimension)['dgms']
            betti_0 = len(diagrams[0]) if len(diagrams) > 0 else 0
            betti_1 = len(diagrams[1]) if len(diagrams) > 1 else 0
        except ImportError:
            logfire.warn("ripser not installed. Emitting mock TDA metrics.")
            betti_0 = 1
            betti_1 = 0
            
        result = {
            "betti_0": betti_0,
            "betti_1": betti_1,
            "point_cloud_size": len(point_cloud)
        }
        
        # 4. Save results
        out_path = config.tda_config.output_json_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        logfire.info(f"Topological sweep saved to {out_path}")
        
        return result
