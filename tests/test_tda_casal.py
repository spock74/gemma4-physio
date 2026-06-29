import pytest
import torch
import json
from pathlib import Path

from gemma4_lab.config.pipeline_schema import (
    TopologicalSweepConfig, 
    TdaConfig, 
    WeightAmortizationConfig
)
from gemma4_lab.interp.topological_sweep import execute_sweep
from gemma4_lab.interp.weight_amortization import execute_casal

def test_topological_sweep_sociable(mock_recorder, tmp_path):
    """
    Tests the topological sweep logic purely using the TinyTransformer
    and without relying on external dataset files.
    """
    # Create fake direction vector
    d_model = 16 # TinyTransformer d_model
    d_know = torch.randn(d_model)
    d_know = d_know / d_know.norm()
    
    vec_path = tmp_path / "d_know.pt"
    torch.save(d_know, vec_path)
    
    out_json = tmp_path / "tda_results.json"
    
    config = TopologicalSweepConfig(
        enabled=True,
        input_vector_path=vec_path,
        layer_intervention=0,
        layer_capture=1,
        magnitude_R=10.0,
        control_planes_K=2,
        angles_deg=[0, 90, 180],
        tda_config=TdaConfig(
            max_dimension=1,
            betti_0_threshold_ratio=0.5,
            output_json_path=out_json
        )
    )
    
    result = execute_sweep(mock_recorder, config)
    
    assert "betti_0" in result
    assert "betti_1" in result
    assert result["point_cloud_size"] == len(config.angles_deg) * config.control_planes_K
    assert out_json.exists()

def test_weight_amortization_sociable(mock_recorder, tmp_path):
    """
    Tests the CASAL gradient loop by asserting that only the target layer
    receives updates.
    """
    out_model = tmp_path / "casal_layer_0.pt"
    
    config = WeightAmortizationConfig(
        enabled=True,
        layer_target=0,
        epochs=3,
        learning_rate=0.01,
        optimizer="AdamW",
        lambda_scale=1.0,
        output_model_path=out_model
    )
    
    model = mock_recorder.gemma._model
    
    # Take a snapshot of weights before training
    # TinyLayer has a parameter `weight`
    layer_0_weights_before = model.model.layers[0].weight.clone()
    layer_1_weights_before = model.model.layers[1].weight.clone()
    
    result = execute_casal(mock_recorder, config)
    
    assert result["epochs_completed"] == 3
    assert out_model.exists()
    
    layer_0_weights_after = model.model.layers[0].weight
    layer_1_weights_after = model.model.layers[1].weight
    
    # Layer 0 (target) should have changed
    assert not torch.allclose(layer_0_weights_before, layer_0_weights_after)
    
    # Layer 1 (frozen) should be identical
    assert torch.allclose(layer_1_weights_before, layer_1_weights_after)
