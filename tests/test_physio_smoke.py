"""
Copyright (c) 2026 Jose E Moraes. All rights reserved.
"""
import torch
import numpy as np
from gemma4_physio.directions import extract_difference_of_means
from gemma4_physio.spps_hooks import spps_rotational_hook, spps_ablation_hook
from gemma4_physio.tda_engine import compute_phase_space_tda

def test_difference_of_means(tiny_model, dummy_tokenizer):
    pos_prompts = ["The capital of France is", "The capital of Japan is"]
    neg_prompts = ["The capital of Wakanda is", "The capital of Atlantis is"]
    
    target_layer = tiny_model.model.layers[1]
    d_vector = extract_difference_of_means(
        tiny_model, 
        dummy_tokenizer, 
        pos_prompts, 
        neg_prompts, 
        target_layer, 
        device="cpu"
    )
    
    assert isinstance(d_vector, torch.Tensor)
    assert d_vector.shape == (16,)
    # Normalized check
    assert torch.isclose(d_vector.norm(), torch.tensor(1.0, dtype=d_vector.dtype), atol=1e-5)

def test_spps_hooks(tiny_model, dummy_tokenizer):
    d_model = tiny_model.d_model
    v1 = torch.zeros(d_model)
    v1[0] = 1.0
    v2 = torch.zeros(d_model)
    v2[1] = 1.0
    
    target_layer = tiny_model.model.layers[1]
    prompt = "The capital of France is"
    inputs = dummy_tokenizer(prompt)
    
    # Test rotational hook (Steering)
    with spps_rotational_hook(target_layer, v1, v2, theta=0.0, R=10.0):
        out = tiny_model.generate(inputs["input_ids"], max_new_tokens=1)
        assert out is not None

    # Test ablation hook
    with spps_ablation_hook(target_layer, v1, v2):
        out = tiny_model.generate(inputs["input_ids"], max_new_tokens=1)
        assert out is not None

def test_tda_computation():
    # Create simple circular point cloud
    theta = np.linspace(0, 2*np.pi, 50)
    circle_2d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    # Project into 16D space
    proj = np.random.randn(2, 16)
    point_cloud = circle_2d @ proj
    
    betti_0, total_h1, h1_dgms = compute_phase_space_tda(point_cloud, threshold_ratio=0.5)
    
    assert isinstance(betti_0, int)
    assert isinstance(total_h1, float)
    assert betti_0 >= 1

def test_multi_layer_scrubbing(tiny_model, dummy_tokenizer):
    from gemma4_physio.spps_hooks import spps_multi_ablation_hook
    from gemma4_physio.directions import extract_multi_difference_of_means
    
    d_model = tiny_model.d_model
    v1 = torch.zeros(d_model)
    v1[0] = 1.0
    v2 = torch.zeros(d_model)
    v2[1] = 1.0
    
    target_layers = [(0, tiny_model.model.layers[0]), (1, tiny_model.model.layers[1])]
    
    pos_prompts = ["The capital of France is", "The capital of Japan is"]
    neg_prompts = ["The capital of Wakanda is", "The capital of Atlantis is"]
    
    v1_dict = extract_multi_difference_of_means(
        tiny_model, 
        dummy_tokenizer, 
        pos_prompts, 
        neg_prompts, 
        target_layers, 
        device="cpu"
    )
    
    assert 0 in v1_dict
    assert 1 in v1_dict
    assert v1_dict[0].shape == (16,)
    assert v1_dict[1].shape == (16,)
    
    layers_with_dirs = [
        (tiny_model.model.layers[0], v1, v2),
        (tiny_model.model.layers[1], v1, v2),
    ]
    prompt = "The capital of France is"
    inputs = dummy_tokenizer(prompt)
    
    with spps_multi_ablation_hook(layers_with_dirs):
        out = tiny_model.generate(inputs["input_ids"], max_new_tokens=1)
        assert out is not None
