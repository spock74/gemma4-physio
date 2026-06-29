import pytest
import json
import torch
from pathlib import Path

from gemma4_lab.interp.extraction import extract_direction
from gemma4_lab.config.pipeline_schema import DirectionExtractionConfig

def test_extract_direction_sociable(mock_recorder, tmp_path):
    """
    Sociable test: validates mathematical logic without mocking internal PyTorch 
    forward passes or ActivationRecorder's internal logic.
    """
    # 1. Create fake dataset
    dataset_path = tmp_path / "fake_dataset.json"
    dataset_path.write_text(json.dumps({
        "known": [{"prompt": "Paris is in", "answer": "France"}],
        "unknown": [{"prompt": "The fictional city is in", "answer": "Nothing"}]
    }))
    
    # 2. Config
    out_vector = tmp_path / "d_know.pt"
    config = DirectionExtractionConfig(
        enabled=True,
        dataset_path=dataset_path,
        layer_target=0,
        sample_size=1,
        output_vector_path=out_vector
    )
    
    # 3. Extract
    # We pass templated=False to avoid chat template errors with our dummy tokenizer
    d_vector = extract_direction(mock_recorder, config, templated=False)
    
    # 4. Assertions
    assert isinstance(d_vector, torch.Tensor)
    assert d_vector.dim() == 1
    # TinyTransformer d_model = 16
    assert d_vector.shape[0] == 16
    
    # Vector must be unit-normalized
    norm = d_vector.norm().item()
    assert abs(norm - 1.0) < 1e-4
    
    # File must be saved
    assert out_vector.exists()
    saved_vector = torch.load(out_vector, weights_only=True)
    assert torch.allclose(d_vector, saved_vector)
