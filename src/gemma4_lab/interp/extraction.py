from __future__ import annotations

from pathlib import Path
import json
import torch
import logfire

from .recorder import ActivationRecorder, contaminated_layers_from_residuals
from .directions import diff_of_means_direction
from ..config.pipeline_schema import DirectionExtractionConfig

def extract_direction(
    rec: ActivationRecorder, 
    config: DirectionExtractionConfig,
    templated: bool = True,
    instruction: str = "Answer with the fact, continuing the sentence."
) -> torch.Tensor:
    """
    Extracts a concept direction using Difference of Means.
    Returns a normalized [d_model] tensor.
    """
    with logfire.span("interp.extraction", layer=config.layer_target):
        # 1. Load dataset
        if not config.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")
            
        corpus = json.loads(config.dataset_path.read_text(encoding="utf-8"))
        # we assume it has "known" and "unknown" keys representing positive/negative examples
        # and sample size limits the number of items we process to save time in tests/sweeps
        positive_items = corpus.get("known", [])[:config.sample_size]
        negative_items = corpus.get("unknown", [])[:config.sample_size]
        
        if not positive_items or not negative_items:
            raise ValueError("Dataset must contain 'known' and 'unknown' arrays.")

        layer = config.layer_target
        candidate_layers = [layer]

        def capture(stem: str) -> dict[int, torch.Tensor]:
            if templated:
                return rec.last_token_residuals(instruction, candidate_layers, True, stem)
            return rec.last_token_residuals(stem, candidate_layers, False)

        # 2. Capture Residuals
        pos_res = [capture(it["prompt"]) for it in positive_items]
        neg_res = [capture(it["prompt"]) for it in negative_items]

        # 3. Numerical Health Check
        contaminated = contaminated_layers_from_residuals(pos_res + neg_res)
        if layer in contaminated:
            raise ValueError(
                f"Layer {layer} is contaminated with non-finite activations. "
                "Extraction failed."
            )

        # 4. Calculate Difference of Means
        d_vector = diff_of_means_direction(
            [r[layer] for r in pos_res], 
            [r[layer] for r in neg_res]
        )

        # 5. Save Output
        out_path = config.output_vector_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d_vector, out_path)
        logfire.info(f"Saved extracted direction to {out_path}")
        
        return d_vector
