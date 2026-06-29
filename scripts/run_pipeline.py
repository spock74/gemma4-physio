import sys
import os
from pathlib import Path
import json
import torch

# Ensure src/ is in the python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gemma4_lab.config.pipeline_schema import (
    PipelineConfig, 
    DirectionExtractionConfig,
    TopologicalSweepConfig,
    WeightAmortizationConfig
)
from gemma4_lab.models.factory import load_model_and_tokenizer
from gemma4_lab.interp.recorder import ActivationRecorder
import gemma4_lab.inference.hf_local as hf_local

def run_factual_probing_extraction(rec: ActivationRecorder, direction_config: DirectionExtractionConfig):
    print(f"Executing: Direction Extraction Pipeline (Layer {direction_config.layer_target})...")
    from gemma4_lab.interp.extraction import extract_direction
    
    d_vector = extract_direction(rec, direction_config)
    print("Direction Extraction completed successfully.")
    
    # 4. Evaluation
    from gemma4_lab.interp.evaluation import evaluate_causal_necessity, evaluate_causal_sufficiency
    corpus = json.loads(direction_config.dataset_path.read_text(encoding="utf-8"))
    
    print("Evaluating Causal Necessity...")
    nec_results = evaluate_causal_necessity(rec, corpus.get("known", []), d_vector)
    print(f"Necessity: Mean Logit Drop = {nec_results['mean_logit_drop']:+.2f}")
    
    print("Evaluating Causal Sufficiency...")
    suf_results = evaluate_causal_sufficiency(rec, corpus.get("unknown", []), d_vector)
    print(f"Sufficiency: Mean Entropy Drop = {suf_results['mean_entropy_drop']:+.3f}")

def run_topological_sweep(rec: ActivationRecorder, sweep_config: TopologicalSweepConfig):
    print(f"Executing: Topological Rotational Sweep (Intervention Layer {sweep_config.layer_intervention})...")
    from gemma4_lab.interp.topological_sweep import execute_sweep
    execute_sweep(rec, sweep_config)
    print("Topological Sweep completed successfully.")

def run_weight_amortization(rec: ActivationRecorder, casal_config: WeightAmortizationConfig):
    print(f"Executing: CASAL Weight Amortization (Target Layer {casal_config.layer_target})...")
    from gemma4_lab.interp.weight_amortization import execute_casal
    execute_casal(rec, casal_config)
    print("Weight Amortization completed successfully.")

def main():
    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    try:
        config = PipelineConfig.load_from_yaml(config_path)
        print(f"Configuration Validated: {config.experiment_name} (v{config.version})")
    except Exception as e:
        print(f"Validation Error in {config_path}:")
        print(e)
        sys.exit(1)

    # 1. Initialize Global Resources (Model, Tokenizer, Recorder) Once
    print(f"\nInitializing Global Resources on {config.model_cfg.device}...")
    model, tokenizer = load_model_and_tokenizer(config.model_cfg)
    
    # Setup Recorder using a dummy GemmaLocal interface to satisfy its type hint
    gemma = hf_local.GemmaLocal.__new__(hf_local.GemmaLocal)
    gemma._model = model
    gemma._tokenizer = tokenizer
    gemma._device = config.model_cfg.device
    gemma._load = lambda: None # bypass load
    
    rec = ActivationRecorder(gemma)
    print("Global Resources Initialized.\n")

    # 2. Execute Enabled Pipeline Steps Sequentially
    direction_raw = config.pipelines.get("direction_extraction", {})
    if direction_raw.get("enabled"):
        direction_config = DirectionExtractionConfig(**direction_raw)
        run_factual_probing_extraction(rec, direction_config)
        
    sweep_raw = config.pipelines.get("topological_sweep", {})
    if sweep_raw.get("enabled"):
        sweep_config = TopologicalSweepConfig(**sweep_raw)
        run_topological_sweep(rec, sweep_config)

    casal_raw = config.pipelines.get("weight_amortization", {})
    if casal_raw.get("enabled"):
        casal_config = WeightAmortizationConfig(**casal_raw)
        run_weight_amortization(rec, casal_config)

    print("\nAll enabled pipelines completed successfully.")

if __name__ == "__main__":
    main()
