"""E1 — Entity-knowledge direction (causal).

[DEPRECATED]
This script has been refactored into a declarative Configuration-as-Code architecture.
Please use `scripts/run_pipeline.py` with `pipeline_config.yaml` to execute 
the Direction Extraction pipeline, which uses the new components in `interp/extraction.py`
and `interp/evaluation.py`.

This thin wrapper remains for legacy compatibility but is simply delegating to the new pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path
from ..config.pipeline_schema import PipelineConfig

RECALL_INSTRUCTION = "Answer with the fact, continuing the sentence."

def run(*args, **kwargs):
    print("WARNING: entity_knowledge.py is deprecated. Using run_pipeline.py underneath.")
    
    # Resolve project root
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "pipeline_config.yaml"
    
    if not config_path.exists():
        print(f"Error: {config_path} not found. Please create it or run from the project root.")
        sys.exit(1)
        
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
        
    config_dict = yaml.load(config_path.read_text(), Loader=Loader)
    config = PipelineConfig(**config_dict)
    
    # Import the pipeline runner
    sys.path.append(str(project_root / "scripts"))
    try:
        import run_pipeline
        run_pipeline.run_factual_probing_extraction(config)
    except ImportError as e:
        print(f"Failed to import run_pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
