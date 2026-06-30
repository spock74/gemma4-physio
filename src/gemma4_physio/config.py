import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
from pydantic import BaseModel, Field, conint, confloat
import yaml

class ModelConfig(BaseModel):
    model_id: str = Field(description="HuggingFace model identifier")
    device: str = Field(default="mps", description="Target hardware accelerator")
    precision: str = Field(default="bfloat16", description="Floating point precision")
    attn_implementation: str = Field(default="eager", description="Attention mechanics")
    cache_dir: Path = Field(description="HF cache directory")

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.precision == "bfloat16":
            return torch.bfloat16
        elif self.precision == "float16":
            return torch.float16
        return torch.float32

class TDAConfig(BaseModel):
    max_dimension: conint(ge=1, le=2) = 1
    betti_0_threshold_ratio: confloat(ge=0.0, le=1.0) = 0.5
    output_json_path: Path

class TopologicalSweepConfig(BaseModel):
    enabled: bool
    layer_intervention: conint(ge=0)
    layer_capture: conint(ge=0)
    magnitude_R: float
    control_planes_K: conint(ge=1)
    angles_deg: List[int]
    tda_config: TDAConfig

class ObservabilityConfig(BaseModel):
    logger: str
    log_level: str
    save_plots: bool
    plots_dir: Path
    send_to_logfire: bool = True

class PipelineConfig(BaseModel):
    version: str
    experiment_name: str
    model_settings: ModelConfig = Field(alias="model_config")
    pipelines: Dict[str, Any]
    observability: ObservabilityConfig
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
