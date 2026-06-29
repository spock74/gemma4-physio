from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, conint, confloat, model_validator
import torch
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

class DirectionExtractionConfig(BaseModel):
    enabled: bool
    dataset_path: Path
    layer_target: conint(ge=0)
    sample_size: conint(ge=1)
    output_vector_path: Path

class TdaConfig(BaseModel):
    max_dimension: conint(ge=1, le=2) = 1
    betti_0_threshold_ratio: confloat(ge=0.0, le=1.0) = 0.5
    output_json_path: Path

class TopologicalSweepConfig(BaseModel):
    enabled: bool
    input_vector_path: Path = Field(default=Path("data/vectors/d_know.pt"))
    layer_intervention: conint(ge=0)
    layer_capture: conint(ge=0)
    magnitude_R: float
    control_planes_K: conint(ge=1)
    angles_deg: List[int]
    tda_config: TdaConfig

class WeightAmortizationConfig(BaseModel):
    enabled: bool
    layer_target: conint(ge=0)
    epochs: conint(ge=1)
    learning_rate: confloat(gt=0.0)
    optimizer: str = "AdamW"
    lambda_scale: float = 1.0
    output_model_path: Path

class ObservabilityConfig(BaseModel):
    logger: str = "logfire"
    log_level: str = "INFO"
    save_plots: bool = True
    plots_dir: Path

class PipelineConfig(BaseModel):
    version: str
    experiment_name: str
    model_cfg: ModelConfig = Field(alias="model_config")
    pipelines: dict # Will hold sub-pipeline models
    observability: ObservabilityConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
