import os
import psutil
import torch
import logfire
from contextlib import contextmanager

# Fix for macOS OpenMP multiple runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_logfire(project_name="gemma4-physio"):
    """
    Inicializa o Logfire para o projeto baseado no pipeline_config.yaml.
    """
    from gemma4_physio.config import PipelineConfig
    from pathlib import Path
    
    # Load from default root location
    config_path = Path("pipeline_config.yaml")
    send_to_logfire = True
    if config_path.exists():
        cfg = PipelineConfig.load_from_yaml(config_path)
        send_to_logfire = cfg.observability.send_to_logfire
        
    logfire.configure(send_to_logfire=send_to_logfire)
    logfire.instrument_pydantic()
    
    # Instrumenta sistema se necessário
    logfire.instrument_system_metrics()
    logfire.info("Logfire setup complete. Monitoring system metrics.")

def get_mps_memory_usage_mb():
    """Retorna o uso aproximado da memória em MB para MPS, se aplicável."""
    if torch.backends.mps.is_available():
        # MPS doesn't have native memory tracking exposed identically to CUDA in PyTorch yet,
        # but we can grab driver allocation if available, or fallback to current RAM.
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0.0

def get_cuda_memory_usage_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0

@contextmanager
def logfire_memory_span(span_name="Inference Block"):
    """
    Context manager para envolver blocos de execução pesados (como model.generate)
    e reportar uso de memória antes e depois no logfire.
    """
    device_type = "MPS" if torch.backends.mps.is_available() else ("CUDA" if torch.cuda.is_available() else "CPU")
    
    with logfire.span(span_name) as span:
        mem_before = get_mps_memory_usage_mb() if device_type == "MPS" else get_cuda_memory_usage_mb()
        cpu_before = psutil.virtual_memory().percent
        
        span.set_attribute("device", device_type)
        span.set_attribute("mem_vram_mb_before", mem_before)
        span.set_attribute("mem_cpu_pct_before", cpu_before)
        
        yield
        
        mem_after = get_mps_memory_usage_mb() if device_type == "MPS" else get_cuda_memory_usage_mb()
        cpu_after = psutil.virtual_memory().percent
        
        span.set_attribute("mem_vram_mb_after", mem_after)
        span.set_attribute("mem_cpu_pct_after", cpu_after)
        
        # Explicitamente logar no console para o usuário ver
        delta = mem_after - mem_before
        logfire.info(
            "{span_name} - VRAM/MPS: {mem_after:.1f} MB (Delta: {delta:+.1f} MB) | CPU: {cpu_after}%",
            span_name=span_name,
            mem_after=mem_after,
            delta=delta,
            cpu_after=cpu_after
        )
        
        # Log a warning if memory spikes dangerously close to 12GB (12288 MB) for MPS
        if device_type == "MPS" and mem_after > 11000:
            logfire.warn("MPS Memory limit warning: Exceeded 11GB out of 12GB.")
