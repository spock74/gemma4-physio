import math
import torch
import torch.nn as nn
from contextlib import contextmanager, ExitStack

@contextmanager
def spps_rotational_hook(layer: nn.Module, v1: torch.Tensor, v2: torch.Tensor, theta: float, R: float):
    """
    Context manager para registrar e remover o gancho SPPS no pre-fill.
    v1 e v2 devem ser tensores pré-normalizados na GPU.
    """
    def hook_fn(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        
        # Ignora a decodificação auto-regressiva (Bypass do KV-cache)
        if h.shape[1] == 1:
            return output
            
        # Alveja estritamente o último token do pre-fill
        h_last = h[:, -1:, :]
        clean_norm = h_last.norm(dim=-1, keepdim=True)
        
        # Projeção ortogonal no plano S
        proj_v1 = torch.einsum('bsd,d->bs', h_last, v1).unsqueeze(-1) * v1
        proj_v2 = torch.einsum('bsd,d->bs', h_last, v2).unsqueeze(-1) * v2
        h_last_perp = h_last - (proj_v1 + proj_v2)
        
        # Perturbação polar norm-preserving
        perturbation = R * (math.cos(theta) * v1 + math.sin(theta) * v2)
        patched_raw = h_last_perp + perturbation
        patched_norm = patched_raw.norm(dim=-1, keepdim=True) + 1e-8
        h_last_patched = patched_raw * (clean_norm / patched_norm)
        
        h_out = h.clone()
        h_out[:, -1:, :] = h_last_patched
        
        return (h_out, *output[1:]) if isinstance(output, tuple) else h_out

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def spps_ablation_hook(layer: nn.Module, v1: torch.Tensor, v2: torch.Tensor):
    """
    Context manager para registrar o gancho de ablação (Causal Scrubbing) no pre-fill.
    Zera (subtrai) os componentes ao longo das direções v1 e v2, sem adicionar ruído compensatório.
    """
    def hook_fn(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        
        if h.shape[1] == 1:
            return output
            
        h_last = h[:, -1:, :]
        clean_norm = h_last.norm(dim=-1, keepdim=True)
        
        proj_v1 = torch.einsum('bsd,d->bs', h_last, v1).unsqueeze(-1) * v1
        proj_v2 = torch.einsum('bsd,d->bs', h_last, v2).unsqueeze(-1) * v2
        
        patched_raw = h_last - (proj_v1 + proj_v2)
        patched_norm = patched_raw.norm(dim=-1, keepdim=True) + 1e-8
        h_last_patched = patched_raw * (clean_norm / patched_norm)
        
        h_out = h.clone()
        h_out[:, -1:, :] = h_last_patched
        
        return (h_out, *output[1:]) if isinstance(output, tuple) else h_out

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def spps_multi_ablation_hook(layers_with_dirs):
    """
    Context manager para registrar múltiplos ganchos de ablação (Causal Scrubbing) simultaneamente.
    layers_with_dirs: Lista de tuplas (layer_module, v1, v2)
    """
    with ExitStack() as stack:
        for layer, v1, v2 in layers_with_dirs:
            stack.enter_context(spps_ablation_hook(layer, v1, v2))
        yield
