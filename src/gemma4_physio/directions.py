"""
Copyright (c) 2026 Jose E Moraes. All rights reserved.
"""
import torch
from tqdm import tqdm

def extract_difference_of_means(model, tokenizer, positive_prompts, negative_prompts, target_layer, device) -> torch.Tensor:
    """
    Extrai a direção conceitual através da diferença de médias (Difference of Means).
    Retorna um tensor normalizado com dimensões [d_model] posicionado no device e tipo do modelo.
    """
    dtype = model.dtype
    
    def capture_last_token_activation(prompts):
        acts = []
        
        def capture_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            # Captura a ativação do último token de pre-fill
            acts.append(h[0, -1, :].detach().float().cpu())
            
        handle = target_layer.register_forward_hook(capture_hook)
        for prompt in tqdm(prompts, desc="Extraindo ativações"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
        handle.remove()
        return torch.stack(acts)

    pos_acts = capture_last_token_activation(positive_prompts)
    neg_acts = capture_last_token_activation(negative_prompts)
    
    if not torch.isfinite(pos_acts).all() or not torch.isfinite(neg_acts).all():
        raise ValueError("Erro: Detectadas ativações inválidas (NaN/Inf) durante a extração.")
        
    diff = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
    diff = diff.to(device).to(dtype)
    return diff / (diff.norm() + 1e-8)
