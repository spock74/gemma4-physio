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

def extract_multi_difference_of_means(model, tokenizer, positive_prompts, negative_prompts, target_layers, device) -> dict:
    """
    Extrai a direção conceitual para múltiplas camadas de uma só vez (otimizado).
    target_layers: Lista de tuplas (layer_idx, layer_module)
    Retorna: Dicionário {layer_idx: v1_tensor}
    """
    dtype = model.dtype
    
    # Dicionário para guardar as ativações de cada camada
    acts_dict = {layer_idx: [] for layer_idx, _ in target_layers}
    
    def make_hook(layer_idx):
        def capture_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            acts_dict[layer_idx].append(h[0, -1, :].detach().float().cpu())
        return capture_hook

    # Registra hooks em todas as camadas
    handles = []
    for layer_idx, layer_module in target_layers:
        handles.append(layer_module.register_forward_hook(make_hook(layer_idx)))
        
    def run_inference(prompts, desc):
        for prompt in tqdm(prompts, desc=desc, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
                
    # Roda inferência para ambos os grupos de prompts
    for layer_idx in acts_dict:
        acts_dict[layer_idx].clear()
    run_inference(positive_prompts, "Extraindo ativações positivas")
    pos_acts_dict = {layer_idx: torch.stack(acts_dict[layer_idx]) for layer_idx in acts_dict}
    
    for layer_idx in acts_dict:
        acts_dict[layer_idx].clear()
    run_inference(negative_prompts, "Extraindo ativações negativas")
    neg_acts_dict = {layer_idx: torch.stack(acts_dict[layer_idx]) for layer_idx in acts_dict}
    
    # Remove todos os hooks
    for handle in handles:
        handle.remove()
        
    # Calcula a diferença de médias por camada
    v1_dict = {}
    for layer_idx in acts_dict:
        pos_acts = pos_acts_dict[layer_idx]
        neg_acts = neg_acts_dict[layer_idx]
        diff = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        diff = diff.to(device).to(dtype)
        v1_dict[layer_idx] = diff / (diff.norm() + 1e-8)
        
    return v1_dict
