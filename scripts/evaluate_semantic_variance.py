import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import torch
import numpy as np
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gemma4_physio.data_loader import PopQASampler
from gemma4_physio.spps_hooks import spps_rotational_hook
from gemma4_physio.observability import setup_logfire, logfire_memory_span

def compute_spatial_variance(points: np.ndarray) -> float:
    """Calcula a variância espacial (média das distâncias quadradas ao centroide)."""
    if len(points) == 0: return 0.0
    centroid = np.mean(points, axis=0)
    var = np.mean(np.sum((points - centroid)**2, axis=1))
    return float(var)

def get_activations_and_texts(model, tokenizer, items, target_layer, v1, v2, theta_rad, R, device):
    activations = []
    generated_logs = []
    
    def capture_hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:
            activations.append(h[0, 0, :].detach().float().cpu().numpy())
            
    capture_handle = target_layer.register_forward_hook(capture_hook)
    
    for item in items:
        prompt = f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 1. Geração Limpa (Baseline) para o log textual
        with logfire_memory_span("Clean Generation"):
            with torch.no_grad():
                out_clean = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        text_clean = tokenizer.decode(out_clean[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 2. Geração Perturbada (Coleta Ativações e Texto)
        with spps_rotational_hook(target_layer, v1, v2, theta_rad, R):
            with logfire_memory_span("Perturbed Generation"):
                with torch.no_grad():
                    out_rot = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        text_rot = tokenizer.decode(out_rot[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        generated_logs.append({
            "class": item.get("relation", "mixed"),
            "question": item["question"],
            "expected": item["answer"],
            "clean_generation": text_clean,
            "perturbed_generation": text_rot
        })
                
    capture_handle.remove()
    return np.array(activations), generated_logs

def main():
    setup_logfire() # Will load from config automatically
    print("Iniciando Teste de Variância Semântica SPPS com Salvação de Texto...")
    
    json_path = Path("data/popqa/popqa_full.json")
    sampler = PopQASampler(json_path)
    
    c_sim = ["screenwriter", "producer", "author", "director", "composer"]
    sim_data = sampler.sample_5x5_representatives(c_sim, seed=42)
    sim_items = [item for class_items in sim_data.values() for item in class_items]
    
    model_id = "google/gemma-3-4b-it"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Carregando {model_id} em {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    ).eval()
    
    d_model = model.config.text_config.hidden_size if hasattr(model.config, 'text_config') else model.config.hidden_size
    torch.manual_seed(42)
    v1 = torch.randn(d_model, device=device, dtype=dtype)
    v1 = v1 / (v1.norm() + 1e-8)
    
    g = torch.Generator(device=device).manual_seed(42)
    w_rand = torch.randn(d_model, generator=g, device=device, dtype=dtype)
    proj = torch.dot(w_rand, v1) * v1
    v2 = w_rand - proj
    v2 = v2 / (v2.norm() + 1e-8)
    
    target_layer = model.model.language_model.layers[12]
    theta_rad = np.pi / 2  # 90 graus
    R = 15.0
    
    all_logs = []
    
    print(f"\nColetando ativações e textos para o grupo Similar ({len(sim_items)} prompts)...")
    sim_acts, sim_logs = get_activations_and_texts(model, tokenizer, sim_items, target_layer, v1, v2, theta_rad, R, device)
    var_sim = compute_spatial_variance(sim_acts)
    print(f"Variância Grupo Similar (sigma^2_sim): {var_sim:.4f}")
    all_logs.extend(sim_logs)
    
    print("\nIniciando subconjuntos aleatórios mistos (Bootstrap)...")
    mixed_variances = []
    
    # Vamos gerar 5 subconjuntos mistos para ter uma boa amostragem (bootstrap)
    for i, mixed_subset in enumerate(sampler.generate_random_subsets_without_replacement()):
        if i >= 5: break # Limita a 5 subconjuntos mistos (25 prompts cada) para não demorar horas
        
        intersection = set(mixed_subset).intersection(set(c_sim))
        if len(intersection) > 2:
            continue
            
        print(f"  Amostrando Misto {i+1}: {mixed_subset}")
        mixed_data = sampler.sample_5x5_representatives(mixed_subset, seed=100+i)
        mixed_items = [item for class_items in mixed_data.values() for item in class_items]
        
        if len(mixed_items) < 25:
            continue
            
        mixed_acts, mixed_logs = get_activations_and_texts(model, tokenizer, mixed_items, target_layer, v1, v2, theta_rad, R, device)
        var_misto = compute_spatial_variance(mixed_acts)
        mixed_variances.append(var_misto)
        all_logs.extend(mixed_logs)
        print(f"  -> Variância Misto {i+1}: {var_misto:.4f}")
        
    var_misto_mean = np.mean(mixed_variances)
    var_misto_std = np.std(mixed_variances)
    
    z_score = (var_misto_mean - var_sim) / (var_misto_std + 1e-8)
    p_value = stats.norm.sf(z_score) # one-tailed
    
    print("\nSalvando logs em JSON...")
    os.makedirs("results", exist_ok=True)
    with open("results/semantic_resonance_texts.json", "w") as f:
        json.dump({
            "statistics": {
                "var_sim": var_sim,
                "var_misto_mean": var_misto_mean,
                "var_misto_std": var_misto_std,
                "z_score": z_score,
                "p_value": p_value
            },
            "generations": all_logs
        }, f, indent=2)
        
    print("\nRESULTADOS SALVOS EM: results/semantic_resonance_texts.json")
    print(f"P-Value: {p_value:.4f}")

if __name__ == "__main__":
    main()
