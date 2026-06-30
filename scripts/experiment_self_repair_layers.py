import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import torch
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gemma4_physio.data_loader import PopQASampler
from gemma4_physio.spps_hooks import spps_rotational_hook
from gemma4_physio.observability import setup_logfire, logfire_memory_span

def evaluate_survival(clean_text, rot_text):
    if not clean_text or not rot_text:
        return 0.0
    return SequenceMatcher(None, clean_text, rot_text).ratio()

def get_layer_texts(model, tokenizer, items, target_layer_idx, layers, v1, v2, theta_rad, R, device):
    survival_scores = []
    
    target_layer = layers[target_layer_idx]
    
    for item in items:
        prompt = f"Answer the following question succinctly.\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 1. Geração Limpa
        with logfire_memory_span(f"Clean Layer {target_layer}"):
            with torch.no_grad():
                out_clean = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        text_clean = tokenizer.decode(out_clean[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 2. Geração Perturbada
        with spps_rotational_hook(target_layer, v1, v2, theta_rad, R):
            with logfire_memory_span(f"Perturbed Layer {target_layer}"):
                with torch.no_grad():
                    out_rot = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        text_rot = tokenizer.decode(out_rot[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        score = evaluate_survival(text_clean, text_rot)
        survival_scores.append(score)
        
    return np.mean(survival_scores)

def main():
    setup_logfire()
    print("Iniciando Experimento de Varrimento de Self-Repair (Intervenção por Camada)...")
    
    json_path = Path("data/popqa/popqa_full.json")
    sampler = PopQASampler(json_path)
    
    # 25 Prompts coesos
    c_sim = ["screenwriter", "producer", "author", "director", "composer"]
    sim_data = sampler.sample_5x5_representatives(c_sim, seed=42)
    items = [item for class_items in sim_data.values() for item in class_items]
    
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
    
    theta_rad = np.pi / 2  # 90 graus
    R = 15.0
    
    layers_attr = model.model.language_model.layers if hasattr(model.model, 'language_model') else model.model.layers
    
    num_layers = len(layers_attr)
    print(f"Total de camadas encontradas: {num_layers}")
    
    # Camadas a intervir: pula de 4 em 4 e adiciona as duas últimas
    test_layers = list(range(0, num_layers - 2, 4))
    test_layers.extend([num_layers - 2, num_layers - 1])
    test_layers = sorted(list(set(test_layers))) # remove duplicates just in case
    
    results_by_layer = {}
    
    print(f"\nRodando varredura para as camadas: {test_layers}")
    
    for l_idx in test_layers:
        print(f"Intervindo na Camada {l_idx}...")
        mean_survival = get_layer_texts(model, tokenizer, items, l_idx, layers_attr, v1, v2, theta_rad, R, device)
        print(f"  -> Taxa de Sobrevivência Semântica: {mean_survival:.2%}")
        results_by_layer[l_idx] = float(mean_survival)
        
    print("\nSalvando logs em JSON...")
    os.makedirs("results", exist_ok=True)
    with open("results/self_repair_sweep.json", "w") as f:
        json.dump({"survival_by_layer": results_by_layer}, f, indent=2)
        
    print("\nEXPERIMENTO CONCLUÍDO. Dados salvos em results/self_repair_sweep.json")

if __name__ == "__main__":
    main()
