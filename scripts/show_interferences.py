import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gemma4_physio.data_loader import PopQASampler
from gemma4_physio.spps_hooks import spps_rotational_hook
from gemma4_physio.observability import setup_logfire, logfire_memory_span

def load_data():
    print("Carregando dataset para separar exemplos...")
    with open("data/popqa/popqa_full.json", "r") as f:
        data = json.load(f)
        
    examples = {
        "author": [],
        "capital": [],
        "sport": [],
        "director": []
    }
    
    for item in data:
        rel = item.get("relation")
        if rel in examples and len(examples[rel]) < 2:
            examples[rel].append(item)
    return examples

def main():
    examples = load_data()
    model_id = "google/gemma-3-4b-it"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Carregando {model_id} em {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    ).eval()
    
    # Preparar Vetores do SPPS
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
    
    print("\n" + "="*60)
    print("GERAÇÃO DE EXEMPLOS: LIMPO vs COM ROTAÇÃO SPPS (90 graus)")
    print("="*60)
    
    for rel_class, items in examples.items():
        print(f"\n--- CLASSE SEMÂNTICA: {rel_class.upper()} ---")
        for item in items:
            question = item["question"]
            answer = eval(item["answer"])[0] if isinstance(item["answer"], str) and item["answer"].startswith("[") else item["answer"]
            
            prompt = f"Answer the following question succinctly.\nQuestion: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print(f"PROMPT (Limpo): {prompt}")
            with logfire_memory_span("Clean Interference Gen"):
                with torch.no_grad():
                    out_clean = model.generate(**inputs, max_new_tokens=15, do_sample=False, use_cache=True)
            text_clean = tokenizer.decode(out_clean[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"RESPOSTA LIMPA: {text_clean}")
            
            # Geração com Interferência Rotacional
            with spps_rotational_hook(target_layer, v1, v2, theta_rad, R):
                with torch.no_grad():
                    out_rot = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            text_rot = tokenizer.decode(out_rot[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            print(f"\nPrompt: {question} (Resposta Esperada: {answer})")
            print(f"-> Geração Limpa (Baseline) : {text_clean}")
            print(f"-> Geração com SPPS (90º)   : {text_rot}")

if __name__ == "__main__":
    setup_logfire()
    main()
