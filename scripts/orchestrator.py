import sys
import os
import subprocess
import time
from pathlib import Path

# Garantir visibilidade do pacote local
sys.path.append(str(Path(__file__).parent.parent / "src"))
from gemma4_physio.config import PipelineConfig

def main():
    config_path = Path("pipeline_config.yaml")
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        sys.exit(1)
    
    config = PipelineConfig.load_from_yaml(config_path)
    sweep_cfg = config.pipelines.get("topological_sweep")
    if not sweep_cfg or not sweep_cfg.get("enabled"):
        print("Topological sweep pipeline disabled in configuration.", flush=True)
        return
        
    K = sweep_cfg["control_planes_K"]
    
    print(f"--- Orchestrator Started: {config.experiment_name} ---")
    print(f"Will process {K} planes incrementally.")
    
    # Criar diretório de checkpoints
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Path para os scripts
    script_dir = Path(__file__).parent
    generate_script = script_dir / "generate_points.py"
    tda_script = script_dir / "compute_tda.py"
    
    for k in range(K):
        checkpoint_file = checkpoint_dir / f"plane_{k}.npy"
        if checkpoint_file.exists():
            print(f"[*] Plane {k} already completed. Skipping.")
            continue
            
        print(f"\n[+] Launching process for Plane {k}...")
        try:
            # Chama o script gerador como um subprocesso separado
            env = os.environ.copy()
            # Define o KMP_DUPLICATE_LIB_OK=TRUE caso não esteja setado (muito importante p/ macOS MPS)
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            result = subprocess.run(
                [sys.executable, str(generate_script), "--plane", str(k)],
                env=env,
                check=True
            )
            print(f"[+] Process for Plane {k} finished successfully.")
            
            # Dar um tempinho extra (sleep) para o OS limpar o Swap e a GPU resfriar
            print("Cooling down OS memory/swap for 5 seconds...")
            time.sleep(5)
            
        except subprocess.CalledProcessError as e:
            print(f"[-] Error: Plane {k} generation failed. Stopping orchestrator.")
            sys.exit(1)
            
    print("\n--- All planes generated! ---")
    print("Launching TDA computation step...")
    
    # Roda o cálculo do TDA no final
    try:
        subprocess.run([sys.executable, str(tda_script)], check=True)
        print("Pipeline orchestrator completed successfully!")
    except subprocess.CalledProcessError:
        print("Error during TDA computation.")

if __name__ == "__main__":
    main()
