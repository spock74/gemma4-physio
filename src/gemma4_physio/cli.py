"""
Copyright (c) 2026 Jose E Moraes. All rights reserved.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import typer
import yaml
from pathlib import Path

app = typer.Typer(help="Gemma4-Physio CLI Utility")
config_app = typer.Typer(help="Gerenciar configurações do projeto")
app.add_typer(config_app, name="config")

CONFIG_PATH = Path("pipeline_config.yaml")

def load_yaml():
    if not CONFIG_PATH.exists():
        typer.secho(f"Erro: Arquivo de configuração '{CONFIG_PATH}' não encontrado.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

@config_app.command("set-logfire")
def set_logfire(
    enable: bool = typer.Option(..., "--enable/--disable", help="Ativar ou desativar o envio de telemetria para o Logfire")
):
    """
    Ativar ou desativar o envio de telemetria para o Logfire no pipeline_config.yaml.
    """
    data = load_yaml()
        
    if "observability" not in data:
        data["observability"] = {}
        
    data["observability"]["send_to_logfire"] = enable
    save_yaml(data)
    
    status = "ATIVADO" if enable else "DESATIVADO"
    color = typer.colors.GREEN if enable else typer.colors.YELLOW
    typer.secho(f"[✓] Logfire 'send_to_logfire' foi {status} no arquivo {CONFIG_PATH}.", fg=color)

run_app = typer.Typer(help="Executar pipelines do projeto")
app.add_typer(run_app, name="run")

def init_model(device="mps"):
    import torch
    import fcntl
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    from gemma4_physio.observability import setup_logfire
    
    # [TRAVA DE SEGURANÇA CONTRA OOM]
    lock_file_path = "/tmp/gemma4_physio_model.lock"
    lock_file = open(lock_file_path, 'a+')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"PID: {os.getpid()} | CMD: {' '.join(sys.argv)}")
        lock_file.flush()
    except BlockingIOError:
        lock_file.seek(0)
        process_info = lock_file.read().strip()
        typer.secho("\n[!!!] ERRO CRÍTICO DE SEGURANÇA [!!!]", fg=typer.colors.RED, bold=True)
        typer.secho(f"Outro experimento já está rodando. Detalhes: [{process_info}]", fg=typer.colors.YELLOW)
        typer.secho("Isso esgotaria a memória do seu Mac e causaria um congelamento do sistema.", fg=typer.colors.RED)
        typer.secho("Execução abortada pelo sistema de proteção Anti-OOM.\n", fg=typer.colors.RED)
        sys.exit(1)
        
    setup_logfire()
    model_id = "google/gemma-3-4b-it"
    dtype = torch.bfloat16
    
    typer.secho(f"Carregando {model_id} em {device}...", fg=typer.colors.CYAN)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map=device, 
        attn_implementation="eager"
    ).eval()
    return model, tokenizer, device

@run_app.command("identity")
def run_identity_cmd():
    from gemma4_physio.pipelines.subspace_identity import run_subspace_identity
    data = load_yaml()
    cfg = data.get("pipelines", {}).get("subspace_identity", {})
    if not cfg.get("enabled", False):
        typer.secho("Pipeline desabilitado no YAML.", fg=typer.colors.RED)
        raise typer.Exit(1)
    model, tokenizer, device = init_model()
    run_subspace_identity(cfg, model, tokenizer, device)

@run_app.command("scrubbing")
def run_scrubbing_cmd():
    from gemma4_physio.pipelines.causal_scrubbing import run_causal_scrubbing
    data = load_yaml()
    cfg = data.get("pipelines", {}).get("causal_scrubbing", {})
    if not cfg.get("enabled", False):
        typer.secho("Pipeline desabilitado no YAML.", fg=typer.colors.RED)
        raise typer.Exit(1)
    model, tokenizer, device = init_model()
    run_causal_scrubbing(cfg, model, tokenizer, device)

@run_app.command("freeman")
def run_freeman_cmd():
    from gemma4_physio.pipelines.freeman_stabilization import run_freeman_stabilization
    data = load_yaml()
    cfg = data.get("pipelines", {}).get("freeman_stabilization", {})
    if not cfg.get("enabled", False):
        typer.secho("Pipeline desabilitado no YAML.", fg=typer.colors.RED)
        raise typer.Exit(1)
    model, tokenizer, device = init_model()
    run_freeman_stabilization(cfg, model, tokenizer, device)

@run_app.command("all")
def run_all_cmd():
    from gemma4_physio.pipelines.subspace_identity import run_subspace_identity
    from gemma4_physio.pipelines.causal_scrubbing import run_causal_scrubbing
    from gemma4_physio.pipelines.freeman_stabilization import run_freeman_stabilization
    
    data = load_yaml()
    cfg_identity = data.get("pipelines", {}).get("subspace_identity", {})
    cfg_scrubbing = data.get("pipelines", {}).get("causal_scrubbing", {})
    cfg_freeman = data.get("pipelines", {}).get("freeman_stabilization", {})

    pipelines_to_run = []
    if cfg_identity.get("enabled", False): pipelines_to_run.append(("Identidade de Subespaço", run_subspace_identity, cfg_identity))
    if cfg_scrubbing.get("enabled", False): pipelines_to_run.append(("Causal Scrubbing", run_causal_scrubbing, cfg_scrubbing))
    if cfg_freeman.get("enabled", False): pipelines_to_run.append(("Estabilização de Freeman", run_freeman_stabilization, cfg_freeman))

    if not pipelines_to_run:
        typer.secho("Nenhum pipeline está habilitado no arquivo pipeline_config.yaml.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"\nPlanejados {len(pipelines_to_run)} testes. Carregando modelo na memória (Isso ocorre apenas uma vez)...", fg=typer.colors.CYAN, bold=True)
    model, tokenizer, device = init_model()

    for i, (name, func, cfg) in enumerate(pipelines_to_run):
        typer.secho(f"\n{'-'*50}", fg=typer.colors.CYAN)
        typer.secho(f"[{i+1}/{len(pipelines_to_run)}] PRONTO PARA INICIAR: {name}", fg=typer.colors.YELLOW, bold=True)
        if typer.confirm("Deseja prosseguir com a execução deste pipeline agora?", default=True):
            func(cfg, model, tokenizer, device)
            typer.secho(f"\n[✓] Execução do pipeline {name} concluída com sucesso.", fg=typer.colors.GREEN, bold=True)
            if i < len(pipelines_to_run) - 1:
                import torch
                if device == "mps":
                    torch.mps.empty_cache()
        else:
            typer.secho(f"Pulando {name}...", fg=typer.colors.YELLOW)
            
    typer.secho(f"\n{'-'*50}", fg=typer.colors.CYAN)
    typer.secho("Todos os testes planejados foram processados.", fg=typer.colors.GREEN, bold=True)

if __name__ == "__main__":
    app()
