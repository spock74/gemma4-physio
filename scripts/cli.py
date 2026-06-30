#!/usr/bin/env python3
import typer
import yaml
from pathlib import Path

app = typer.Typer(help="Gemma4-Physio Mini CLI Utility")
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

if __name__ == "__main__":
    app()
