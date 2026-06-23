"""Ad-hoc Gemma 4 generation from the command line.

    python scripts/infer.py "Explain attention in one paragraph"
    python scripts/infer.py "Solve: 17 * 23" --thinking
    python scripts/infer.py "Hi" --model e4b --max-tokens 64
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from gemma4_lab.config import Settings
from gemma4_lab.inference.hf_local import GemmaLocal
from gemma4_lab.observability import setup as setup_observability

app = typer.Typer(add_completion=False, help="Run a one-shot Gemma 4 generation.")
console = Console()


@app.command()
def main(
    prompt: Annotated[str, typer.Argument(help="The user prompt.")],
    model: Annotated[str, typer.Option(help="Model variant: e2b or e4b.")] = "e2b",
    thinking: Annotated[bool, typer.Option("--thinking/--no-thinking")] = False,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="max_new_tokens.")] = 512,
    system: Annotated[str | None, typer.Option(help="Optional system message.")] = None,
) -> None:
    if model not in ("e2b", "e4b"):
        raise typer.BadParameter("--model must be 'e2b' or 'e4b'")

    settings = Settings(model_variant=model)  # type: ignore[arg-type]
    setup_observability()

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    console.print(f"[dim]Loading {settings.model_id} (thinking={thinking}, device={settings.device})…[/dim]")
    runner = GemmaLocal(settings)
    result = runner.generate(messages=messages, thinking=thinking, max_new_tokens=max_tokens)

    if result.thought:
        console.print(Panel(result.thought, title="thought", border_style="dim", style="dim"))
    console.print(Panel(result.text, title=f"answer · {settings.model_id}", border_style="cyan"))
    console.print(
        f"[dim]in={result.input_tokens}  out={result.output_tokens}  "
        f"latency={result.latency_ms:.0f} ms[/dim]"
    )


if __name__ == "__main__":
    app()
