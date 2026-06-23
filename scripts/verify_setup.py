"""End-to-end smoke test for gemma4-lab.

Run:
    python scripts/verify_setup.py

Checks (in order, each independent — one failure does not abort the rest):
  1. MPS available
  2. Settings load + secret constants present
  3. Logfire bootstrap
  4. Gemma 4 E2B load + one-shot generation (downloads weights on first run)
  5. Gemini one-shot ping

Exits 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

console = Console()
results: list[tuple[str, bool, str]] = []


def _require_project_env(expected: str = "gemma4-lab") -> None:
    """Abort early with a clear message if running under the wrong interpreter.

    Importing torch from a foreign conda env can crash the process with a
    cryptic OpenMP "libomp.dylib already initialized" abort. This check turns
    that into a readable error before any heavy import happens.
    """
    if expected not in sys.prefix:
        sys.stderr.write(
            f"\n  gemma4-lab: wrong Python environment.\n"
            f"  Active interpreter : {sys.executable}\n"
            f"  Expected the '{expected}' conda env.\n"
            f"  Fix:  conda activate {expected}\n\n"
        )
        raise SystemExit(1)


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    console.print(f"  {icon}  [bold]{name}[/bold]  —  {detail}")


def check_mps() -> None:
    try:
        import torch

        ok = bool(torch.backends.mps.is_available())
        built = bool(torch.backends.mps.is_built())
        detail = f"available={ok}, built={built}, torch={torch.__version__}"
        record("MPS", ok, detail)
    except Exception as e:
        record("MPS", False, f"error: {e!r}")


def check_settings() -> object | None:
    try:
        from gemma4_lab.config import (
            GEMINI_API_KEY,
            HF_TOKEN,
            LOGFIRE_TOKEN,
            Settings,
        )

        s = Settings()
        present = [
            label
            for label, val in [
                ("GEMINI_API_KEY", GEMINI_API_KEY),
                ("LOGFIRE_TOKEN", LOGFIRE_TOKEN),
                ("HF_TOKEN", HF_TOKEN),
            ]
            if val
        ]
        record(
            "Settings",
            True,
            f"variant={s.model_variant}, model_id={s.model_id}, secrets present: {present or 'none'}",
        )
        return s
    except Exception as e:
        record("Settings", False, f"error: {e!r}")
        return None


def check_logfire() -> None:
    try:
        from gemma4_lab.config import LOGFIRE_TOKEN
        from gemma4_lab.observability import setup

        setup()
        import logfire

        with logfire.span("verify_setup.smoke"):
            logfire.info("verify_setup_running")
        record(
            "Logfire",
            True,
            f"configured (token={'yes' if LOGFIRE_TOKEN else 'local-only'})",
        )
    except Exception as e:
        record("Logfire", False, f"error: {e!r}")


def check_gemma(settings: object | None) -> None:
    if settings is None:
        record("Gemma E2B", False, "skipped (settings unavailable)")
        return
    try:
        from gemma4_lab.config import Settings
        from gemma4_lab.inference.hf_local import GemmaLocal

        # Force E2B for the smoke test even if env says otherwise — smaller, faster.
        e2b_settings = Settings(model_variant="e2b")  # type: ignore[arg-type]
        runner = GemmaLocal(e2b_settings)
        result = runner.generate(
            messages=[{"role": "user", "content": "Reply in 5 words: how are you?"}],
            thinking=False,
            max_new_tokens=32,
        )
        snippet = result.text.replace("\n", " ")[:80]
        record(
            "Gemma E2B",
            True,
            f"in={result.input_tokens} out={result.output_tokens} {result.latency_ms:.0f}ms — “{snippet}”",
        )
    except Exception as e:
        record("Gemma E2B", False, f"error: {e!r}")


def check_gemini() -> None:
    try:
        from gemma4_lab.synthetic_data.gemini_client import GeminiClient

        client = GeminiClient()
        text = client.generate("Reply with the single word OK.")
        snippet = text.replace("\n", " ").strip()[:80]
        record("Gemini", bool(snippet), f"reply: “{snippet}”")
    except Exception as e:
        record("Gemini", False, f"error: {e!r}")


def render_summary() -> int:
    table = Table(title="gemma4-lab — verify_setup", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Result")
    table.add_column("Detail", overflow="fold")

    failures = 0
    for name, ok, detail in results:
        result_cell = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, result_cell, detail)
        if not ok:
            failures += 1

    console.print()
    console.print(table)
    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/red]")
        return 1
    console.print("\n[green]All checks passed.[/green]")
    return 0


def main() -> int:
    _require_project_env()
    console.rule("[bold]gemma4-lab verify_setup")
    check_mps()
    settings = check_settings()
    check_logfire()
    check_gemma(settings)
    check_gemini()
    return render_summary()


if __name__ == "__main__":
    sys.exit(main())
