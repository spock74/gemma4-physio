#!/usr/bin/env python3
"""Launcher for the `.agent/` queue CLI.

Real implementation lives in `src/gemma4_lab/agentq.py`. After `pip install -e .`
the `agentq` console command is the preferred entry point; this shim lets you run
the CLI with no install (e.g. from the Cowork VM) via `python scripts/agentq.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma4_lab.agentq import main  # noqa: E402

if __name__ == "__main__":
    main()
