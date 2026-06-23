"""agentq — thin CLI over the `.agent/` cross-surface task queue.

The directory *is* the status: queue/ -> active/ -> done/ (or blocked/).
One task per `NNN-slug.md` file; `NNN` is the stable id. See `.agent/README.md`.

Pure stdlib + typer/rich (already locked deps), no MPS/torch dependency, so it
runs identically in the Cowork VM and on the Code host. Importing this module
does NOT pull the ML stack (`gemma4_lab/__init__.py` only sets env vars).

Invocation:
    agentq list                       # after `pip install -e .` on the host
    python scripts/agentq.py list     # no install needed (e.g. Cowork VM)

    agentq new "Run E4B audit" --target code
    agentq claim 1
    agentq done 1
    agentq block 1 --reason "waiting on 002"
    agentq show 1
"""
from __future__ import annotations

import platform
import re
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

STATUSES = ("queue", "active", "done", "blocked")
_STATUS_STYLE = {"queue": "yellow", "active": "cyan", "done": "green", "blocked": "red"}


def detect_surface() -> str:
    """Best-effort guess of which surface is running this. Linux -> Cowork VM,
    Darwin -> Code host. Used only as a default for created_by / log lines."""
    system = platform.system()
    if system == "Linux":
        return "cowork"
    if system == "Darwin":
        return "code"
    return "human"


def repo_root() -> Path:
    """Walk up from this file (and cwd) to find the dir containing `.agent/`."""
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for d in (start, *start.parents):
            if (d / ".agent").is_dir():
                return d
    raise typer.BadParameter("Could not locate a `.agent/` directory above cwd or this script.")


def agent_dir() -> Path:
    return repo_root() / ".agent"


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s or "task"


# --- tiny frontmatter (deliberately not a YAML dep) -------------------------

def parse_task(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body.lstrip("\n")


def dump_task(meta: dict[str, str], body: str) -> str:
    fm = "\n".join(f"{k}: {v}" for k, v in meta.items())
    return f"---\n{fm}\n---\n\n{body.rstrip()}\n"


def find_task(task_id: int) -> tuple[Path, str]:
    """Return (path, status) for a task id, searching all status dirs."""
    tag = f"{task_id:03d}-"
    for status in STATUSES:
        for p in (agent_dir() / status).glob(f"{tag}*.md"):
            return p, status
    raise typer.BadParameter(f"No task with id {task_id:03d} found in .agent/.")


def next_id() -> int:
    ids = [
        int(m.group(1))
        for status in STATUSES
        for p in (agent_dir() / status).glob("*.md")
        if (m := re.match(r"(\d+)-", p.name))
    ]
    return (max(ids) + 1) if ids else 1


def _append_log(body: str, msg: str) -> str:
    line = f"- {date.today().isoformat()} {detect_surface()}: {msg}"
    if "## Log" in body:
        return body.rstrip() + "\n" + line + "\n"
    return body.rstrip() + "\n\n## Log\n" + line + "\n"


def _move(task_id: int, to_status: str, log_msg: str) -> Path:
    path, cur = find_task(task_id)
    dest = agent_dir() / to_status / path.name
    if dest != path:
        try:
            path.rename(dest)  # atomic; raises cleanly without leaving a duplicate
        except OSError as e:
            raise typer.BadParameter(
                f"Could not move the task file ({e.strerror}). On Cowork the repo mount is "
                "create-only — run claim/done/block from the Code host instead."
            ) from e
    meta, body = parse_task(dest)
    meta["status"] = to_status
    dest.write_text(dump_task(meta, _append_log(body, log_msg)), encoding="utf-8")
    return dest


# --- commands ----------------------------------------------------------------

@app.command()
def new(
    title: str = typer.Argument(..., help="Short imperative title."),
    target: str = typer.Option("code", help="code | cowork | either — who must run it."),
    depends_on: str = typer.Option("", help="Comma-separated ids this task waits on."),
    branch: str = typer.Option("", help="Optional git branch for code tasks."),
):
    """Create a new task in queue/."""
    if target not in ("code", "cowork", "either"):
        raise typer.BadParameter("target must be code | cowork | either")
    tid = next_id()
    deps = "[" + ", ".join(d.strip() for d in depends_on.split(",") if d.strip()) + "]"
    meta = {
        "id": f"{tid:03d}",
        "title": title,
        "status": "queue",
        "target": target,
        "created_by": detect_surface(),
        "created_at": date.today().isoformat(),
        "depends_on": deps,
    }
    if branch:
        meta["branch"] = branch
    body = (
        "## Goal\n<one or two sentences>\n\n"
        "## Context / pointers\n<files, prior results, doc links>\n\n"
        "## Done when\n<explicit, checkable conditions>\n\n"
        "## Log\n"
        f"- {date.today().isoformat()} {detect_surface()}: created\n"
    )
    path = agent_dir() / "queue" / f"{tid:03d}-{slugify(title)}.md"
    path.write_text(dump_task(meta, body), encoding="utf-8")
    console.print(f"[green]created[/] {path.relative_to(repo_root())}")


@app.command()
def claim(task_id: int):
    """Move a task queue/ -> active/ (the atomic claim)."""
    path, _ = find_task(task_id)
    meta, _ = parse_task(path)
    if detect_surface() == "cowork" and meta.get("target") == "code":
        raise typer.BadParameter(
            "This is a target:code task and you appear to be on Cowork (no MPS/host bash). "
            "Refine it here, but claim/execute from the Code tab."
        )
    dest = _move(task_id, "active", "claimed")
    console.print(f"[cyan]active[/] {dest.relative_to(repo_root())}")


@app.command()
def done(task_id: int):
    """Move a task -> done/."""
    dest = _move(task_id, "done", "done")
    console.print(f"[green]done[/] {dest.relative_to(repo_root())}")


@app.command()
def block(task_id: int, reason: str = typer.Option(..., help="Why it's blocked.")):
    """Move a task -> blocked/ with a reason."""
    dest = _move(task_id, "blocked", f"blocked: {reason}")
    console.print(f"[red]blocked[/] {dest.relative_to(repo_root())}")


@app.command(name="list")
def list_tasks(status: str = typer.Option("", help="Filter: queue|active|done|blocked.")):
    """Show the board."""
    wanted = [status] if status else list(STATUSES)
    table = Table(title=".agent queue")
    table.add_column("id", justify="right")
    table.add_column("status")
    table.add_column("target")
    table.add_column("title")
    table.add_column("deps")
    rows = 0
    for st in wanted:
        for p in sorted((agent_dir() / st).glob("*.md")):
            meta, _ = parse_task(p)
            table.add_row(
                meta.get("id", "?"),
                f"[{_STATUS_STYLE[st]}]{st}[/]",
                meta.get("target", "-"),
                meta.get("title", p.stem),
                meta.get("depends_on", "[]"),
            )
            rows += 1
    if rows == 0:
        console.print("[dim]no tasks[/]")
    else:
        console.print(table)


@app.command()
def show(task_id: int):
    """Print a task file."""
    path, _ = find_task(task_id)
    console.print(path.read_text(encoding="utf-8"))


def main() -> None:
    """Console-script entry point (`agentq`)."""
    app()


if __name__ == "__main__":
    main()
