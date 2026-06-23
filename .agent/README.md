# `.agent/` — cross-surface task queue

Shared, git-tracked task state for the two execution surfaces that touch this repo.
The repo on disk is the **only** state both surfaces share, so the queue lives here.

| Surface | Runtime | MPS? | Role in this queue |
|---|---|---|---|
| **Cowork** | device-local Linux VM (folder mounted) | **no** | plans, researches, drafts tasks → **writes** to `queue/` |
| **Code tab / `antigravity` CLI** | bare host macOS | **yes** | implements, runs inference, git → **executes** tasks |

There is no live API bridge between the two. This directory is the bridge.

## Layout — the directory *is* the status
```
.agent/
  README.md        # this file — the protocol
  queue/           # ready to pick up
  active/          # claimed by an executor (the move here IS the claim)
  done/            # finished, kept for audit
  blocked/         # waiting on a human or a dependency
```

One task = one file `NNN-slug.md`. `NNN` is a zero-padded monotonic id and is the
**stable identity**: it never changes when the file moves between directories.

## Task file format

```markdown
---
id: 007
title: <short imperative>
status: queue            # mirrors the directory; directory wins on conflict
target: code             # code | cowork | either — who must run it
created_by: cowork       # cowork | code | human
created_at: 2026-06-11
depends_on: []           # list of ids that must reach done/ first
branch: phase6-probing   # optional, for code tasks
---

## Goal
What outcome is wanted, in one or two sentences.

## Context / pointers
Files, prior results, doc links the executor needs. No re-discovery.

## Done when
Explicit, checkable conditions. The task is not done until these are literally true.

## Log
- 2026-06-11 cowork: created
```

## Protocol

**Create** (usually Cowork)
1. Pick the next free `NNN`.
2. Write `queue/NNN-slug.md`. Fill *Goal*, *Context*, and a concrete *Done when*.
3. Set `target:`. Anything needing MPS, host bash, or git → `target: code`.
4. Commit.

**Claim + execute** (usually Code)
1. Move `queue/NNN-slug.md` → `active/`. This move is the atomic claim — one executor per task.
2. Do the work. Append dated lines to `## Log`.
3. Success → move to `done/`, set `status: done`. Stuck → move to `blocked/`, write why.
4. Commit, referencing the task id in the message.

## CLI — `scripts/agentq.py`

Thin wrapper (typer + rich, no new deps) so you don't move files by hand.
Implementation: `src/gemma4_lab/agentq.py`. After `pip install -e .` on the host
it's the `agentq` command; with no install (e.g. Cowork VM) use the shim
`python scripts/agentq.py`. Both forms are equivalent.

```bash
agentq list                                  # the board
agentq new "Run E4B audit" --target code [--depends-on 1 --branch phase6-probing]
agentq claim 1                               # queue/ -> active/
agentq done 1                                # -> done/
agentq block 1 --reason "waiting on 002"
agentq show 1
# no-install equivalent: python scripts/agentq.py list
```

`created_by` and log lines are auto-tagged by surface (`Linux` → cowork, `Darwin` → code).
Moves use an atomic `rename` — a failed move never leaves a duplicate.

### Surface caveat (verified 2026-06-11)

The Cowork repo mount is **create-only**: the VM can write new files but cannot
delete or move them on the host folder (deletion is gated behind a separate
approval). So from Cowork, `new` works but `claim`/`done`/`block` will fail
cleanly with a message — which is correct, since **execution belongs on the Code
host** anyway. Run the move-commands from the Code tab / `antigravity` CLI.

## Rules

- **One executor per task.** The move into `active/` is the lock. Don't touch a file already in `active/` unless you put it there.
- **Cowork must not claim `target: code` tasks** — it has no MPS or host bash. It may draft, refine, and research them only.
- **`status:` mirrors the directory.** Keep them in sync; if they disagree, the directory is authoritative.
- **No silent partials.** A task leaves `active/` only when its *Done when* holds or it is explicitly `blocked/`.
- Keep everything in git. The transitions are the audit trail.
