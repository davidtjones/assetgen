"""Ask an agent one narrow question and get its answer in a fixed shape.

A tool runs its own step and calls an agent only where the step needs judgment: reading a
painting, choosing between candidates. The question is one task with a JSON schema for the
answer; the tool does the rest with it. This keeps the agent's job small and lets the tool,
not the agent, own the step's folder.

    from common.agent import ask
    answer = ask("Look at crop.png. What is the roof made of?", schema, record=step_dir / "ask",
                 files=[step_dir / "crop.png"])

`ask` copies `files` into a fresh scratch folder, runs the agent there with only the tools it
is given (default: reading files), and returns the answer as a dict checked against the
schema's required keys. It writes `record/question.md`, `record/answer.json` and
`record/run.json` (the agent's model, cost and time), so the step's folder shows what was
asked and what came back.

The only backend today is Claude Code run headless (`claude -p`), on the login of whoever runs
the tool. `ANTHROPIC_API_KEY` is taken out of its environment: with it set, `claude -p` would
bill the API instead of the subscription. The scratch folder sits outside the workspace, so the
agent loads no project instructions and does nothing but answer. Another backend (an API model,
a local one) goes in `BACKENDS` with the same signature.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Default model for a narrow question: quick, and good at reading pictures.
MODEL = "sonnet"


class AgentError(RuntimeError):
    pass


def _claude(question: str, schema: dict, cwd: Path, model: str, tools: tuple[str, ...], timeout: float) -> tuple[dict, dict]:
    """One `claude -p` run in `cwd`: the structured answer, and what the run reports about itself."""
    env = {k: v for k, v in os.environ.items()
           if k != "ANTHROPIC_API_KEY" and not k.startswith("CLAUDE_CODE_") and k not in ("CLAUDECODE", "CLAUDE_PID")}
    cmd = ["claude", "-p", question, "--output-format", "json", "--json-schema", json.dumps(schema),
           "--model", model, "--tools", ",".join(tools), "--allowedTools", ",".join(tools),
           "--strict-mcp-config", "--no-session-persistence"]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise AgentError(f"claude -p exited {proc.returncode} without JSON: {(proc.stderr or proc.stdout)[-2000:]}")
    if out.get("is_error") or "structured_output" not in out:
        raise AgentError(f"claude -p gave no structured answer ({out.get('subtype')}): {str(out.get('result'))[-2000:]}")
    run = {k: out.get(k) for k in ("total_cost_usd", "duration_ms", "num_turns", "session_id")}
    return out["structured_output"], run


BACKENDS = {"claude": _claude}


def ask(question: str, schema: dict, record: Path, *, files: list[Path] = (), model: str = MODEL,
        tools: tuple[str, ...] = ("Read",), backend: str = "claude", timeout: float = 600) -> dict:
    """Ask `question`, answered as JSON matching `schema`, with `files` beside it in the agent's
    folder (named as they are). Records the question and answer in `record/`."""
    record.mkdir(parents=True, exist_ok=True)
    (record / "question.md").write_text(question + "\n\n## Answer schema\n\n```json\n" + json.dumps(schema, indent=2) + "\n```\n")
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="assetgen-ask-") as tmp:
        for f in files:
            shutil.copyfile(f, Path(tmp) / Path(f).name)
        answer, run = BACKENDS[backend](question, schema, Path(tmp), model, tools, timeout)
    missing = [k for k in schema.get("required", []) if k not in answer]
    if missing:
        raise AgentError(f"the answer lacks {missing}: {answer}")
    (record / "answer.json").write_text(json.dumps(answer, indent=2) + "\n")
    run.update(backend=backend, model=model, files=[Path(f).name for f in files], seconds=round(time.time() - start, 1))
    (record / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return answer
