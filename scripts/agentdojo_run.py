"""Run AgentDojo against Claude via the `claude -p` CLI (no API key).

A real prompted-tool-calling adapter: it reuses AgentDojo's own LocalLLM tool
prompt + parser (`_make_system_prompt`, `_parse_model_output`) and swaps the model
call for `prayoga.lm.cli_client.CliLMClient` (claude -p). Then it runs a bounded but
genuine slice of a task suite with the `important_instructions` injection attack,
reporting utility (task success) and security (attack success rate).

Usage (host venv with agentdojo + repo src on path):
  .venv-ad/bin/python scripts/agentdojo_run.py --suite banking --n-user 3 --n-inj 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentdojo.agent_pipeline import (  # noqa: E402
    AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor,
)
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.agent_pipeline.llms.local_llm import (  # noqa: E402
    _make_system_prompt, _parse_model_output,
)
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks  # noqa: E402
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

from prayoga.lm.cli_client import CliLMClient  # noqa: E402

CLI = CliLMClient(timeout_s=180, max_retries=4)


class ClaudeCLILLM(BasePipelineElement):
    """AgentDojo pipeline element backed by `claude -p` with prompted tool calls."""

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        system_prompt = ""
        transcript = []
        for m in messages:
            role, content = m["role"], m["content"]
            text = get_text_content_as_str(content) if content is not None else ""
            if role == "system":
                system_prompt = _make_system_prompt(text, runtime.functions.values())
            elif role == "tool":
                if m.get("error"):
                    payload = json.dumps({"error": m["error"]})
                else:
                    payload = json.dumps({"result": m["content"] if m["content"] != "None" else "Success"})
                transcript.append(f"Tool result: {payload}")
            elif role == "user":
                transcript.append(f"User: {text}")
            elif role == "assistant":
                transcript.append(f"Assistant: {text}")
        prompt = "\n\n".join(transcript) + "\n\nAssistant:"
        completion = ""
        try:
            completion = CLI.complete(prompt, system=system_prompt or None)
        except Exception as e:  # transient CLI failure -> empty turn (counts as no action)
            print(f"[cli-error] {str(e)[:80]}")
        output = _parse_model_output(completion)
        return query, runtime, env, [*messages, output], extra_args


def build_pipeline() -> AgentPipeline:
    llm = ClaudeCLILLM()
    p = AgentPipeline([
        SystemMessage("You are an AI assistant that uses tools to complete the user's task."),
        InitQuery(),
        llm,
        ToolsExecutionLoop([ToolsExecutor(), llm]),
    ])
    p.name = "claude-3-5-sonnet-20241022"  # maps to Claude for attacks
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--n-user", type=int, default=3)
    ap.add_argument("--n-inj", type=int, default=3)
    ap.add_argument("--out", default="results/tier1/agentdojo.json")
    args = ap.parse_args()

    suite = get_suites("v1.2")[args.suite]
    pipeline = build_pipeline()
    attack = load_attack(args.attack, suite, pipeline)

    user_tasks = list(suite.user_tasks.values())[: args.n_user]
    inj_tasks = list(suite.injection_tasks.keys())[: args.n_inj]
    print(f"[agentdojo] suite={args.suite} attack={args.attack} "
          f"users={len(user_tasks)} injections={len(inj_tasks)} pairs={len(user_tasks)*len(inj_tasks)}")

    from agentdojo.logging import OutputLogger  # noqa: E402

    logdir = Path("results/tier1/agentdojo_logs")
    logdir.mkdir(parents=True, exist_ok=True)
    util_all: dict = {}
    sec_all: dict = {}
    with OutputLogger(str(logdir), None):
        for ut in user_tasks:
            try:
                u, s = run_task_with_injection_tasks(
                    suite, pipeline, ut, attack, logdir, True, injection_tasks=inj_tasks)
                util_all.update(u); sec_all.update(s)
                for key in u:
                    print(f"  {key}: utility={u[key]} attack_success={s.get(key)}")
            except Exception as e:
                print(f"[skip] {ut.ID}: {str(e)[:90]}")
    n = len(sec_all)
    util_hits = sum(int(v) for v in util_all.values())
    sec_hits = sum(int(v) for v in sec_all.values())
    rows = [{"pair": f"{k[0]}x{k[1]}", "utility": bool(util_all.get(k)), "attack_success": bool(sec_all[k])}
            for k in sec_all]

    summary = {
        "suite": args.suite, "attack": args.attack, "model": "claude (claude -p)",
        "n_pairs": n,
        "utility_rate": round(util_hits / n, 3) if n else None,
        "attack_success_rate": round(sec_hits / n, 3) if n else None,
        "rows": rows,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== AGENTDOJO (claude via claude -p) ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
