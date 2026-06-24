"""CliLMClient — Tier-1 Claude substrate via the `claude` CLI (print mode).

The program has no Anthropic API key, so every black-box (Tier-1) Claude call is
routed through the `claude -p` command, inheriting the local Claude Code auth.
This keeps Tier-1 dependency-free (no SDK) and reproducible.

Claim-tier: this module is plumbing for the MECHANISM/behavioral (Axis A, Tier-1)
measurements; it makes no claim itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field


class CliLMError(RuntimeError):
    """Raised when the `claude` CLI invocation fails after retries."""


@dataclass
class CliLMClient:
    """Minimal wrapper around `claude -p` for non-interactive completions.

    Parameters
    ----------
    model:
        Pinned model id passed to ``--model`` (pin for reproducibility). If None,
        the CLI default is used.
    binary:
        Path/name of the Claude CLI (default ``claude``).
    timeout_s:
        Per-call timeout in seconds.
    max_retries:
        Retries on non-zero exit / timeout, with exponential backoff.
    extra_args:
        Additional CLI flags appended verbatim.
    """

    model: str | None = None
    binary: str = "claude"
    timeout_s: float = 120.0
    max_retries: int = 3
    backoff_s: float = 2.0
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if shutil.which(self.binary) is None:
            raise CliLMError(
                f"Claude CLI '{self.binary}' not found on PATH. "
                "Tier-1 requires the `claude` CLI (no API key configured)."
            )

    def _build_cmd(self, prompt: str, system: str | None) -> list[str]:
        cmd = [self.binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if system:
            cmd += ["--append-system-prompt", system]
        cmd += self.extra_args
        return cmd

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the assistant text for a single prompt.

        Parses the CLI's JSON envelope and returns the ``result`` text. Raises
        CliLMError after exhausting retries.
        """
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                proc = subprocess.run(
                    self._build_cmd(prompt, system),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                if proc.returncode != 0:
                    raise CliLMError(
                        f"claude CLI exit {proc.returncode}: {proc.stderr.strip()[:500]}"
                    )
                return self._parse(proc.stdout)
            except (subprocess.TimeoutExpired, CliLMError) as err:
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (2**attempt))
        raise CliLMError(f"claude CLI failed after {self.max_retries} attempts: {last_err}")

    @staticmethod
    def _parse(stdout: str) -> str:
        stdout = stdout.strip()
        if not stdout:
            raise CliLMError("empty CLI output")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            # `--output-format json` not honored by this CLI version: treat as raw text.
            return stdout
        if isinstance(payload, dict):
            for key in ("result", "completion", "text", "response"):
                if isinstance(payload.get(key), str):
                    return payload[key]
        return stdout
