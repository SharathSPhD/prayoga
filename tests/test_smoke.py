"""Phase-0 smoke tests: package imports + CliLMClient parsing.

Exit-gate check for WP0.1/WP0.3 — confirms the monorepo imports and the Tier-1
substrate parses CLI output without requiring a live `claude` call.
"""

from __future__ import annotations

import prayoga
from prayoga.lm.cli_client import CliLMClient


def test_version() -> None:
    assert prayoga.__version__ == "0.0.1"


def test_subpackages_import() -> None:
    import importlib

    for mod in ("shared", "lm", "axis_a", "axis_b", "axis_c", "benchmarks"):
        importlib.import_module(f"prayoga.{mod}")


def test_cli_parse_json_envelope() -> None:
    assert CliLMClient._parse('{"result": "hello"}') == "hello"
    assert CliLMClient._parse('{"completion": "hi"}') == "hi"


def test_cli_parse_raw_text_fallback() -> None:
    assert CliLMClient._parse("not json, raw text") == "not json, raw text"


def test_cli_lean_flags_present_by_default() -> None:
    c = CliLMClient.__new__(CliLMClient)  # skip __post_init__ PATH check
    c.model = None
    c.lean = True
    c.extra_args = []
    cmd = c._build_cmd("hi", None)
    assert "--strict-mcp-config" in cmd
    assert "--setting-sources" in cmd
    # lean disabled => no MCP-strip flags
    c.lean = False
    assert "--strict-mcp-config" not in c._build_cmd("hi", None)
