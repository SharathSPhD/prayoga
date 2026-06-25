"""Validate the local Cursor/Claude plugin metadata and command files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"
MANIFEST = PLUGIN / ".claude-plugin" / "plugin.json"
COMMANDS = PLUGIN / "commands"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("name"):
        raise SystemExit("plugin manifest missing name")
    if manifest.get("name") != "prayoga":
        raise SystemExit(f"unexpected plugin name: {manifest.get('name')}")

    command_paths = [p for p in COMMANDS.glob("*.md") if not p.name.startswith("._")]
    commands = sorted(p.stem for p in command_paths)
    required = {
        "active",
        "agentdojo",
        "dimensionality",
        "dose",
        "ec50-scaling",
        "refusal",
        "satkarma",
        "status",
        "symmetry",
        "trajectory-probe",
        "triangulation",
    }
    missing = sorted(required - set(commands))
    if missing:
        raise SystemExit(f"missing plugin commands: {', '.join(missing)}")

    readme = (PLUGIN / "README.md").read_text(encoding="utf-8")
    undocumented = [cmd for cmd in required if f"/prayoga:{cmd}" not in readme]
    if undocumented:
        raise SystemExit(f"plugin README omits commands: {', '.join(sorted(undocumented))}")

    for path in command_paths:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise SystemExit(f"command missing frontmatter: {path.relative_to(ROOT)}")
        if "description:" not in text:
            raise SystemExit(f"command missing description: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
