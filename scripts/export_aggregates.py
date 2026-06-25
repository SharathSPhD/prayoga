"""Export public aggregate artifacts from canonical tracked sources.

Raw experiment outputs under ``results/`` are intentionally gitignored because
they may contain dual-use vectors or harmful generations. This script only
mirrors public aggregate JSON that is already safe to publish, then validates
the mirror files against the tier registry.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "findings_registry.json"

CANONICAL_PUBLIC = {
    ROOT / "site" / "public" / "data" / "findings.json": [
        ROOT / "site" / "src" / "data" / "findings.json",
        ROOT / "hf_space" / "findings.json",
    ],
    ROOT / "site" / "public" / "data" / "metrics.json": [
        ROOT / "site" / "src" / "data" / "metrics.json",
        ROOT / "hf_space" / "metrics.json",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def registry_ids() -> set[str]:
    registry = load_json(REGISTRY)
    return {row["id"] for row in registry["findings"]}


def public_finding_ids() -> set[str]:
    findings = load_json(ROOT / "site" / "public" / "data" / "findings.json")
    return {row["id"] for row in findings["findings"]}


def validate_registry_coverage() -> None:
    missing = sorted(public_finding_ids() - registry_ids())
    if missing:
        raise SystemExit(f"registry is missing public finding IDs: {', '.join(missing)}")


def mirror_public_json(*, check: bool) -> None:
    for src, destinations in CANONICAL_PUBLIC.items():
        if not src.exists():
            raise SystemExit(f"canonical aggregate missing: {src.relative_to(ROOT)}")
        for dst in destinations:
            if check:
                if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
                    raise SystemExit(
                        "stale aggregate mirror: "
                        f"{dst.relative_to(ROOT)} differs from {src.relative_to(ROOT)}"
                    )
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate mirrors without writing files",
    )
    args = parser.parse_args()

    validate_registry_coverage()
    mirror_public_json(check=args.check)


if __name__ == "__main__":
    main()
