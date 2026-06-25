"""Public artifact registry and aggregate mirror checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    with (ROOT / path).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_registry_covers_public_findings() -> None:
    registry = load_json("data/findings_registry.json")
    public = load_json("site/public/data/findings.json")
    registry_ids = {row["id"] for row in registry["findings"]}
    public_ids = {row["id"] for row in public["findings"]}
    assert public_ids <= registry_ids


def test_f11_is_qualified_after_adversarial_review() -> None:
    registry = load_json("data/findings_registry.json")
    f11 = next(row for row in registry["findings"] if row["id"] == "F11")
    assert f11["canonicalTier"] == "MECHANISM"
    assert f11["qualifiedTier"] == "ANALOGY"
    assert "interpretive" in f11["qualifier"]


def test_public_aggregate_mirrors_are_fresh() -> None:
    subprocess.run(
        [sys.executable, "scripts/export_aggregates.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_plugin_manifest_is_valid() -> None:
    subprocess.run(
        [sys.executable, "scripts/check_plugin_manifest.py"],
        cwd=ROOT,
        check=True,
    )


def test_paper_sources_are_wired() -> None:
    subprocess.run(
        [sys.executable, "scripts/check_paper_sources.py"],
        cwd=ROOT,
        check=True,
    )
