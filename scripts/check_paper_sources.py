"""Fast paper-source checks that do not require a LaTeX installation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def main() -> None:
    paper_tex = (PAPER / "paper.tex").read_text(encoding="utf-8")
    required_inputs = [
        "sections/abstract",
        "sections/introduction",
        "sections/background",
        "sections/methods",
        "sections/results",
        "sections/dynamics",
        "sections/positioning",
        "sections/discussion",
        "sections/conclusion",
        "sections/appendix",
    ]
    for item in required_inputs:
        if item not in paper_tex:
            raise SystemExit(f"paper.tex does not include {item}")

    dynamics = (PAPER / "sections" / "dynamics.tex").read_text(encoding="utf-8")
    required_phrases = [
        "exclude harmful prompts",
        "raw completions",
        "interpretive status",
        "mechanism--analogy boundary",
    ]
    dynamics_lower = dynamics.lower()
    for phrase in required_phrases:
        if phrase.lower() not in dynamics_lower:
            raise SystemExit(f"dynamics section missing phrase: {phrase}")


if __name__ == "__main__":
    main()
