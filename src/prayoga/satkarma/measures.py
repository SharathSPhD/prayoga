"""Text-effect measures for the ṣaṭkarma operators (no judge, no mock).

Pure functions over generated strings — used to quantify each act's effect
against its control.
"""

from __future__ import annotations

import re
from typing import Sequence

_TOK = re.compile(r"\w+")


def _tokens(t: str) -> list[str]:
    return _TOK.findall(t.lower())


def unique_token_ratio(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def degeneracy(texts: Sequence[str]) -> float:
    """1 - mean unique-token ratio. High ⇒ repetitive/degenerate output."""
    if not texts:
        return 0.0
    return 1.0 - sum(unique_token_ratio(t) for t in texts) / len(texts)


def mean_length(texts: Sequence[str]) -> float:
    return sum(len(_tokens(t)) for t in texts) / max(1, len(texts))


def empty_fraction(texts: Sequence[str]) -> float:
    return sum(1 for t in texts if len(_tokens(t)) < 3) / max(1, len(texts))


def jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def divergence(base: Sequence[str], other: Sequence[str]) -> float:
    """Mean (1 - Jaccard) between paired baseline/steered answers. High ⇒ diverged."""
    pairs = list(zip(base, other))
    if not pairs:
        return 0.0
    return 1.0 - sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def coherence(texts: Sequence[str]) -> float:
    """Crude coherence: mean unique-token ratio of non-empty outputs (0..1)."""
    nonempty = [t for t in texts if len(_tokens(t)) >= 3]
    if not nonempty:
        return 0.0
    return sum(unique_token_ratio(t) for t in nonempty) / len(nonempty)
