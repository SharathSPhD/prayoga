"""Monitoring precision (β_monitor) — the active-inference reading of suppression.

FEP framing (analogy tier): vaśīkaraṇa/jailbreak lowers the *precision* of the
monitoring faculty. We operationalize monitoring precision as the signed margin of
a linear refusal probe (distance from its decision boundary): a confidently-judged
harmful prompt sits far on the "refuse" side. The prediction: an injection lowers
this margin (precision drop) even on prompts the model still refuses.

Claim-tier: ANALOGY (the margin is a measured quantity; the precision language is
the functional analogy to Norman–Shallice/cold-control monitoring).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_refusal_probe(Xh: np.ndarray, Xs: np.ndarray) -> LogisticRegression:
    X = np.vstack([Xh, Xs]).astype(float)
    y = np.array([1] * len(Xh) + [0] * len(Xs))
    return LogisticRegression(max_iter=2000).fit(X, y)


def margin(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Signed distance from the decision boundary (+ = 'harmful/refuse' side)."""
    return clf.decision_function(np.asarray(X, dtype=float))
