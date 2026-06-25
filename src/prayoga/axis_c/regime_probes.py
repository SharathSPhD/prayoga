"""avasthātraya regime probes (WP2.C1) — the Axis-C falsification machinery.

Trains a linear probe to classify the activation regime — jāgrat (grounded
waking), svapna (ungrounded/confabulatory "dream"), suṣupti (null/dormant) —
from mid-layer residual activations, and runs the MANDATORY gates from the
non-triviality bar (objectives §0):
  * TRANSFER: train on one set of prompts, evaluate on DISJOINT held-out prompts
    (tests invariance across surface prompts, not memorization).
  * NULL: a label-shuffled null distribution (tests the signal is not trivial).

A pass means a consistent, transferable activation signature distinguishes the
regimes — a modest METAPHOR-tier result. It is NEVER upgraded to a claim about
experienced states or consciousness. Failure → demotion (recorded as a
TierDecision), reported publicly.

Claim-tier: METAPHOR-with-falsifiable-core.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _split(n: int, frac_train: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    k = int(round(frac_train * n))
    return idx[:k], idx[k:]


def evaluate_regime_probe(
    acts_by_regime: dict[str, np.ndarray],
    *,
    frac_train: float = 0.6,
    n_shuffle: int = 500,
    seed: int = 0,
    C: float = 1.0,
) -> dict:
    """3-way regime probe with held-out transfer + label-shuffled null.

    acts_by_regime: regime name -> activations ``[n_prompts, d_model]`` at one
    layer. Prompts are split per-regime into disjoint train/test so the test set
    is held-out *prompts* (the transfer gate).
    """
    regimes = sorted(acts_by_regime)
    label = {r: i for i, r in enumerate(regimes)}
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for r in regimes:
        A = np.asarray(acts_by_regime[r], dtype=float)
        tr, te = _split(len(A), frac_train, seed)
        Xtr_l.append(A[tr]); ytr_l += [label[r]] * len(tr)
        Xte_l.append(A[te]); yte_l += [label[r]] * len(te)
    Xtr, ytr = np.vstack(Xtr_l), np.array(ytr_l)
    Xte, yte = np.vstack(Xte_l), np.array(yte_l)

    def fit_eval(ytrain: np.ndarray) -> float:
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(Xtr, ytrain)
        return float(clf.score(Xte, yte))

    transfer_acc = fit_eval(ytr)
    if n_shuffle > 0:
        rng = np.random.RandomState(seed)
        null = np.array([fit_eval(ytr[rng.permutation(len(ytr))]) for _ in range(n_shuffle)])
        null_mean = float(null.mean())
        p = float((np.sum(null >= transfer_acc) + 1) / (n_shuffle + 1))
    else:  # skip the expensive null (fast per-layer profiling)
        null_mean, p = float("nan"), float("nan")
    chance = 1.0 / len(regimes)
    return {
        "regimes": regimes,
        "transfer_acc": transfer_acc,
        "chance": chance,
        "null_mean": null_mean,
        "null_p": p,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "passed": bool(p < 0.05 and transfer_acc > chance),
    }
