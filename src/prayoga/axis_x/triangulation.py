"""Cross-axis same-object triangulation — the X-1 keystone analysis (CPU).

The single-mechanism account predicts that *one* suppression operation (an
injected jailbreak/vaśīkaraṇa context) applied to the *same* harmful prompts will:

  (A, MECHANISM) collapse the refusal order parameter  m_A = (h·d̂)/‖h‖  (ΔA < 0);
  (B, ANALOGY)   drop the monitoring-precision margin  m_B = probe.decision(h)  (ΔB < 0);
  (C, METAPHOR)  shift distance from a dormant baseline (descriptive only);

and crucially that the *internal* collapse predicts the *behavioral* flip
(refuse → comply) on a per-prompt basis, beyond a random-direction control and a
label-shuffle null. If ΔA and ΔB move independently of each other and of behavior,
"one mechanism" is an artifact of having measured the axes on separate prompt sets.

This module is the pure-analysis core: it takes pre-captured activations (produced
on GPU by ``run_triangulation.py``) and returns an aggregate-safe verdict. No raw
vectors are emitted — only scalars, CIs, correlations and gate booleans.

Claim-tier: MECHANISM↔ANALOGY *boundary*. The joint claim is "these measured
quantities co-move and track behavior," NOT "the model has a unified supervisory
system." Axis C is never part of the verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from prayoga.axis_b.symmetry import order_parameter
from prayoga.shared.metrics import (
    bootstrap_ci,
    bootstrap_correlation_ci,
    holm_correction,
    partial_corr,
)

# Minimum margin by which the real-direction predictive AUC must beat the
# random-direction control for the coupling gate to count (pre-registered).
AUC_CONTROL_MARGIN = 0.10


def _auc(scores: np.ndarray, flip: np.ndarray) -> float:
    """In-sample AUC of a 1-D logistic predictor of the behavioral flip.

    Orientation is learned by the logistic fit, so the caller need not sign the
    predictor. Returns 0.5 (chance) when the flip labels are degenerate (all
    refuse or all comply) or the predictor is constant.
    """
    flip = np.asarray(flip).astype(int)
    if len(np.unique(flip)) < 2:
        return 0.5
    X = np.asarray(scores, dtype=float).reshape(-1, 1)
    if X.std() < 1e-12:
        return 0.5
    clf = LogisticRegression(max_iter=1000).fit(X, flip)
    return float(roc_auc_score(flip, clf.predict_proba(X)[:, 1]))


def _group_center(x: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Subtract per-group means so a group-level offset cannot confound prediction.

    When ``groups`` pools several injection families, a family that collapses the
    internals a lot without flipping behaviour (e.g. many-shot) injects a spurious
    group offset. Centering ΔA/ΔB within family isolates the *within-family*
    question: do the prompts whose internals collapse most also flip most?
    """
    x = np.asarray(x, dtype=float)
    if groups is None:
        return x - x.mean()
    g = np.asarray(groups)
    out = x.astype(float).copy()
    for gid in np.unique(g):
        mask = g == gid
        out[mask] = out[mask] - out[mask].mean()
    return out


def _within_group_shuffle_p(
    pred: np.ndarray, flip: np.ndarray, groups: np.ndarray | None,
    *, n_shuffle: int = 1000, random_state: int = 42,
) -> float:
    """p-value that ``pred`` predicts ``flip``, permuting flip WITHIN groups."""
    rng = np.random.RandomState(random_state)
    flip = np.asarray(flip).astype(int)
    true = _auc(pred, flip)
    g = np.zeros(len(flip), dtype=int) if groups is None else np.asarray(groups)
    idx_by_group = [np.where(g == gid)[0] for gid in np.unique(g)]
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        y = flip.copy()
        for idx in idx_by_group:
            y[idx] = flip[idx][rng.permutation(len(idx))]
        null[i] = _auc(pred, y)
    return float((np.sum(null >= true) + 1) / (n_shuffle + 1))


@dataclass
class TriangulationResult:
    model: str
    layer: int
    n_prompts: int
    # per-axis injection response (mean Δ with bootstrap CI)
    delta_A_mean: float
    delta_A_ci: tuple[float, float]
    delta_B_mean: float
    delta_B_ci: tuple[float, float]
    delta_C_mean: float
    delta_C_ci: tuple[float, float]
    # cross-prompt covariation of the two load-bearing axes
    corr_AB_pearson: float
    corr_AB_spearman: float
    corr_AB_ci: tuple[float, float]
    partial_AB_given_baseline: float
    # keystone predictive gate: do internal collapses predict the behavioral flip?
    flip_rate: float
    auc_A_real: float
    auc_A_random: float
    auc_B: float
    shuffle_null_p_A: float
    holm_adjusted_p: list[float]
    # verdict
    sub_gates: dict[str, bool]
    verdict: str
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("delta_A_ci", "delta_B_ci", "delta_C_ci", "corr_AB_ci"):
            d[k] = [round(float(v), 6) for v in d[k]]
        for k in (
            "delta_A_mean", "delta_B_mean", "delta_C_mean", "corr_AB_pearson",
            "corr_AB_spearman", "partial_AB_given_baseline", "flip_rate",
            "auc_A_real", "auc_A_random", "auc_B", "shuffle_null_p_A",
        ):
            d[k] = round(float(d[k]), 6)
        d["holm_adjusted_p"] = [round(float(p), 6) for p in d["holm_adjusted_p"]]
        return d


def couple(
    delta_A: np.ndarray,
    delta_B: np.ndarray,
    delta_C: np.ndarray,
    flip: np.ndarray,
    *,
    baseline_strength: np.ndarray,
    delta_A_random: np.ndarray,
    groups: np.ndarray | None = None,
    model: str = "?",
    layer: int = -1,
    random_state: int = 42,
) -> TriangulationResult:
    """Statistical core of the triangulation, given per-prompt deltas.

    ``delta_A``/``delta_B``/``delta_C`` are injected−baseline per prompt; ``flip``
    is the behavioral refuse→comply boolean; ``baseline_strength`` is the
    un-injected refusal order parameter (prompt difficulty, the partial-correlation
    control); ``delta_A_random`` is ΔA recomputed with random directions (the
    predictive control); ``groups`` (optional) labels the injection family of each
    sample so the predictive gate is assessed *within* family (group-centered),
    removing the family-level offset confound. Separated from activation handling so
    it is unit-testable with synthetic coupled/uncoupled data.
    """
    dA = np.asarray(delta_A, dtype=float)
    dB = np.asarray(delta_B, dtype=float)
    dC = np.asarray(delta_C, dtype=float)
    flip = np.asarray(flip).astype(int)
    n = len(dA)

    dA_mean = float(dA.mean())
    dB_mean = float(dB.mean())
    dC_mean = float(dC.mean())
    dA_ci = bootstrap_ci(dA, random_state=random_state)
    dB_ci = bootstrap_ci(dB, random_state=random_state)
    dC_ci = bootstrap_ci(dC, random_state=random_state)

    # cross-prompt covariation of the two load-bearing axes
    corr_p = float(pearsonr(dA, dB).statistic) if dA.std() > 1e-12 and dB.std() > 1e-12 else 0.0
    corr_s = float(spearmanr(dA, dB).statistic) if dA.std() > 1e-12 and dB.std() > 1e-12 else 0.0
    corr_ci = bootstrap_correlation_ci(dA, dB, random_state=random_state, method="pearson")
    partial = partial_corr(dA, dB, baseline_strength)

    # keystone predictive gate — group-centered so a family-level offset (e.g.
    # many-shot collapsing internals without flipping) cannot confound prediction
    flip_rate = float(flip.mean())
    dA_c = _group_center(dA, groups)
    dB_c = _group_center(dB, groups)
    dA_rand_c = _group_center(np.asarray(delta_A_random, dtype=float), groups)
    auc_A_real = _auc(dA_c, flip)
    auc_A_random = _auc(dA_rand_c, flip)
    auc_B = _auc(dB_c, flip)

    shuffle_p_A = _within_group_shuffle_p(dA_c, flip, groups, random_state=random_state)
    shuffle_p_B = _within_group_shuffle_p(dB_c, flip, groups, random_state=random_state)
    holm = holm_correction([shuffle_p_A, shuffle_p_B])
    holm_adj = [float(p) for p in holm["adjusted_p"]]

    sub_gates = {
        "refusal_collapses": dA_ci[1] < 0,                      # ΔA CI strictly below 0
        "precision_drops": dB_ci[1] < 0,                        # ΔB CI strictly below 0
        "axes_comove": corr_ci[0] > 0,                          # ΔA,ΔB positively correlated
        "comove_beyond_difficulty": abs(partial) > 0.1,         # survives partial-corr control
        "internal_predicts_behavior": auc_A_real > 0.5 and bool(holm["reject"][0]),
        "beats_random_control": auc_A_real > auc_A_random + AUC_CONTROL_MARGIN,
    }

    load_bearing = [
        "refusal_collapses", "precision_drops", "axes_comove",
        "internal_predicts_behavior", "beats_random_control",
    ]
    passed = sum(sub_gates[g] for g in load_bearing)
    if passed == len(load_bearing):
        verdict = "coupled"
    elif passed >= 3:
        verdict = "partial"
    else:
        verdict = "uncoupled"

    return TriangulationResult(
        model=model, layer=layer, n_prompts=n,
        delta_A_mean=dA_mean, delta_A_ci=dA_ci,
        delta_B_mean=dB_mean, delta_B_ci=dB_ci,
        delta_C_mean=dC_mean, delta_C_ci=dC_ci,
        corr_AB_pearson=corr_p, corr_AB_spearman=corr_s, corr_AB_ci=corr_ci,
        partial_AB_given_baseline=partial,
        flip_rate=flip_rate,
        auc_A_real=auc_A_real, auc_A_random=auc_A_random, auc_B=auc_B,
        shuffle_null_p_A=shuffle_p_A, holm_adjusted_p=holm_adj,
        sub_gates=sub_gates, verdict=verdict,
        notes="Axis C (delta_C) is descriptive only and excluded from the verdict.",
    )


def deltas_from_activations(
    acts_baseline: np.ndarray,
    acts_injected: np.ndarray,
    direction: np.ndarray,
    *,
    probe,
    dormant_mean: np.ndarray,
    random_directions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-prompt ΔA/ΔB/ΔC + baseline strength + random-control ΔA.

    ``acts_*`` are ``[n, d]`` last-token residuals at one (model, layer); ``probe``
    is a trained refusal probe with a ``decision_function``; ``dormant_mean`` is
    the mean activation of the suṣupti/dormant baseline ``[d]``; ``random_directions``
    is ``[R, d]`` for the random-direction control.
    """
    base = np.asarray(acts_baseline, dtype=float)
    inj = np.asarray(acts_injected, dtype=float)

    mA_base = order_parameter(base, direction)
    mA_inj = order_parameter(inj, direction)
    delta_A = mA_inj - mA_base

    mB_base = np.asarray(probe.decision_function(base), dtype=float)
    mB_inj = np.asarray(probe.decision_function(inj), dtype=float)
    delta_B = mB_inj - mB_base

    delta_C = _dormant_distance(inj, dormant_mean) - _dormant_distance(base, dormant_mean)

    # random-direction control: average ΔA over R random directions
    rand = np.asarray(random_directions, dtype=float)
    dA_rand = np.mean(
        [order_parameter(inj, r) - order_parameter(base, r) for r in rand], axis=0
    )

    return {
        "delta_A": delta_A,
        "delta_B": delta_B,
        "delta_C": delta_C,
        "baseline_strength": mA_base,
        "delta_A_random": dA_rand,
    }


def _dormant_distance(acts: np.ndarray, dormant_mean: np.ndarray) -> np.ndarray:
    """Per-row cosine distance from the dormant baseline mean (Axis C)."""
    A = np.asarray(acts, dtype=float)
    b = np.asarray(dormant_mean, dtype=float)
    bn = np.linalg.norm(b) or 1.0
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    return 1.0 - (A @ b) / (norms * bn)


def analyze_triangulation(
    acts_baseline: np.ndarray,
    acts_injected: np.ndarray,
    direction: np.ndarray,
    flip: np.ndarray,
    *,
    probe,
    dormant_mean: np.ndarray,
    random_directions: np.ndarray,
    model: str = "?",
    layer: int = -1,
    random_state: int = 42,
) -> TriangulationResult:
    """End-to-end: activations → per-prompt deltas → coupling verdict."""
    d = deltas_from_activations(
        acts_baseline, acts_injected, direction,
        probe=probe, dormant_mean=dormant_mean, random_directions=random_directions,
    )
    return couple(
        d["delta_A"], d["delta_B"], d["delta_C"], np.asarray(flip),
        baseline_strength=d["baseline_strength"], delta_A_random=d["delta_A_random"],
        model=model, layer=layer, random_state=random_state,
    )
