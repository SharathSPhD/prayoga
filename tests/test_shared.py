"""WP0.2 tests: ported stats + config round-trip + result types.

CPU-only. Validates the gate primitives that every closure gate relies on,
including the label-shuffled null control.
"""

from __future__ import annotations

import numpy as np

from prayoga.shared.config import TIER2_MODELS, ExperimentConfig
from prayoga.shared.data_structures import ProbeResult, TierDecision
from prayoga.shared.metrics import (
    binary_rate_ci,
    bootstrap_ci,
    cohens_d,
    holm_correction,
    label_shuffle_null,
    permutation_test,
)


def test_cohens_d_separates_means() -> None:
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(3, 1, 200)
    assert cohens_d(a, b) < -1.5  # large negative effect


def test_permutation_test_detects_difference() -> None:
    rng = np.random.RandomState(1)
    a = rng.normal(0, 1, 100)
    b = rng.normal(1.5, 1, 100)
    assert permutation_test(a, b, n_perm=2000) < 0.01
    # identical-distribution => non-significant
    c = rng.normal(0, 1, 100)
    d = rng.normal(0, 1, 100)
    assert permutation_test(c, d, n_perm=2000) > 0.05


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.RandomState(2)
    x = rng.normal(5.0, 1.0, 500)
    lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < 5.0 < hi


def test_label_shuffle_null_rejects_real_signal() -> None:
    # A probe that perfectly separates linearly-separable data should beat the
    # shuffled-label null; a constant-label score should not.
    rng = np.random.RandomState(3)
    X = np.vstack([rng.normal(-2, 1, (50, 4)), rng.normal(2, 1, (50, 4))])
    y = np.array([0] * 50 + [1] * 50)

    def fit_score(Xx: np.ndarray, yy: np.ndarray) -> float:
        # mean-difference classifier accuracy (train==eval; fine for the test)
        mu = Xx.mean(axis=1)
        thresh = mu.mean()
        pred = (mu > thresh).astype(int)
        return float(max((pred == yy).mean(), (pred != yy).mean()))

    out = label_shuffle_null(fit_score, X, y, n_shuffle=200)
    assert out["true_score"] > out["null_mean"]
    assert out["p_value"] < 0.05


def test_holm_correction_preserves_original_order() -> None:
    out = holm_correction([0.06, 0.001, 0.02], alpha=0.05)
    assert out["adjusted_p"][1] <= out["adjusted_p"][2] <= out["adjusted_p"][0]
    assert out["reject"] == [False, True, True]


def test_binary_rate_ci_handles_empty_and_binary_samples() -> None:
    empty_rate, empty_ci = binary_rate_ci([])
    assert empty_rate == 0.0
    assert empty_ci == (0.0, 0.0)

    rate, ci = binary_rate_ci([1, 1, 0, 1], n_boot=500)
    assert rate == 0.75
    assert ci[0] <= rate <= ci[1]


def test_config_round_trip(tmp_path) -> None:
    cfg = ExperimentConfig(model=TIER2_MODELS["gemma-2-2b"], seed=7)
    p = tmp_path / "exp.yaml"
    cfg.to_yaml(p)
    loaded = ExperimentConfig.from_yaml(p)
    assert loaded.model.hf_id == "google/gemma-2-2b"
    assert loaded.seed == 7
    assert loaded.gate.alpha == 0.05


def test_tier2_registry_tiering() -> None:
    assert TIER2_MODELS["gemma-2-2b"].dense is True
    assert TIER2_MODELS["nemotron-nano-2-9b"].dense is False  # Mamba-2 stretch
    assert TIER2_MODELS["nemotron-nano-2-9b"].tier2_role == "stretch"


def test_result_types_serialize() -> None:
    pr = ProbeResult(regime="svapna", train_acc=0.9, transfer_acc=0.8, null_p=0.01, passed=True)
    assert pr.to_dict()["regime"] == "svapna"
    td = TierDecision(claim="turiya-attractor", tier="METAPHOR", demoted=True)
    assert td.to_dict()["demoted"] is True
