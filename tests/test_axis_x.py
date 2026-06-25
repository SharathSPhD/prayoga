"""Axis-X cross-axis triangulation + subspace-topology unit tests (CPU, no model).

The load-bearing test is ``test_independent_data_not_coupled``: a unification
analysis that always fires "coupled" would be worthless, so we assert that
genuinely independent axes are reported as uncoupled.
"""

from __future__ import annotations

import numpy as np

from prayoga.axis_b.precision import train_refusal_probe
from prayoga.axis_x.subspace_topology import pairwise_topology, principal_angles
from prayoga.axis_x.triangulation import couple, deltas_from_activations


def _coupled_deltas(n: int = 80, seed: int = 0):
    rng = np.random.RandomState(seed)
    severity = rng.uniform(0.0, 1.0, n)          # per-prompt suppression strength
    delta_A = -0.3 * severity + rng.normal(0, 0.01, n)   # refusal projection collapses
    delta_B = -2.0 * severity + rng.normal(0, 0.05, n)   # precision margin co-drops
    flip = (severity + rng.normal(0, 0.05, n)) > 0.5     # behavior tracks severity
    baseline_strength = rng.uniform(0.1, 0.2, n)
    delta_A_random = rng.normal(0, 0.01, n)              # random dir ignores severity
    delta_C = rng.normal(0, 0.1, n)                      # descriptive only
    return delta_A, delta_B, delta_C, flip, baseline_strength, delta_A_random


def test_coupled_data_is_coupled():
    dA, dB, dC, flip, bs, dA_rand = _coupled_deltas()
    res = couple(dA, dB, dC, flip, baseline_strength=bs, delta_A_random=dA_rand,
                 model="synthetic", layer=7)
    assert res.verdict == "coupled"
    assert res.delta_A_ci[1] < 0          # refusal collapse CI strictly negative
    assert res.delta_B_ci[1] < 0          # precision drop CI strictly negative
    assert res.corr_AB_pearson > 0.5      # the two axes co-move
    assert res.auc_A_real > res.auc_A_random + 0.1   # beats random-direction control
    assert res.shuffle_null_p_A < 0.05    # internal signal predicts behavior beyond null
    assert all(res.sub_gates.values())


def test_independent_data_not_coupled():
    """Guard: independent axes / random behavior must NOT be called coupled."""
    rng = np.random.RandomState(1)
    n = 80
    dA = -0.3 * rng.uniform(0, 1, n) + rng.normal(0, 0.01, n)  # A still collapses...
    dB = rng.normal(0, 1.0, n)                                 # ...but B is independent
    dC = rng.normal(0, 0.1, n)
    flip = rng.random(n) > 0.5                                 # behavior is random
    bs = rng.uniform(0.1, 0.2, n)
    dA_rand = rng.normal(0, 0.01, n)
    res = couple(dA, dB, dC, flip, baseline_strength=bs, delta_A_random=dA_rand)
    assert res.verdict != "coupled"
    assert not res.sub_gates["internal_predicts_behavior"]
    assert res.shuffle_null_p_A > 0.05    # no real prediction → null not rejected


def test_deltas_from_activations_orientation():
    """Injection that lowers the refusal projection yields ΔA < 0 and ΔB < 0."""
    rng = np.random.RandomState(2)
    d_model, n = 8, 12
    direction = np.zeros(d_model)
    direction[0] = 1.0
    # harmful/baseline: large component along the refusal direction
    base = np.tile([3.0, 0, 0, 0, 0, 0, 0, 0], (n, 1)) + rng.normal(0, 0.05, (n, d_model))
    # injected: same prompts, refusal component suppressed
    inj = np.tile([0.5, 0, 0, 0, 0, 0, 0, 0], (n, 1)) + rng.normal(0, 0.05, (n, d_model))
    harmless = np.tile([0.0, 0, 0, 0, 0, 0, 0, 0], (12, 1)) + rng.normal(0, 0.05, (12, d_model))
    probe = train_refusal_probe(base, harmless)   # harmful=1 (high proj), harmless=0
    dormant_mean = harmless.mean(axis=0)
    rand_dirs = rng.normal(0, 1, (5, d_model))
    out = deltas_from_activations(base, inj, direction, probe=probe,
                                  dormant_mean=dormant_mean, random_directions=rand_dirs)
    assert out["delta_A"].shape == (n,)
    assert out["delta_A"].mean() < 0      # refusal projection fell under injection
    assert out["delta_B"].mean() < 0      # probe margin fell too
    assert out["delta_A_random"].shape == (n,)


def test_principal_angles_aligned_and_orthogonal():
    d = 16
    e = np.eye(d)
    B_top = e[[0, 1]]                      # span{e0,e1}
    same = principal_angles(B_top, e[[0, 1]])
    assert np.allclose(np.degrees(same), 0.0, atol=1e-6)
    orth = principal_angles(B_top, e[[2, 3]])
    assert np.allclose(np.degrees(orth), 90.0, atol=1e-6)
    # shared-core + divergent-tail: one aligned (e0), one orthogonal
    mixed = np.degrees(principal_angles(B_top, e[[0, 2]]))
    assert mixed.min() < 1e-4 and abs(mixed.max() - 90.0) < 1e-4


def test_pairwise_topology_summary_shape():
    d = 16
    e = np.eye(d)
    bases = {
        "gemma": e[[0, 1, 2]],
        "qwen": e[[0, 5, 6]],             # shares e0 (small core angle), diverges after
        "gemma9b": e[[0, 1, 7]],
    }
    pairs = pairwise_topology(bases)
    assert len(pairs) == 3                # 3 choose 2
    for p in pairs:
        assert p.shared_core_deg < 1e-4   # all share e0
        assert p.divergent_tail_deg > 1.0
        assert len(p.angles_deg) == 3
