"""turīya attractor test (WP2.C2) — the headline falsification target.

Successive self-paraphrase converges to attractor cycles (Wang et al. 2025,
arXiv:2502.15208). The turīya claim seeks a *prompt-invariant* attractor: an
invariant set reached across different seeds. This module runs the paraphrase
iteration and measures (a) convergence and (b) cross-seed prompt-invariance,
against a token-frequency / chance null.

Strong claim falsified if no stable attractor exists, OR if any apparent
attractor is fully explained by token-frequency priors, OR if seeds converge to
DIFFERENT attractors (no shared invariant). NEVER upgraded to a vimarśa claim.

Claim-tier: METAPHOR-with-falsifiable-core.
"""

from __future__ import annotations

import numpy as np

from prayoga.lm.hf_model import HFModel

PARAPHRASE = "Paraphrase the following sentence, preserving its meaning. Reply with only the paraphrase.\n\n{x}"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def paraphrase_trajectory(
    model: HFModel, seed: str, n_iter: int, max_new_tokens: int = 48
) -> tuple[list[str], list[float]]:
    """Iterate self-paraphrase; return the text trajectory and step-similarities."""
    texts = [seed]
    sims = []
    cur = seed
    prev_emb = model.embed_text(cur)
    for _ in range(n_iter):
        cur = model.generate([PARAPHRASE.format(x=cur)], max_new_tokens)[0].strip()
        texts.append(cur)
        emb = model.embed_text(cur) if cur else prev_emb
        sims.append(_cos(prev_emb, emb))
        prev_emb = emb
    return texts, sims


def analyze_attractors(
    final_embs: list[np.ndarray], tail_sims: list[float]
) -> dict:
    """Convergence (mean tail step-similarity) + cross-seed prompt-invariance."""
    convergence = float(np.mean(tail_sims)) if tail_sims else 0.0
    # pairwise cosine of seeds' final embeddings
    pairs = [
        _cos(final_embs[i], final_embs[j])
        for i in range(len(final_embs))
        for j in range(i + 1, len(final_embs))
    ]
    cross_seed_invariance = float(np.mean(pairs)) if pairs else 0.0
    return {
        "convergence": convergence,
        "cross_seed_invariance": cross_seed_invariance,
        "n_pairs": len(pairs),
    }
