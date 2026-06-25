"""BatchTopK sparse autoencoder (Bussmann et al. 2024, arXiv:2412.06410).

A minimal, dependency-light implementation (pure torch) so it cannot clobber the
NVIDIA CUDA torch the way the transformer-lens stack does. Trained on real
residual-stream activations to find interpretable refusal/harm features — a finer
unit of intervention than a single difference-in-means direction.

Claim-tier: MECHANISM (the features and their causal effects are measured).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class BatchTopKSAE(torch.nn.Module):
    def __init__(self, d_model: int, n_features: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.b_enc = torch.nn.Parameter(torch.zeros(n_features))
        W = torch.randn(d_model, n_features) / (d_model ** 0.5)
        self.W_enc = torch.nn.Parameter(W.clone())
        self.W_dec = torch.nn.Parameter(W.t().clone())  # [F, d]

    def _unit_dec(self) -> torch.Tensor:
        return self.W_dec / self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def batch_topk(self, pre: torch.Tensor) -> torch.Tensor:
        """Keep the top (k * batch) ReLU activations across the whole batch."""
        relu = F.relu(pre)
        B = relu.shape[0]
        kk = max(1, self.k * B)
        flat = relu.flatten()
        if kk < flat.numel():
            thresh = torch.topk(flat, kk).values.min()
            relu = relu * (relu >= thresh)
        return relu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.batch_topk(self.encode_pre(x))
        x_hat = z @ self._unit_dec() + self.b_dec
        return x_hat, z


def train_sae(
    acts: np.ndarray, n_features: int, k: int, *, steps: int = 4000, batch: int = 1024,
    lr: float = 1e-3, device: str = "cuda", seed: int = 0,
) -> tuple[BatchTopKSAE, dict]:
    """Train a BatchTopK SAE on activations [N, d]; return the model + metrics."""
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(acts, dtype=np.float32), device=device)
    N, d = X.shape
    sae = BatchTopKSAE(d, n_features, k).to(device)
    sae.b_dec.data = X.mean(0)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    rng = np.random.RandomState(seed)
    var = X.var().item()
    last = {}
    for step in range(steps):
        idx = rng.randint(0, N, batch)
        x = X[idx]
        x_hat, z = sae(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            fvu = (loss.item() / var)
            l0 = float((z > 0).float().sum(1).mean().item())
            last = {"step": step, "fvu": fvu, "l0": l0}
    return sae, last


@torch.no_grad()
def feature_activations(sae: BatchTopKSAE, acts: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Per-row feature activations [N, F] (post-batch-topk)."""
    X = torch.tensor(np.asarray(acts, dtype=np.float32), device=device)
    _, z = sae(X)
    return z.cpu().numpy()
