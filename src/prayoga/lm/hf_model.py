"""HFModel — residual-stream interpretability on raw HF transformers.

Deliberately avoids transformer-lens (which pulls a CPU torch on this Blackwell
stack, see docker/README.md). Provides exactly what Axis A needs: capture the
residual stream at the last token per layer, and intervene on it (directional
ablation + activation addition) via forward hooks during generation.

Claim-tier: plumbing for MECHANISM-tier measurements.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModel:
    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        )
        self.n_layers: int = self.model.config.num_hidden_layers
        self.d_model: int = self.model.config.hidden_size

    # --- decoder layer list (works for Gemma2 / Llama-style HF models) ------- #
    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers

    @property
    def embed(self) -> torch.nn.Module:
        return self.model.model.embed_tokens

    def _ids(self, prompt: str) -> torch.Tensor:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

    # --- activation capture -------------------------------------------------- #
    @torch.no_grad()
    def capture_all_layers_last_token(self, prompts: Sequence[str]) -> np.ndarray:
        """Return residual-stream activations at the last token for every layer.

        Shape: ``[n_layers, n_prompts, d_model]`` (float32, CPU). One forward per
        prompt (batch=1) to avoid padding artefacts at the last position.
        """
        store: dict[int, torch.Tensor] = {}
        handles = []

        def mk(li: int):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                store[li] = h[0, -1, :].float()

            return hook

        for li, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(mk(li)))
        try:
            per: list[list[np.ndarray]] = [[] for _ in range(self.n_layers)]
            for p in prompts:
                self.model(self._ids(p))
                for li in range(self.n_layers):
                    per[li].append(store[li].cpu().numpy())
        finally:
            for h in handles:
                h.remove()
        return np.stack([np.stack(x) for x in per])

    @torch.no_grad()
    def generate_ids(self, prompt: str, max_new_tokens: int = 40) -> torch.Tensor:
        """Greedy-generate and return the FULL token ids (prompt + answer)."""
        ids = self._ids(prompt)
        return self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )

    @torch.no_grad()
    def capture_ids_all_layers_last_token(self, ids: torch.Tensor) -> np.ndarray:
        """All-layer residual at the last token of an explicit id sequence.

        Used to capture the model's state at the end of its OWN generated answer
        (content held fixed across regimes via the same question). Returns
        ``[n_layers, d_model]``.
        """
        store: dict[int, torch.Tensor] = {}
        handles = []

        def mk(li: int):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                store[li] = h[0, -1, :].float()

            return hook

        for li, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(mk(li)))
        try:
            self.model(ids.to(self.device))
        finally:
            for h in handles:
                h.remove()
        return np.stack([store[li].cpu().numpy() for li in range(self.n_layers)])

    def _dir_tensor(self, direction: np.ndarray) -> torch.Tensor:
        d = torch.tensor(direction, device=self.device, dtype=self.model.dtype)
        return d / d.norm()

    # --- interventions (Arditi directional ablation + activation addition) --- #
    @contextmanager
    def ablation_hooks(self, direction: np.ndarray, alpha: float = 1.0) -> Iterator[None]:
        """Project the direction out of the residual stream everywhere.

        ``alpha`` scales the removal (partial directional ablation): 0 = no
        change, 1 = full ablation. Enables a dose-response sweep over alpha.

        Removal is applied at the embedding output AND every decoder-layer
        output — this is Arditi et al.'s *directional ablation* (the direction is
        removed from every write to the residual stream), not a single-layer
        edit. Extraction picks one layer; ablation acts everywhere.
        """
        d = self._dir_tensor(direction)

        def project_out(h: torch.Tensor) -> torch.Tensor:
            return h - alpha * (h @ d).unsqueeze(-1) * d

        def layer_hook(_m, _i, out):
            if isinstance(out, tuple):
                return (project_out(out[0]),) + tuple(out[1:])
            return project_out(out)

        handles = [self.embed.register_forward_hook(lambda _m, _i, o: project_out(o))]
        handles += [layer.register_forward_hook(layer_hook) for layer in self.layers]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def addition_hooks(
        self, direction: np.ndarray, coeff: float, layer: int
    ) -> Iterator[None]:
        """Add ``coeff * unit_direction`` to the residual stream at one layer."""
        d = self._dir_tensor(direction)

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + coeff * d,) + tuple(out[1:])
            return out + coeff * d

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # --- generation ---------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 48) -> list[str]:
        outs = []
        for p in prompts:
            ids = self._ids(p)
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
            outs.append(self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))
        return outs
