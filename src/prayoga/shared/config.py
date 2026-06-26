"""Configuration models for prayoga experiments (Pydantic v2).

Replaces ACD's dataclass config with validated Pydantic models and a registry
of the Tier-2 target models (sub-7B dense; Nemotron is stretch/contrast only).
Round-trips to/from the YAMLs in ``configs/``.

Claim-tier: plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    hf_id: str
    n_layers: int
    mid_layer: int  # default residual-stream layer for direction/SAE work
    dense: bool = True  # False => not attention-dense (Nemotron Mamba-2)
    tier2_role: Literal["core", "scale", "stretch"] = "core"


class GateConfig(BaseModel):
    alpha: float = 0.05
    n_boot: int = 10_000
    n_perm: int = 10_000
    n_shuffle: int = 1_000


class ExperimentConfig(BaseModel):
    model: ModelConfig
    seed: int = 42
    out_dir: Path = Path("results")
    gate: GateConfig = Field(default_factory=GateConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))


# Tier-2 model registry (objectives §6). Sub-7B dense transformers are core;
# Nemotron Nano 2 9B (hybrid Mamba-2) is stretch/contrast ONLY.
TIER2_MODELS: dict[str, ModelConfig] = {
    "gemma-2-2b": ModelConfig(
        name="gemma-2-2b", hf_id="google/gemma-2-2b", n_layers=26, mid_layer=13,
        tier2_role="core",
    ),
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b", hf_id="meta-llama/Llama-3.2-3B", n_layers=28, mid_layer=14,
        tier2_role="core",
    ),
    "llama-3.2-1b": ModelConfig(
        name="llama-3.2-1b", hf_id="meta-llama/Llama-3.2-1B", n_layers=16, mid_layer=8,
        tier2_role="scale",
    ),
    "gemma-3-1b": ModelConfig(
        name="gemma-3-1b", hf_id="google/gemma-3-1b-pt", n_layers=26, mid_layer=13,
        tier2_role="scale",
    ),
    "gemma-3-4b": ModelConfig(
        name="gemma-3-4b", hf_id="google/gemma-3-4b-pt", n_layers=34, mid_layer=17,
        tier2_role="scale",
    ),
    "nemotron-nano-2-9b": ModelConfig(
        name="nemotron-nano-2-9b", hf_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        n_layers=56, mid_layer=28, dense=False, tier2_role="stretch",
    ),
}
