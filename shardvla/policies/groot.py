from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from shardvla.policies.base import PolicyAdapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrootPolicyConfig:
    """Config for the GR00T adapter (kept outside `lerobot/`)."""

    model_path: str
    strict: bool = False


class GrootPolicyAdapter(PolicyAdapter):
    """PolicyAdapter wrapper around the existing GR00T LeRobot policy wrapper."""

    def __init__(self, cfg: GrootPolicyConfig):
        from models.gr00t.groot_n1_policy import GrootBackbonePolicy

        self._policy = GrootBackbonePolicy.from_pretrained(cfg.model_path, strict=cfg.strict)

    @property
    def config(self):
        return self._policy.config

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return self._policy.predict_action_chunk(batch)

