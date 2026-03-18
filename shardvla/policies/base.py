from __future__ import annotations

from typing import Protocol

import torch


class PolicyAdapter(Protocol):
    """Minimal policy boundary used by the high-node server."""

    def predict_action_chunk(self, batch: dict) -> torch.Tensor: ...

