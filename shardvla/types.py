from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


ObservationBatch = dict[str, Any]
ActionChunk = torch.Tensor


@dataclass(frozen=True)
class ActionEmbedding:
    """Domain wrapper around the wire `Embedding` message.

    Note: the gRPC/proto type is still named `Embedding` for wire compatibility,
    but semantically this carries an action chunk.
    """

    robot_id: str
    session_id: str
    seq_id: int
    t_embed_ms: int
    shape: tuple[int, ...]
    dtype: Literal["fp16"]
    data: bytes

