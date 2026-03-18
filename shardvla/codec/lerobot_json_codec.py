from __future__ import annotations

import base64
import io
import json
from typing import Any, TypedDict

import numpy as np
import torch
from PIL import Image


class LeRobotObservationBatch(TypedDict, total=False):
    # Minimal keys used by current runtime/server wiring.
    observation_state: torch.Tensor
    observation_images_ego_view: torch.Tensor
    task: list[str]


def encode_lerobot_observation(batch: dict[str, Any]) -> tuple[bytes, str]:
    """Encode a LeRobot-style batch into JSON + base64-encoded JPEGs.

    Expected keys (current convention):
    - observation.state: Float tensor (B, D)
    - observation.images.ego_view: Float tensor (B, 3, H, W) in [0,1]
    - task: list[str]
    """
    batch_size = int(batch["observation.images.ego_view"].shape[0])

    imgs_b64: list[str] = []
    for b in range(batch_size):
        img_tensor = batch["observation.images.ego_view"][b]  # (C, H, W) in [0,1]
        img_uint8 = (img_tensor.clamp(0, 1) * 255.0).to(torch.uint8)
        img_hwc = img_uint8.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        pil_img = Image.fromarray(img_hwc, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        jpeg_bytes = buf.getvalue()
        imgs_b64.append(base64.b64encode(jpeg_bytes).decode("ascii"))

    payload_dict = {
        "observation.state": batch["observation.state"].cpu().tolist(),
        "observation.images.ego_view_b64_jpeg": imgs_b64,
        "task": batch.get("task", []),
    }
    return json.dumps(payload_dict).encode("utf-8"), "json:LeRobotBatchV1"


def decode_lerobot_observation(payload: bytes, payload_format: str) -> dict[str, Any]:
    """Decode JSON + base64 JPEG payload into a LeRobot-style batch dict."""
    if not payload:
        return {}
    if not payload_format.startswith("json:"):
        raise ValueError(f"Unsupported payload_format: {payload_format}")

    data = json.loads(payload.decode("utf-8"))
    batch: dict[str, Any] = {}

    if "observation.state" in data:
        batch["observation.state"] = torch.as_tensor(data["observation.state"], dtype=torch.float32)

    imgs_b64 = data.get("observation.images.ego_view_b64_jpeg")
    if imgs_b64 is None:
        raise ValueError("observation.images.ego_view_b64_jpeg is required")
    if isinstance(imgs_b64, str):
        imgs_b64 = [imgs_b64]

    imgs: list[torch.Tensor] = []
    for s in imgs_b64:
        jpeg_bytes = base64.b64decode(s)
        pil_img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img_np = np.array(pil_img, copy=False)  # (H, W, C), uint8
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        imgs.append(img_t)
    batch["observation.images.ego_view"] = torch.cat(imgs, dim=0)

    # Pass through any extra keys (e.g., task)
    for k, v in data.items():
        if k in {"observation.state", "observation.images.ego_view_b64_jpeg"}:
            continue
        batch[k] = v

    return batch

