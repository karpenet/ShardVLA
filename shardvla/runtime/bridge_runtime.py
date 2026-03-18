from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch

from shardvla.codec.lerobot_json_codec import encode_lerobot_observation
from shardvla.config import CommunicationConfig, ShardVLARuntimeConfig
from shardvla.transport.grpc_intent_client import IntentClient, IntentClientConfig

logger = logging.getLogger(__name__)


class DummyActionHead:
    def forward(self, embedding_msg) -> str:
        return f"action(seq={embedding_msg.seq_id}, shape={list(embedding_msg.shape)}, dtype={embedding_msg.dtype})"

    def safe_action(self) -> str:
        return "SAFE_STOP"

    def actuate(self, action: str) -> None:
        logger.info("low_node_actuate action=%s", action)


@dataclass
class BridgeLoopsConfig:
    sensing_hz: float = 12.0
    control_hz: float = 100.0
    max_age_s: float = 0.2


class ShardVLARuntime:
    """Low-node runtime: sensing loop publishes observations; control loop consumes embeddings."""

    def __init__(
        self,
        cfg: Optional[CommunicationConfig] = None,
        runtime_cfg: Optional[ShardVLARuntimeConfig] = None,
        action_head: Optional[DummyActionHead] = None,
        loops_cfg: Optional[BridgeLoopsConfig] = None,
    ):
        if runtime_cfg is not None and cfg is not None:
            raise ValueError("Pass either cfg=CommunicationConfig or runtime_cfg=ShardVLARuntimeConfig, not both")

        if runtime_cfg is None:
            runtime_cfg = ShardVLARuntimeConfig(comms=cfg or CommunicationConfig(server_addr="127.0.0.1:50051"))
        runtime_cfg = runtime_cfg.with_defaults_applied()

        self.cfg = runtime_cfg.comms
        self.action_head = action_head or DummyActionHead()
        self.loops_cfg = loops_cfg or BridgeLoopsConfig(
            sensing_hz=runtime_cfg.sensing_hz,
            control_hz=runtime_cfg.control_hz,
            max_age_s=float(runtime_cfg.max_embedding_age_s),
        )

        self.comms = IntentClient(
            IntentClientConfig(
                server_addr=self.cfg.server_addr,
                robot_id=self.cfg.robot_id,
                session_id=self.cfg.session_id,
                max_pending_observations=self.cfg.max_pending_observations,
                reconnect_backoff_initial_s=self.cfg.reconnect_backoff_initial_s,
                reconnect_backoff_max_s=self.cfg.reconnect_backoff_max_s,
            )
        )

        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    async def _sensing_loop(self):
        period = 1.0 / float(self.loops_cfg.sensing_hz)
        seq = 0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            batch_size = 1
            prompt = "Pick up the red cube and place it in the bin"
            dummy_state_dim = 44
            dummy_action_horizon = 16
            image_size = 256

            state = torch.randn(batch_size, dummy_state_dim, dtype=torch.float32)
            images = torch.rand(batch_size, 3, image_size, image_size, dtype=torch.float32)

            batch = {
                "observation.state": state,
                "observation.images.ego_view": images,
                "task": [prompt for _ in range(batch_size)],
                "n_action_steps": dummy_action_horizon,
            }

            payload, payload_format = encode_lerobot_observation(batch)
            logger.info(
                "low_node_observation_published seq_id=%d payload_format=%s bytes=%d state_shape=%s image_shape=%s",
                seq,
                payload_format,
                len(payload),
                tuple(state.shape),
                tuple(images.shape),
            )

            self.comms.publish_observation(
                seq_id=seq,
                t_capture_ms=int(time.time() * 1000),
                payload=payload,
                payload_format=payload_format,
            )

            seq += 1
            dt = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, period - dt))

    async def _action_loop(self):
        period = 1.0 / float(self.loops_cfg.control_hz)
        last_emb = None
        last_emb_time = 0.0

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            emb = await self.comms.latest_embedding.get()
            if emb is not None:
                last_emb = emb
                last_emb_time = time.time()
                logger.info(
                    "low_node_embedding_received robot_id=%s session_id=%s seq_id=%s shape=%s dtype=%s bytes=%d",
                    emb.robot_id,
                    emb.session_id,
                    emb.seq_id,
                    list(emb.shape),
                    emb.dtype,
                    len(emb.data),
                )

            if last_emb is None or (time.time() - last_emb_time) > float(self.loops_cfg.max_age_s):
                action = self.action_head.safe_action()
            else:
                action = self.action_head.forward(last_emb)

            self.action_head.actuate(action)

            dt = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, period - dt))

    def start(self) -> None:
        if self._tasks:
            return

        self._stop_event.clear()
        self._tasks = [
            asyncio.create_task(self.comms.run(), name="comms_run"),
            asyncio.create_task(self._sensing_loop(), name="sensing_loop"),
            asyncio.create_task(self._action_loop(), name="control_loop"),
        ]

    async def wait(self) -> None:
        if not self._tasks:
            self.start()
        await asyncio.gather(*self._tasks)

    async def run_forever(self) -> None:
        self.start()
        await self.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        await self.comms.stop()
        for t in self._tasks:
            t.cancel()

