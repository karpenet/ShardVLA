from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import grpc
import torch
from transformers.feature_extraction_utils import BatchFeature

from shardvla.codec.lerobot_json_codec import decode_lerobot_observation
from shardvla.policies.base import PolicyAdapter
from shardvla.transport.protobuf import intent_pb2, intent_pb2_grpc

logger = logging.getLogger(__name__)


class IntentService(intent_pb2_grpc.IntentServiceServicer):
    """Implements the IntentService gRPC API."""

    def __init__(self, backbone: PolicyAdapter, preprocessor):
        self.backbone = backbone
        self.preprocessor = preprocessor

    async def Stream(self, request_iterator, context):
        loop = asyncio.get_running_loop()
        async for obs in request_iterator:
            logger.info(
                "high_node_input_received robot_id=%s session_id=%s seq_id=%s payload_format=%s payload_bytes=%d",
                obs.robot_id,
                obs.session_id,
                obs.seq_id,
                obs.payload_format,
                len(obs.payload),
            )

            batch = decode_lerobot_observation(obs.payload, obs.payload_format)
            batch = self.preprocessor(batch)

            actions = await loop.run_in_executor(None, lambda: self.backbone.predict_action_chunk(batch))

            if isinstance(actions, BatchFeature):
                if "action_pred" in actions:
                    actions = actions["action_pred"]
                else:
                    tensor_values = [v for v in actions.values() if isinstance(v, torch.Tensor)]
                    if not tensor_values:
                        raise TypeError(
                            f"Expected at least one tensor in BatchFeature for actions, got keys={list(actions.keys())}"
                        )
                    actions = tensor_values[0]

            if not isinstance(actions, torch.Tensor):
                raise TypeError(f"Expected actions as Tensor, got {type(actions)} instead")

            actions = actions.detach().to(dtype=torch.float16, device="cpu")
            logger.info(
                "high_node_actions_computed seq_id=%s shape=%s dtype=%s device=%s",
                obs.seq_id,
                tuple(actions.shape),
                str(actions.dtype),
                str(actions.device),
            )

            data_bytes = actions.numpy().tobytes()
            embedding = intent_pb2.Embedding(
                robot_id=obs.robot_id,
                session_id=obs.session_id,
                seq_id=obs.seq_id,
                t_embed_ms=int(time.time() * 1000),
                data=data_bytes,
                shape=list(actions.shape),
                dtype="fp16",
            )

            logger.info(
                "high_node_embedding_sent robot_id=%s session_id=%s seq_id=%s shape=%s dtype=%s bytes=%d",
                embedding.robot_id,
                embedding.session_id,
                embedding.seq_id,
                list(embedding.shape),
                embedding.dtype,
                len(embedding.data),
            )
            yield embedding


class IntentServer:
    """High-node gRPC server that hosts the IntentService."""

    def __init__(self, backbone: PolicyAdapter, preprocessor, port: int = 50051):
        self.port = port
        self.server = grpc.aio.server()
        intent_service = IntentService(backbone=backbone, preprocessor=preprocessor)
        intent_pb2_grpc.add_IntentServiceServicer_to_server(intent_service, self.server)
        self.server.add_insecure_port(f"[::]:{self.port}")

    async def start(self):
        await self.server.start()
        logger.info("high_node_server_started port=%d", self.port)

    async def wait(self):
        await self.server.wait_for_termination()

    async def run_forever(self):
        await self.start()
        await self.wait()

    async def stop(self, grace: float = 0.0):
        await self.server.stop(grace)

