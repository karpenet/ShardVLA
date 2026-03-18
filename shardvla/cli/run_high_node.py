from __future__ import annotations

import argparse
import asyncio
import logging

from shardvla.policies.groot import GrootPolicyAdapter, GrootPolicyConfig
from shardvla.transport.grpc_intent_server import IntentServer


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="ShardVLA high node (gRPC intent server)")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--model-path",
        type=str,
        default="aractingi/bimanual-handover-groot-10k",
        help="HuggingFace model repo or local path for GR00T.",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

    backbone = GrootPolicyAdapter(GrootPolicyConfig(model_path=args.model_path, strict=False))
    preprocessor, _ = make_groot_pre_post_processors(config=backbone.config, dataset_stats=None)

    server = IntentServer(backbone=backbone, preprocessor=preprocessor, port=args.port)
    await server.run_forever()


if __name__ == "__main__":
    asyncio.run(main())

