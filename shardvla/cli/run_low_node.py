from __future__ import annotations

import argparse
import asyncio
import logging

from shardvla.config import CommunicationConfig
from shardvla.runtime.bridge_runtime import ShardVLARuntime


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="ShardVLA low node (runtime bridge)")
    parser.add_argument("--server-addr", type=str, default="127.0.0.1:50051")
    parser.add_argument("--robot-id", type=str, default="robot1")
    parser.add_argument("--session-id", type=str, default="sess-001")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _configure_logging(args.log_level)

    cfg = CommunicationConfig(
        server_addr=args.server_addr,
        robot_id=args.robot_id,
        session_id=args.session_id,
    )
    runtime = ShardVLARuntime(cfg=cfg)
    await runtime.run_forever()


if __name__ == "__main__":
    asyncio.run(main())

