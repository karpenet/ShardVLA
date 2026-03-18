"""ShardVLA: lightweight sharded VLA transport/runtime utilities.

This package contains only the ShardVLA glue code. It intentionally does not
modify or vendor `lerobot/`; instead it provides adapters around it.
"""

from .config import CommunicationConfig, ShardVLARuntimeConfig

__all__ = [
    "CommunicationConfig",
    "ShardVLARuntimeConfig",
]

