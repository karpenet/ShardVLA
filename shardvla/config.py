from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommunicationConfig:
    """Configuration for sharded VLA comms.

    One robot ↔ one server over Wi‑Fi, using a single long‑lived gRPC bidi stream.
    """

    server_addr: str = "127.0.0.1:50051"

    # Keep this tiny to avoid stale buildup (latest-wins semantics).
    max_pending_observations: int = 2

    # Staleness cutoff used by the control loop (seconds).
    max_embedding_age_s: float = 0.2

    # Reconnect behavior (seconds).
    reconnect_backoff_initial_s: float = 0.2
    reconnect_backoff_max_s: float = 2.0

    # Session metadata
    robot_id: str = "robot1"
    session_id: str = "sess-001"


@dataclass(frozen=True)
class ShardVLARuntimeConfig:
    """Runtime orchestration config (low node)."""

    comms: CommunicationConfig = CommunicationConfig()
    sensing_hz: float = 12.0
    control_hz: float = 100.0
    max_embedding_age_s: float | None = None

    def with_defaults_applied(self) -> "ShardVLARuntimeConfig":
        """Fill derived defaults without mutating the dataclass."""
        max_age = self.max_embedding_age_s
        if max_age is None:
            max_age = self.comms.max_embedding_age_s
        return ShardVLARuntimeConfig(
            comms=self.comms,
            sensing_hz=self.sensing_hz,
            control_hz=self.control_hz,
            max_embedding_age_s=max_age,
        )

