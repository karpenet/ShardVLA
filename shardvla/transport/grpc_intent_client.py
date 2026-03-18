from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import grpc

from shardvla.transport.protobuf import intent_pb2, intent_pb2_grpc

T = TypeVar("T")

logger = logging.getLogger(__name__)


class LatestValue(Generic[T]):
    """Async-safe 'latest wins' container."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._value: Optional[T] = None

    async def set(self, v: T) -> None:
        async with self._lock:
            self._value = v

    async def get(self) -> Optional[T]:
        async with self._lock:
            return self._value


class LatestWinsQueue(Generic[T]):
    """Bounded queue that drops the oldest item when full."""

    def __init__(self, maxsize: int = 2) -> None:
        self._q: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)

    def put_nowait_drop_oldest(self, item: T) -> None:
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            try:
                _ = self._q.get_nowait()  # drop oldest
                self._q.task_done()
            except asyncio.QueueEmpty:
                pass
            self._q.put_nowait(item)

    async def get(self) -> T:
        return await self._q.get()

    def task_done(self) -> None:
        self._q.task_done()


@dataclass(frozen=True)
class IntentClientConfig:
    server_addr: str
    robot_id: str
    session_id: str
    max_pending_observations: int = 2
    reconnect_backoff_initial_s: float = 0.2
    reconnect_backoff_max_s: float = 2.0


class IntentClient:
    """Robot-side client owning a single gRPC bidi stream."""

    def __init__(self, cfg: IntentClientConfig) -> None:
        self.cfg = cfg
        self._stop = asyncio.Event()

        self._obs_q: LatestWinsQueue[intent_pb2.Observation] = LatestWinsQueue(
            maxsize=cfg.max_pending_observations
        )
        self.latest_embedding: LatestValue[intent_pb2.Embedding] = LatestValue()

    def publish_observation(
        self,
        seq_id: int,
        t_capture_ms: int,
        payload: bytes,
        payload_format: str = "json:ObservationV1",
    ) -> None:
        msg = intent_pb2.Observation(
            robot_id=self.cfg.robot_id,
            session_id=self.cfg.session_id,
            seq_id=seq_id,
            t_capture_ms=t_capture_ms,
            payload=payload,
            payload_format=payload_format,
        )
        self._obs_q.put_nowait_drop_oldest(msg)

    async def run(self) -> None:
        """Connect, stream, and reconnect on failure."""
        backoff = self.cfg.reconnect_backoff_initial_s

        while not self._stop.is_set():
            try:
                async with grpc.aio.insecure_channel(self.cfg.server_addr) as channel:
                    stub = intent_pb2_grpc.IntentServiceStub(channel)
                    call = stub.Stream()

                    tx_task = asyncio.create_task(self._tx_loop(call), name="intent_tx")
                    rx_task = asyncio.create_task(self._rx_loop(call), name="intent_rx")

                    done, pending = await asyncio.wait(
                        {tx_task, rx_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for t in pending:
                        t.cancel()

                    for t in done:
                        exc = t.exception()
                        if exc:
                            raise exc

                backoff = self.cfg.reconnect_backoff_initial_s

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("IntentClient stream error; reconnecting")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, self.cfg.reconnect_backoff_max_s)

    async def stop(self) -> None:
        self._stop.set()

    async def _tx_loop(self, call: grpc.aio.StreamStreamCall) -> None:
        while not self._stop.is_set():
            obs = await self._obs_q.get()
            try:
                await call.write(obs)
            finally:
                self._obs_q.task_done()

    async def _rx_loop(self, call: grpc.aio.StreamStreamCall) -> None:
        async for emb in call:
            await self.latest_embedding.set(emb)

        raise RuntimeError("Server closed embedding stream")

