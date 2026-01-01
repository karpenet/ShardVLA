"""
ZeroMQ observation subscriber for high-compute node.

Subscribes to sensor observations from low-compute node.
Uses ZeroMQ PUSH/PULL pattern (subscriber pulls from publisher).

Why ZeroMQ PULL:
- Non-blocking: High node processes as fast as possible
- Handles bursts: Queue can buffer multiple observations
- Stateless: No connection state, resilient to network issues
"""

import io
import struct
import time
from typing import Dict, Optional, Tuple

import numpy as np
import zmq
from PIL import Image

from ..config import Config


class Observation:
    """Deserialized observation data."""
    
    def __init__(
        self,
        sequence_number: int,
        timestamp: float,
        image: np.ndarray,
        robot_state: Dict,
        task_label: Optional[str] = None
    ):
        self.sequence_number = sequence_number
        self.timestamp = timestamp
        self.image = image
        self.robot_state = robot_state
        self.task_label = task_label


class ObservationSubscriber:
    """
    Subscribes to observations from low-compute node via ZeroMQ.
    
    Designed for high-compute node that runs OpenVLA inference.
    Pulls observations from queue and processes them as fast as possible.
    """

    NP_DTYPE = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
}
    
    def __init__(self, config: Config):
        """
        Initialize observation subscriber.
        
        Args:
            config: Configuration object with observation settings
        """
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        
        # Configure socket for throughput
        self.socket.setsockopt(zmq.RCVTIMEO, config.observation.zmq_receive_timeout_ms)
        self.socket.setsockopt(zmq.RCVHWM, config.observation.zmq_high_water_mark)
        
        # Bind to address (high node binds, low node connects)
        self.socket.bind(config.observation.zmq_pub_address)
        
        self.last_sequence = -1
        self.dropped_frames = 0
    
    def get_latest_observation(self, timeout_ms: Optional[int] = None) -> Optional[Observation]:
        """
        Get latest observation from queue.
        
        Args:
            timeout_ms: Optional timeout in milliseconds (uses config default if None)
        
        Returns:
            Observation object if available, None if timeout or error
        """
        if timeout_ms is None:
            timeout_ms = self.config.observation.zmq_receive_timeout_ms
        
        # Set timeout temporarily
        old_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        
        try:
            # Receive message (non-blocking with timeout)
            try:
                header = self.socket.recv_json(zmq.NOBLOCK)  # frame 0
                input_dict = {}

                for meta in header["meta"]:
                    msg = self.socket.recv(copy=False)      # next frame
                    data_tensor = memoryview(msg.buffer)      # zero-copy view
                    data_tensor = np.frombuffer(data_tensor, dtype=np.dtype(self.NP_DTYPE[meta["dtype"]])).reshape(meta["shape"])
                    input_dict[meta["name"]] = data_tensor
            except zmq.Again:
                return None
            
            # # Deserialize observation
            # obs = self._deserialize_observation(message)
            
            # Track dropped frames
            if self.last_sequence >= 0:
                dropped = header["sequence_number"] - self.last_sequence - 1
                if dropped > 0:
                    self.dropped_frames += dropped
                    print(f"Warning: Dropped {dropped} frame(s), total: {self.dropped_frames}")
            
            self.last_sequence = header["sequence_number"]
            return input_dict
            
        except Exception as e:
            print(f"Error receiving observation: {e}")
            return None
        finally:
            # Restore original timeout
            self.socket.setsockopt(zmq.RCVTIMEO, old_timeout)
    
    def _deserialize_observation(self, message: bytes) -> Observation:
        """
        Deserialize observation from binary format.
        
        Note: In production, replace with FlatBuffers deserialization.
        This matches the format used in publisher._serialize_observation.
        """
        return NotImplementedError
    
    def close(self):
        """Close socket and context."""
        self.socket.close()
        self.context.term()

