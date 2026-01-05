"""
ZeroMQ observation publisher for low-compute node.

Publishes sensor observations (images + robot state) to high-compute node.
Uses ZeroMQ PUSH/PULL pattern for fire-and-forget semantics.

Why ZeroMQ PUSH/PULL:
- Non-blocking: Low node never waits for high node
- Throughput-optimized: Handles larger payloads efficiently
- Resilient: Missing frames acceptable, control loop continues
"""

import io
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import zmq

from ..config import Config


class ObservationPublisher:
    """
    Publishes observations to high-compute node via ZeroMQ.
    
    Designed for low-compute node (Jetson) that owns sensors and actuators.
    Publishes at configurable rate (1-5 Hz) without blocking control loop.
    """
    
    def __init__(self, config: Config):
        """
        Initialize observation publisher.
        
        Args:
            config: Configuration object with observation settings
        """
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        
        # Configure socket for throughput (not latency)
        self.socket.setsockopt(zmq.SNDTIMEO, config.observation.zmq_send_timeout_ms)
        self.socket.setsockopt(zmq.SNDHWM, config.observation.zmq_high_water_mark)
        
        # Connect to high node
        # Note: For low node, we connect (not bind) to high node's address
        self.socket.connect(config.observation.zmq_sub_address)
        
        self.sequence_number = 0
        self.last_publish_time = 0.0
        self.min_publish_interval = 1.0 / config.observation.publish_rate_hz

    def publish_obs(self, obs_dict: Dict) -> bool:
        # Rate limiting: don't publish faster than configured rate
        current_time = time.time()
        if current_time - self.last_publish_time < self.min_publish_interval:
            return False
        
        try:
            metadata, frames = [], []
            header = {"sequence_number": self.sequence_number}

            for name, data_tensor in obs_dict.items():
                if name == "task":
                    header["task"] = data_tensor
                else:
                    data_tensor = np.ascontiguousarray(data_tensor)  # avoid copies later
                    metadata.append({"name": name, "dtype": str(data_tensor.dtype), "shape": data_tensor.shape})
                    frames.append(memoryview(data_tensor))

            header["meta"] = metadata

            # Send via ZeroMQ (non-blocking with timeout)
            try:
                self.socket.send_json(header, flags=zmq.NOBLOCK | zmq.SNDMORE)
                for i, fr in enumerate(frames):
                    self.socket.send(fr, flags=(zmq.NOBLOCK | zmq.SNDMORE if i < len(frames)-1 else 0), copy=False)
                self.sequence_number += 1
                self.last_publish_time = current_time
                return True
            except zmq.Again:
                # Socket buffer full, drop frame (acceptable for throughput-oriented channel)
                return False
                
        except Exception as e:
            print(f"Error publishing observation: {e}")
            return False


    def publish(
        self,
        image: np.ndarray,
        robot_state: Dict,
        task_label: Optional[str] = None
    ) -> bool:
        """
        Publish observation to high node.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            robot_state: Dictionary with keys:
                - ee_position: [x, y, z]
                - ee_orientation: [qx, qy, qz, qw] (quaternion)
                - joint_positions: [j1, j2, ...]
                - joint_velocities: [v1, v2, ...]
                - gripper_position: float
                - gripper_velocity: float
            task_label: Optional task description string
        
        Returns:
            True if published successfully, False otherwise
        """
        # Rate limiting: don't publish faster than configured rate
        current_time = time.time()
        if current_time - self.last_publish_time < self.min_publish_interval:
            return False
        
        try:
            # Compress image to JPEG
            image_jpeg = self._compress_image(image)
            
            # Check size constraint (≤ 500 KB)
            if len(image_jpeg) > self.config.observation.max_image_size_kb * 1024:
                # Reduce quality if too large
                image_jpeg = self._compress_image(image, quality=70)
                if len(image_jpeg) > self.config.observation.max_image_size_kb * 1024:
                    print(f"Warning: Image still too large after compression: {len(image_jpeg)} bytes")
                    return False
            
            # Serialize observation
            # Note: In production, use FlatBuffers. For now, use simple binary format
            message = self._serialize_observation(
                image_jpeg,
                image.shape,
                robot_state,
                task_label
            )
            
            # Send via ZeroMQ (non-blocking with timeout)
            try:
                self.socket.send(message, zmq.NOBLOCK)
                self.sequence_number += 1
                self.last_publish_time = current_time
                return True
            except zmq.Again:
                # Socket buffer full, drop frame (acceptable for throughput-oriented channel)
                return False
                
        except Exception as e:
            print(f"Error publishing observation: {e}")
            return False
    
    def _serialize_observation(
        self,
        obs_dict: Dict
    ) -> bytes:
        """
        Serialize observation to binary format.
        
        Note: In production, replace with FlatBuffers serialization.
        This is a simple binary format for demonstration.
        """
        return NotImplementedError
    
    def close(self):
        """Close socket and context."""
        self.socket.close()
        self.context.term()

