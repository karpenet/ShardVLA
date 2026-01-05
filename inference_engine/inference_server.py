from copy import deepcopy
from communication.observation.subscriber import ObservationSubscriber
from communication.config import Config

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from typing import Any, Dict

import gc
import numpy as np
import threading
import torch

class InferenceServer:
    def __init__(
        self, 
        config: Config = Config(),
        backbone: PreTrainedPolicy | None = None,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    ):
        self.observation_subscriber = ObservationSubscriber(config)
        self._run_stopped = threading.Event()
        self._thread = None

        self.backbone = backbone
        self.backbone.eval()

        self.preprocessor = preprocessor

    def start(self):
        self._run_stopped.clear()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._run_stopped.set()
        if self._thread:
            self._thread.join(timeout=5)
        
        # close sockets/context if you own them
        self.observation_subscriber.close()
        self.cleanup_memory()

    def cleanup_memory(self):
        """Clean up GPU/MPS memory to prevent OOM errors between tests."""
        print("\nCleaning up memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Memory cleanup complete.")

    def run_forever(self):
        # runs until stop() is called
        while not self._run_stopped.is_set():
            obs = self.observation_subscriber.get_latest_observation(timeout_ms=100)  # pick a small timeout
            if obs is None:
                continue  # no message yet, keep looping

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for k, v in list(obs.items()):
                if k == "task":
                    # Always convert to a list (language prompt expected as list)
                    obs[k] = [v] if not isinstance(v, list) else v
                elif isinstance(v, np.ndarray):
                    obs[k] = torch.from_numpy(v).to(device)

            # Do inference / processing
            batch = self.preprocessor(deepcopy(obs))

            with torch.no_grad():
                embeddings = self.backbone.predict_action_chunk(batch)

    def forward(self, observation: Dict[str, np.ndarray] | None = None, timeout_ms: int = 100) -> torch.Tensor | None:
        # Get observation from subscriber
        if observation is None:
            obs_dict = self.observation_subscriber.get_latest_observation(timeout_ms=timeout_ms)
            if obs_dict is None:
                return None
        else:
            obs_dict = observation
        
        # Convert numpy arrays to torch tensors and add batch dimension
        batch = {}
        device = next(self.policy.parameters()).device
        
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                # Convert to tensor and add batch dimension
                tensor = torch.from_numpy(value).to(device)
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor
            elif isinstance(value, torch.Tensor):
                # Already a tensor, just ensure batch dimension and device
                tensor = value.to(device)
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor
            else:
                batch[key] = value
        
        # Run inference
        with torch.no_grad():
            action = self.policy.select_action(batch)
        
        return action