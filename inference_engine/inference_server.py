from typing import Dict
import torch
import numpy as np

from communication.observation.subscriber import ObservationSubscriber
from communication.config import Config
from models.gr00t.groot_n1_backbone import GR00TN15Backbone
import threading

class InferenceServer:
    def __init__(
        self, 
        config: Config = Config(),
        backbone: GR00TN15Backbone | None = None,
    ):
        self.backbone = backbone
        self.observation_subscriber = ObservationSubscriber(config)

        self._run_stopped = threading.Event()
        self._thread = None

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


    def run_forever(self):
        # runs until stop() is called
        while not self._run_stopped.is_set():
            obs = self.observation_subscriber.get_latest_observation(timeout_ms=100)  # pick a small timeout
            if obs is None:
                continue  # no message yet, keep looping

            # TODO: do inference / processing



    def forward(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
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