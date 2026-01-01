import torch
import numpy as np

from communication.observation.publisher import ObservationPublisher
from communication.config import Config


class InferenceClient:
    def __init__(self, config: Config = Config()):
        self.observation_publisher = ObservationPublisher(config)

    def infer_single_sample(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        if observation is None:
                raise ValueError("observation must be provided for low_node")
            
        # Publish observation
        success = self.observation_publisher.publish_obs(observation)
        if not success:
            return None
        
        # For low_node, we don't run inference locally
        # The action would come back via gRPC intent server
        # Return None to indicate action should be retrieved separately
        return None