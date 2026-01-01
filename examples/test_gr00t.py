"""
Example usage of InferenceEngine.

This demonstrates how to initialize and use the InferenceEngine
for both high_node (GPU server) and low_node (Jetson) scenarios.
"""

from lerobot.utils.utils import auto_select_torch_device
from inference_engine.inference_server import InferenceServer
from inference_engine.inference_client import InferenceClient

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import OpenVLA utilities
sys.path.append(str(Path(__file__).parent.parent / "models" / "openvla"))

parser = argparse.ArgumentParser(description="InferenceEngine example")
parser.add_argument(
    "--node-type",
    type=str,
    choices=["high_node", "low_node"],
    default="high_node",
    help="Node type: 'high_node' for GPU server, 'low_node' for Jetson",
)
args = parser.parse_args()


from models.gr00t.groot_n1_policy import GrootBackbonePolicy, GrootActionHeadPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig


# Define constants for dummy data
DUMMY_STATE_DIM = 44
DUMMY_ACTION_DIM = 44
DUMMY_ACTION_HORIZON = 16
IMAGE_SIZE = 256
DEVICE = auto_select_torch_device()
MODEL_PATH = "aractingi/bimanual-handover-groot-10k"

# Initialize policy configuration
config = GrootConfig(
        base_model_path=MODEL_PATH,
        n_action_steps=DUMMY_ACTION_HORIZON,
        chunk_size=DUMMY_ACTION_HORIZON,
        image_size=[IMAGE_SIZE, IMAGE_SIZE],
        device=DEVICE,
        embodiment_tag="gr1",
)

print(f"[*] Initializing InferenceEngine for {args.node_type}...")

if args.node_type == "high_node":
    print("[*] High node mode: Will receive observations via ZeroMQ subscriber")
    print("[*] Waiting for observations from low node...")
    backbone = GrootBackbonePolicy.from_pretrained(
        MODEL_PATH,
        strict=False,
    )
    server = InferenceServer(backbone=backbone)
    print("[*] High node initialized. Use server.forward() to get observations and run inference.")

    # Run inference loop
    action = server.infer_single_sample(timeout_ms=1000)
    if action is not None:
        print(f"Predicted action shape: {action.shape}")
    
    print("[*] High node initialized. Use engine.infer_single_sample() to get observations and run inference.")

# Example: Low node usage
elif args.node_type == "low_node":
    print("[*] Low node mode: Will publish observations via ZeroMQ publisher")
    policy = GrootActionHeadPolicy.from_pretrained(
        MODEL_PATH,
        strict=False,
    )
    client = InferenceClient()
    print("[*] Low node initialized.")

    client.infer_single_sample(observation)


print("\n[*] Example usage:")
print("  python inference_engine.py --node-type high_node")
print("  python inference_engine.py --node-type low_node")