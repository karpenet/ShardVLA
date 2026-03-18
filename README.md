## ShardVLA

ShardVLA is a lightweight glue layer around LeRobot policies for running a **sharded visual-language-action (VLA)** stack:

## Running the high and low nodes

### 1. High node (gRPC server)

Run this in **terminal 1**:

```bash
cd /home/getpanked/projects/ShardVLA

python -m shardvla.cli.run_high_node \
  --port 50051 \
  --log-level INFO \
  --model-path aractingi/bimanual-handover-groot-10k  # or a local path
```

Flags:

- `--port`: gRPC port to listen on (default `50051`).
- `--log-level`: logging level (`DEBUG`, `INFO`, `WARNING`, ...).
- `--model-path`: HuggingFace repo ID or local path for the GR00T model.

> Note: in offline / restricted environments you will need to point `--model-path` at a **local** GR00T checkpoint instead of a remote HF repo.

### 2. Low node (runtime bridge)

Run this in **terminal 2**:

```bash
cd /home/getpanked/projects/ShardVLA

python -m shardvla.cli.run_low_node \
  --server-addr 127.0.0.1:50051 \
  --robot-id robot1 \
  --session-id sess-001 \
  --log-level INFO
```

---

## Using ShardVLA as a library

You can embed ShardVLA components directly in your own code:

```python
from shardvla.runtime import ShardVLARuntime
from shardvla.transport.grpc_intent_server import IntentServer
from shardvla.policies.groot import GrootPolicyAdapter, GrootPolicyConfig
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors


# High node wiring
backbone = GrootPolicyAdapter(GrootPolicyConfig(model_path="your/model/path", strict=False))
preprocessor, _ = make_groot_pre_post_processors(config=backbone.config, dataset_stats=None)
server = IntentServer(backbone=backbone, preprocessor=preprocessor, port=50051)

# Low node wiring
runtime = ShardVLARuntime()
```

This separation lets you swap `PolicyAdapter` implementations (e.g., dummy policies for testing) without touching the transport or runtime code.

