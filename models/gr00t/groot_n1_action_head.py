from lerobot.policies.groot.groot_n1 import GR00TN15Config
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
from lerobot.policies.groot.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

import torch
import tree

import numpy as np

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3

class GR00TN15ActionHead(PreTrainedModel):
    base_model_prefix:str = "_groot_model.backbone"
    supports_gradient_checkpointing = True
    config_class = GR00TN15Config

    def __init__(
        self,
        config: GR00TN15Config,
        local_model_path: str,
    ):
        super().__init__(config)
        self.local_model_path = local_model_path
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            # In inference, action may be omitted or None; validate only when it's a tensor.
            if action is None:
                pass  # allow None during inference
            elif isinstance(action, torch.Tensor):
                shape_ok = (
                    len(action.shape) == 3
                    and action.shape[1] == self.action_horizon
                    and action.shape[2] == self.action_dim
                )
                if not shape_ok:
                    error_msg += f"\n{action.shape=}"
                    detected_error = True
            else:
                # Unexpected non-tensor type provided for action
                error_msg += f"\nInvalid type for action: {type(action)}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def get_action(
        self,
        inputs: dict,
        backbone_embeddings: BatchFeature,
    ) -> BatchFeature:
        action_inputs = self.prepare_input(inputs)
        action_head_outputs = self.action_head.get_action(backbone_embeddings, action_inputs)
        self.validate_data(action_head_outputs, backbone_embeddings, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> BatchFeature:
        self.validate_inputs(inputs)
        action_inputs = self.action_head.prepare_input(inputs)


        def to_device_with_maybe_dtype(x):
            # Cast floating tensors to a memory-efficient compute dtype when requested.
            # Rationale: Upcasting backbone activations to fp32 significantly increases VRAM.
            # When compute_dtype is bfloat16, prefer bf16 for activations to match AMP behavior.
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                if getattr(self, "compute_dtype", None) == "bfloat16":
                    return x.to(self.device, dtype=torch.bfloat16)
                # Fallback: preserve previous behavior if not using bf16 compute
                return x.to(self.device, dtype=self.action_head.dtype)
            # Non-floating tensors: move device only
            return x.to(self.device)

        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load ONLY backbone.* keys from a full checkpoint (single safetensors).
        """
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        print(f"Loading pretrained action head from {pretrained_model_name_or_path}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

       
        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        return pretrained_model