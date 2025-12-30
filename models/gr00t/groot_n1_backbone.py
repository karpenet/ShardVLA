from lerobot.policies.groot.groot_n1 import GR00TN15Config, EagleBackbone
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

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

class GR00TN15Backbone(PreTrainedModel):
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
        self.backbone = EagleBackbone(**config.backbone_cfg)
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

    def get_embeddings(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return backbone_outputs

    def prepare_input(self, inputs) -> BatchFeature:
        backbone_inputs = self.backbone.prepare_input(inputs)

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

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        return backbone_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load ONLY backbone.* keys from a full checkpoint (single safetensors).
        """
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")

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
        pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)

        return pretrained_model