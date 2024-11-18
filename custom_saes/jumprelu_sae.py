import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np
from typing import Optional

import baselines.custom_sae_config as sae_config


class JumpReLUSAE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        hook_layer: int,
        model_name: str = "gemma-2-2b",
        hook_name: Optional[str] = None,
    ):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = sae_config.CustomSAEConfig(
            model_name, d_in=d_model, d_sae=d_sae, hook_name=hook_name, hook_layer=hook_layer
        )

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

    # required as we have device and dtype class attributes
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Update the device and dtype attributes based on the first parameter
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        # Update device and dtype if they were provided
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self


def load_jumprelu_sae(repo_id: str, filename: str, layer: int) -> JumpReLUSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
    )

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

    sae = JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1], layer)
    sae.load_state_dict(pt_params)

    return sae


if __name__ == "__main__":
    repo_id = "google/gemma-scope-2b-pt-res"
    filename = "layer_20/width_16k/average_l0_71/params.npz"

    sae = load_jumprelu_sae(repo_id, filename, 20)
