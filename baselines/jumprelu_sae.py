import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    context_size: int = 128  # Can be used for auto-interp
    hook_head_index: Optional[int] = None


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, hook_layer: int, model_name: str = "gemma-2-2b"):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = SAEConfig(
            model_name, d_in=d_model, d_sae=d_model, hook_name=hook_name, hook_layer=hook_layer
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
