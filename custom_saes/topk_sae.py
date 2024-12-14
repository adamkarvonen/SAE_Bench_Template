import torch as t
import einops
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np
from abc import ABC, abstractmethod
import json
from typing import Optional

from custom_saes import custom_sae_config as sae_config


class AutoEncoderTopK(nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    Implementation adapted from saprmarks/dictionary_learning repo.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        hook_name: Optional[str] = None,
    ):
        super().__init__()
        self.activation_dim = d_in
        self.dict_size = d_sae
        self.k = k

        # self.W_enc = nn.Linear(d_in, d_sae)
        # self.W_enc.bias.data.zero_()
        # self.W_dec = nn.Linear(d_sae, d_in, bias=False)
        # self.W_dec.weight.data = self.W_enc.weight.data.clone().T
        # self.b_dec = nn.Parameter(t.zeros(d_in))
        # self.device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        # self.dtype: t.dtype = t.float32
        # self.set_decoder_norm_to_unit_norm()

        self.W_enc = nn.Parameter(t.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(t.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(t.zeros(d_sae))
        self.b_dec = nn.Parameter(t.zeros(d_in))
        self.device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.dtype: t.dtype = t.float32

        if hook_name is None:
            hook_name = f"blocks.{hook_layer}.hook_resid_post"

        self.cfg = sae_config.CustomSAEConfig(
            model_name, d_in=d_in, d_sae=d_sae, hook_name=hook_name, hook_layer=hook_layer
        )

    def encode(self, x: t.Tensor, return_topk: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return x @ self.W_dec + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

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


def load_topk_sae(repo_id: str, filename: str, config_filename: str, layer: int) -> AutoEncoderTopK:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
    )
    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
    )

    pt_params = t.load(path_to_params)
    config = json.load(open(path_to_config))
    print(f"Loaded config: {config}")

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "b_dec": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    # Create the VanillaSAE model
    sae = AutoEncoderTopK(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=config["trainer"]["k"],
        model_name=config["trainer"]["lm_name"],
        hook_layer=layer,
    )

    sae.load_state_dict(renamed_params)

    return sae


if __name__ == "__main__":
    repo_id = "webcrg/additivity"
    filename = "gemma-2-2b_layer-4_width-2pow13_date-1204/trainer_0/ae.pt"
    config_filename = "gemma-2-2b_layer-4_width-2pow13_date-1204/trainer_0/config.json"
    layer = 4

    sae = load_topk_sae(repo_id, filename, config_filename, layer)
    sae = sae.to(sae.device)
    dummy_act = t.randn(1, 20, 2304).to(sae.device)
    print(sae(dummy_act))
