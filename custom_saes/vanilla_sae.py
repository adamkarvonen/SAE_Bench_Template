import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import numpy as np


class VanillaSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
    ):
        print("d_in:", d_in)
        print("d_sae:", d_sae)
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

    def encode(self, input_acts):
        pre_acts = (input_acts - self.b_dec) @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        return acts

    def decode(self, acts):
        return (acts @ self.W_dec) + self.b_dec

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


def load_vanilla_sae(repo_id: str, filename: str, layer: int) -> VanillaSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
    )

    pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

    # Print original keys for debugging
    print("Original keys in state_dict:", pt_params.keys())

    # Map old keys to new keys
    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "bias": "b_dec",
    }

    # Create a new dictionary with renamed keys
    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

    # due to the way torch uses nn.Linear, we need to transpose the weight matrices
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T

    # Print renamed keys for debugging
    print("Renamed keys in state_dict:", renamed_params.keys())

    # Create the VanillaSAE model
    sae = VanillaSAE(d_in=renamed_params["b_dec"].shape[0], d_sae=renamed_params["b_enc"].shape[0])

    sae.load_state_dict(renamed_params)

    return sae


if __name__ == "__main__":
    repo_id = "canrager/lm_sae"
    filename = "pythia70m_sweep_standard_ctx128_0712/resid_post_layer_4/trainer_8/ae.pt"
    layer = 4

    sae = load_vanilla_sae(repo_id, filename, layer)
