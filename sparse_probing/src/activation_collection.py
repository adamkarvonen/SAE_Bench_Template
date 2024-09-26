import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable, Optional
from jaxtyping import Int, Float, jaxtyped, BFloat16
from beartype import beartype
import einops
from transformer_lens import HookedTransformer
from sae_lens import SAE

LLM_NAME_TO_BATCH_SIZE = {
    "pythia-70m-deduped": 500,
    "gemma-2-2b": 32,
}


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_all_llm_activations(
    tokenized_inputs_dict: dict[str, dict[str, Int[torch.Tensor, "dataset_size seq_len"]]],
    model: HookedTransformer,
    batch_size: int,
    hook_name: str,
) -> dict[str, Float[torch.Tensor, "batch_size seq_len d_model"]]:
    """VERY IMPORTANT NOTE: We zero out masked token activations in this function. Later, we ignore zeroed activations."""
    all_classes_acts_BLD = {}

    for class_name in tokenized_inputs_dict:
        all_acts_BLD = []
        tokenized_inputs = tokenized_inputs_dict[class_name]

        for i in tqdm(
            range(len(tokenized_inputs["input_ids"]) // batch_size),
            desc=f"Collecting activations for class {class_name}",
        ):
            tokens_BL = tokenized_inputs["input_ids"][i * batch_size : (i + 1) * batch_size]
            attention_mask_BL = tokenized_inputs["attention_mask"][
                i * batch_size : (i + 1) * batch_size
            ]

            acts_BLD = None

            def activation_hook(resid_BLD: torch.Tensor, hook):
                nonlocal acts_BLD
                acts_BLD = resid_BLD

            model.run_with_hooks(
                tokens_BL, return_type=None, fwd_hooks=[(hook_name, activation_hook)]
            )

            acts_BLD = acts_BLD * attention_mask_BL[:, :, None]
            all_acts_BLD.append(acts_BLD)

        all_acts_BLD = torch.cat(all_acts_BLD, dim=0)

        all_classes_acts_BLD[class_name] = all_acts_BLD

    return all_classes_acts_BLD


@jaxtyped(typechecker=beartype)
def create_meaned_model_activations(
    all_llm_activations_BLD: dict[str, Float[torch.Tensor, "batch_size seq_len d_model"]],
) -> dict[str, Float[torch.Tensor, "batch_size d_model"]]:
    """VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    all_llm_activations_BD = {}
    for class_name in all_llm_activations_BLD:
        acts_BLD = all_llm_activations_BLD[class_name]
        dtype = acts_BLD.dtype

        activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        meaned_acts_BD = einops.reduce(acts_BLD, "B L D -> B D", "sum") / nonzero_acts_B[:, None]
        all_llm_activations_BD[class_name] = meaned_acts_BD

    return all_llm_activations_BD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_sae_meaned_activations(
    all_llm_activations_BLD: dict[str, Float[torch.Tensor, "batch_size seq_len d_model"]],
    sae: SAE,
    sae_batch_size: int,
    dtype: torch.dtype,
) -> dict[str, Float[torch.Tensor, "batch_size d_sae"]]:
    """VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""
    all_sae_activations_BF = {}
    for class_name in all_llm_activations_BLD:
        all_acts_BLD = all_llm_activations_BLD[class_name]

        all_acts_BF = []

        for i in range(len(all_acts_BLD) // sae_batch_size):
            acts_BLD = all_acts_BLD[i * sae_batch_size : (i + 1) * sae_batch_size]
            acts_BLF = sae.encode(acts_BLD)

            activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
            nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
            nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

            acts_BLF = acts_BLF * nonzero_acts_BL[:, :, None]
            acts_BF = einops.reduce(acts_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None]
            acts_BF = acts_BF.to(dtype=dtype)

            all_acts_BF.append(acts_BF)

        all_acts_BF = torch.cat(all_acts_BF, dim=0)
        all_sae_activations_BF[class_name] = all_acts_BF

    return all_sae_activations_BF
