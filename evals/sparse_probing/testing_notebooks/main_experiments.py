# %%

import copy
import json

# TODO make import from shared directory more robust
# I wanted to avoid the pip install -e . in the shared directory, but maybe that's the best way to do it
import os
import random
import sys
import time
from dataclasses import asdict

import eval_config
import pandas as pd
import probe_training
import torch
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm
from transformer_lens import HookedTransformer

import sae_bench_utils.activation_collection as activation_collection
import sae_bench_utils.dataset_utils as dataset_utils
import utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sae_bench_utils.formatting_utils as formatting_utils


def average_test_accuracy(test_accuracies: dict[str, float]) -> float:
    return sum(test_accuracies.values()) / len(test_accuracies)


start_time = time.time()

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

config = eval_config.EvalConfig()

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

# populate selected_saes_dict
for release in config.sae_releases:
    if "gemma-scope" in release:
        config.selected_saes_dict[release] = (
            formatting_utils.find_gemmascope_average_l0_sae_names(config.layer)
        )
    else:
        config.selected_saes_dict[release] = formatting_utils.filter_sae_names(
            sae_names=release,
            layers=[config.layer],
            include_checkpoints=config.include_checkpoints,
            trainer_ids=config.trainer_ids,
        )

    print(f"SAE release: {release}, SAEs: {config.selected_saes_dict[release]}")

# %%

# TODO: Make this nicer.
sae_map_df = pd.DataFrame.from_records(
    {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
).T

results_dict = {}
results_dict["custom_eval_results"] = {}

llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

model = HookedTransformer.from_pretrained_no_processing(
    config.model_name, device=device, dtype=llm_dtype
)

train_df, test_df = dataset_utils.load_huggingface_dataset(config.dataset_name)
train_data, test_data = dataset_utils.get_multi_label_train_test_data(
    train_df,
    test_df,
    config.dataset_name,
    config.probe_train_set_size,
    config.probe_test_set_size,
    config.random_seed,
)

train_data = utils.filter_dataset(train_data, config.chosen_classes)
test_data = utils.filter_dataset(test_data, config.chosen_classes)

train_data = utils.tokenize_data(train_data, model.tokenizer, config.context_length, device)
test_data = utils.tokenize_data(test_data, model.tokenizer, config.context_length, device)

print(f"Running evaluation for layer {config.layer}")
hook_name = f"blocks.{config.layer}.hook_resid_post"

all_train_acts_BLD = activation_collection.get_all_llm_activations(
    train_data, model, llm_batch_size, hook_name
)
all_test_acts_BLD = activation_collection.get_all_llm_activations(
    test_data, model, llm_batch_size, hook_name
)

all_train_acts_BD = activation_collection.create_meaned_model_activations(all_train_acts_BLD)
all_test_acts_BD = activation_collection.create_meaned_model_activations(all_test_acts_BLD)

llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
    all_train_acts_BD,
    all_test_acts_BD,
    select_top_k=None,
)

llm_results = {"llm_test_accuracy": average_test_accuracy(llm_test_accuracies)}

# %%

from typing import Optional

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import sae_bench_utils.dataset_info as dataset_info


class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@jaxtyped(typechecker=beartype)
def prepare_probe_data(
    all_activations: dict[str, Float[torch.Tensor, "num_datapoints d_model"]],
    class_idx: str,
) -> tuple[Float[torch.Tensor, "batch_size d_model"], Int[torch.Tensor, "batch_size"]]:
    positive_acts_BD = all_activations[class_idx]
    device = positive_acts_BD.device

    num_positive = len(positive_acts_BD)

    # Collect all negative class activations and labels
    negative_acts = []
    for idx, acts in all_activations.items():
        if idx != class_idx:
            negative_acts.append(acts)

    negative_acts = torch.cat(negative_acts)

    # Randomly select num_positive samples from negative class
    indices = torch.randperm(len(negative_acts))[:num_positive]
    selected_negative_acts_BD = negative_acts[indices]

    assert selected_negative_acts_BD.shape == positive_acts_BD.shape

    # Combine positive and negative samples
    combined_acts = torch.cat([positive_acts_BD, selected_negative_acts_BD])

    combined_labels = torch.empty(len(combined_acts), dtype=torch.int, device=device)
    combined_labels[:num_positive] = dataset_info.POSITIVE_CLASS_LABEL
    combined_labels[num_positive:] = dataset_info.NEGATIVE_CLASS_LABEL

    # Shuffle the combined data
    shuffle_indices = torch.randperm(len(combined_acts))
    shuffled_acts = combined_acts[shuffle_indices]
    shuffled_labels = combined_labels[shuffle_indices]

    return shuffled_acts, shuffled_labels


@jaxtyped(typechecker=beartype)
def get_top_k_mean_diff_mask(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
) -> Bool[torch.Tensor, "k"]:
    positive_mask_B = labels_B == dataset_info.POSITIVE_CLASS_LABEL
    negative_mask_B = labels_B == dataset_info.NEGATIVE_CLASS_LABEL

    positive_distribution_D = acts_BD[positive_mask_B].mean(dim=0)
    negative_distribution_D = acts_BD[negative_mask_B].mean(dim=0)
    distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
    top_k_indices_D = torch.argsort(distribution_diff_D, descending=True)[:k]

    mask_D = torch.ones(acts_BD.shape[1], dtype=torch.bool, device=acts_BD.device)
    mask_D[top_k_indices_D] = False

    return mask_D


@jaxtyped(typechecker=beartype)
def apply_topk_mask_gpu(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()
    masked_acts_BD[:, mask_D] = 0.0

    return masked_acts_BD


@jaxtyped(typechecker=beartype)
def apply_topk_mask_sklearn(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()

    masked_acts_BD = masked_acts_BD[:, ~mask_D]

    return masked_acts_BD


@beartype
def train_sklearn_probe(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    max_iter: int = 100,  # non-default sklearn value, increased due to convergence warnings
    C: float = 1.0,  # default sklearn value
    verbose: bool = False,
    l1_ratio: Optional[float] = None,
) -> tuple[LogisticRegression, float]:
    # Convert torch tensors to numpy arrays
    train_inputs_np = train_inputs.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_inputs_np = test_inputs.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Initialize the LogisticRegression model
    if l1_ratio is not None:
        # Use Elastic Net regularization
        probe = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            verbose=int(verbose),
        )
    else:
        # Use L2 regularization
        probe = LogisticRegression(penalty="l2", C=C, max_iter=max_iter, verbose=int(verbose))

    # Train the model
    probe.fit(train_inputs_np, train_labels_np)

    # Compute accuracies
    train_accuracy = accuracy_score(train_labels_np, probe.predict(train_inputs_np))
    test_accuracy = accuracy_score(test_labels_np, probe.predict(test_inputs_np))

    if verbose:
        print(f"\nTraining completed.")
        print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}\n")

    return probe, test_accuracy


# Helper function to test the probe
@beartype
def test_sklearn_probe(
    inputs: Float[torch.Tensor, "dataset_size d_model"],
    labels: Int[torch.Tensor, "dataset_size"],
    probe: LogisticRegression,
) -> float:
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions = probe.predict(inputs_np)
    return accuracy_score(labels_np, predictions)


@jaxtyped(typechecker=beartype)
@torch.no_grad
def test_probe_gpu(
    inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    labels: Int[torch.Tensor, "test_dataset_size"],
    batch_size: int,
    probe: Probe,
) -> float:
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for i in range(0, len(labels), batch_size):
            acts_BD = inputs[i : i + batch_size]
            labels_B = labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            correct_B = (preds_B == labels_B).float()

            all_corrects.append(correct_B)
            corrects_0.append(correct_B[labels_B == 0])
            corrects_1.append(correct_B[labels_B == 1])

            loss = criterion(logits_B, labels_B.to(dtype=probe.net.weight.dtype))
            losses.append(loss)

        accuracy_all = torch.cat(all_corrects).mean().item()
        accuracy_0 = torch.cat(corrects_0).mean().item() if corrects_0 else 0.0
        accuracy_1 = torch.cat(corrects_1).mean().item() if corrects_1 else 0.0
        all_loss = torch.stack(losses).mean().item()

    return accuracy_all


@jaxtyped(typechecker=beartype)
def train_probe_gpu(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    verbose: bool = False,
    l1_penalty: Optional[float] = None,
    early_stopping_patience: int = 5,
):  # tuple[Probe, float]:
    device = train_inputs.device
    model_dtype = train_inputs.dtype

    print(f"Training probe with dim: {dim}, device: {device}, dtype: {model_dtype}")

    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_test_accuracy = -1.0
    best_probe = None
    patience_counter = 0

    for epoch in range(epochs):
        for i in range(0, len(train_inputs), batch_size):
            acts_BD = train_inputs[i : i + batch_size]
            labels_B = train_labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            loss = criterion(
                logits_B, labels_B.clone().detach().to(device=device, dtype=model_dtype)
            )

            if l1_penalty is not None:
                l1_loss = l1_penalty * torch.sum(torch.abs(probe.net.weight))
                loss += l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = test_probe_gpu(train_inputs, train_labels, batch_size, probe)
        test_accuracy = test_probe_gpu(test_inputs, test_labels, batch_size, probe)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_probe = copy.deepcopy(probe)
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}"
            )

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(type(best_probe))

    return best_probe, best_test_accuracy


@jaxtyped(typechecker=beartype)
def train_probe_on_activations(
    train_activations: dict[str, Float[torch.Tensor, "train_dataset_size d_model"]],
    test_activations: dict[str, Float[torch.Tensor, "test_dataset_size d_model"]],
    select_top_k: Optional[int] = None,
):  # -> tuple[dict[str, LogisticRegression], dict[str, float]]:
    torch.set_grad_enabled(True)

    probes, test_accuracies = {}, {}

    for profession in train_activations.keys():
        train_acts, train_labels = prepare_probe_data(train_activations, profession)

        test_acts, test_labels = prepare_probe_data(test_activations, profession)

        if select_top_k is not None:
            activation_mask_D = get_top_k_mean_diff_mask(
                train_acts, train_labels, select_top_k
            )
            train_acts = apply_topk_mask_sklearn(train_acts, activation_mask_D)
            test_acts = apply_topk_mask_sklearn(test_acts, activation_mask_D)

            # train_acts = apply_topk_mask_gpu(train_acts, activation_mask_D)
            # test_acts = apply_topk_mask_gpu(test_acts, activation_mask_D)

        activation_dim = train_acts.shape[1]

        print(f"Num non-zero elements: {activation_dim}")

        probe, test_accuracy = train_sklearn_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels,
            verbose=False,
        )

        # probe, test_accuracy = train_probe_gpu(
        #     train_acts,
        #     train_labels,
        #     test_acts,
        #     test_labels,
        #     dim=activation_dim,
        #     batch_size=1000,
        #     epochs=100,
        #     lr=1e-2,
        #     verbose=False,
        #     early_stopping_patience=10,
        # )

        print(f"\nTest accuracy for {profession}: {test_accuracy}")

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

    return probes, test_accuracies


# %%

random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

config.k_values = [100]

for k in config.k_values:
    llm_top_k_probes, llm_top_k_test_accuracies = train_probe_on_activations(
        all_train_acts_BD,
        all_test_acts_BD,
        select_top_k=k,
    )
    llm_results[f"llm_top_{k}_test_accuracy"] = average_test_accuracy(
        llm_top_k_test_accuracies
    )
    print(f"Top {k} test accuracy: {llm_results[f'llm_top_{k}_test_accuracy']}")

# %%

sae_release = None

for sae_release in config.selected_saes_dict:
    sae_release = sae_release
print(
    f"Running evaluation for SAE release: {sae_release}, SAEs: {config.selected_saes_dict[sae_release]}"
)
sae_id_to_name_map = sae_map_df.saes_map[sae_release]
sae_name_to_id_map = {v: k for k, v in sae_id_to_name_map.items()}

for sae_name in tqdm(
    config.selected_saes_dict[sae_release],
    desc="Running SAE evaluation on all selected SAEs",
):
    sae_id = sae_name_to_id_map[sae_name]

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae = sae.to(device=device)

# %%

start_time = time.time()
config.sae_batch_size = 125

all_sae_train_acts_BF = activation_collection.get_sae_meaned_activations(
    all_train_acts_BLD, sae, config.sae_batch_size, llm_dtype
)
all_sae_test_acts_BF = activation_collection.get_sae_meaned_activations(
    all_test_acts_BLD, sae, config.sae_batch_size, llm_dtype
)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")

# %%

start_time = time.time()

sae_probes, sae_test_accuracies = train_probe_on_activations(
    all_sae_train_acts_BF,
    all_sae_test_acts_BF,
    select_top_k=None,
)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")

start_time = time.time()

results_dict["custom_eval_results"][sae_name] = {}

for llm_result_key, llm_result_value in llm_results.items():
    results_dict["custom_eval_results"][sae_name][llm_result_key] = llm_result_value

# results_dict["custom_eval_results"][sae_name]["sae_test_accuracy"] = average_test_accuracy(
#     sae_test_accuracies
# )

config.k_values = [1, 2, 5]

for k in config.k_values:
    sae_top_k_probes, sae_top_k_test_accuracies = train_probe_on_activations(
        all_sae_train_acts_BF,
        all_sae_test_acts_BF,
        select_top_k=k,
    )
    results_dict["custom_eval_results"][sae_name][f"sae_top_{k}_test_accuracy"] = (
        average_test_accuracy(sae_top_k_test_accuracies)
    )

results_dict["custom_eval_config"] = asdict(config)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")


# %%
print(sae_top_k_probes["0"])
# %%
