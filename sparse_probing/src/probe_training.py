import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Callable, Optional
from jaxtyping import Int, Float, jaxtyped, BFloat16
from beartype import beartype
import einops
from transformer_lens import HookedTransformer
from sae_lens import SAE

import dataset_info


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
def select_top_k_mean_diff(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    labels_B: Int[torch.Tensor, "batch_size"],
    k: int,
) -> Float[torch.Tensor, "batch_size d_model"]:
    positive_mask_B = labels_B == dataset_info.POSITIVE_CLASS_LABEL
    negative_mask_B = labels_B == dataset_info.NEGATIVE_CLASS_LABEL

    positive_distribution_D = acts_BD[positive_mask_B].mean(dim=0)
    negative_distribution_D = acts_BD[negative_mask_B].mean(dim=0)
    distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
    top_k_indices_D = torch.argsort(distribution_diff_D, descending=True)[:k]

    mask_D = torch.ones(acts_BD.shape[1], dtype=torch.bool, device=acts_BD.device)
    mask_D[top_k_indices_D] = False

    masked_acts_BD = acts_BD.clone()
    masked_acts_BD[:, mask_D] = 0.0

    return masked_acts_BD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def test_probe(
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
def train_probe(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    dim: int,
    batch_size: int,
    epochs: int,
    device: str,
    model_dtype: torch.dtype,
    lr: float,
    verbose: bool = False,
) -> tuple[Probe, float]:
    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for i in range(0, len(train_inputs), batch_size):
            acts_BD = train_inputs[i : i + batch_size]
            labels_B = train_labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            loss = criterion(
                logits_B, labels_B.clone().detach().to(device=device, dtype=model_dtype)
            )
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        train_accuracy = test_probe(train_inputs, train_labels, batch_size, probe)

        test_accuracy = test_probe(test_inputs, test_labels, batch_size, probe)

        if epoch == epochs - 1 and verbose:
            print(
                f"\nEpoch {epoch + 1}/{epochs} Loss: {loss.item()}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}\n"
            )

    return probe, test_accuracy


@jaxtyped(typechecker=beartype)
def train_probe_on_activations(
    train_activations: dict[str, Float[torch.Tensor, "train_dataset_size d_model"]],
    test_activations: dict[str, Float[torch.Tensor, "test_dataset_size d_model"]],
    probe_batch_size: int,
    epochs: int,
    lr: float,
    model_dtype: torch.dtype,
    device: str,
    select_top_k: Optional[int] = None,
) -> tuple[dict[str, Probe], dict[str, float]]:
    torch.set_grad_enabled(True)

    probes, test_accuracies = {}, {}

    for profession in train_activations.keys():
        train_acts, train_labels = prepare_probe_data(train_activations, profession)

        test_acts, test_labels = prepare_probe_data(test_activations, profession)

        if select_top_k is not None:
            train_acts = select_top_k_mean_diff(train_acts, train_labels, select_top_k)
            test_acts = select_top_k_mean_diff(test_acts, test_labels, select_top_k)

        activation_dim = train_acts.shape[1]

        print(f"activation dim: {activation_dim}")

        probe, test_accuracy = train_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels,
            dim=activation_dim,
            batch_size=probe_batch_size,
            epochs=epochs,
            device=device,
            model_dtype=model_dtype,
            lr=lr,
            verbose=False,
        )

        print(f"Test accuracy for {profession}: {test_accuracy}")

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

    return probes, test_accuracies
