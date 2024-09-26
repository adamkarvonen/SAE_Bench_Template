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
    batch_size: int,
    select_top_k: Optional[int] = None,  # experimental feature
) -> tuple[list[Float[torch.Tensor, "batch_size d_model"]], list[Int[torch.Tensor, "batch_size"]]]:
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

    # Experimental feature: find the top k features that differ the most between in distribution and out of distribution
    # zero out the rest. Useful for k-sparse probing experiments.
    if select_top_k is not None:
        positive_distribution_D = positive_acts_BD.mean(dim=(0))
        negative_distribution_D = negative_acts.mean(dim=(0))
        distribution_diff_D = (positive_distribution_D - negative_distribution_D).abs()
        top_k_indices_D = torch.argsort(distribution_diff_D, descending=True)[:select_top_k]

        mask_D = torch.ones(
            distribution_diff_D.shape[0], dtype=torch.bool, device=positive_acts_BD.device
        )
        mask_D[top_k_indices_D] = False

        masked_positive_acts_BD = positive_acts_BD.clone()
        masked_negative_acts_BD = selected_negative_acts_BD.clone()

        masked_positive_acts_BD[:, mask_D] = 0.0
        masked_negative_acts_BD[:, mask_D] = 0.0
    else:
        masked_positive_acts_BD = positive_acts_BD
        masked_negative_acts_BD = selected_negative_acts_BD

    # Combine positive and negative samples
    combined_acts = torch.cat([masked_positive_acts_BD, masked_negative_acts_BD])

    combined_labels = torch.empty(len(combined_acts), dtype=torch.int, device=device)
    combined_labels[:num_positive] = dataset_info.POSITIVE_CLASS_LABEL
    combined_labels[num_positive:] = dataset_info.NEGATIVE_CLASS_LABEL

    # Shuffle the combined data
    shuffle_indices = torch.randperm(len(combined_acts))
    shuffled_acts = combined_acts[shuffle_indices]
    shuffled_labels = combined_labels[shuffle_indices]

    # Reshape into lists of tensors with specified batch_size
    num_samples = len(shuffled_acts)
    num_batches = num_samples // batch_size

    batched_acts = [
        shuffled_acts[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]
    batched_labels = [
        shuffled_labels[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]

    return batched_acts, batched_labels


def test_probe(
    input_batches: list[Float[torch.Tensor, "batch_size d_model"]],
    label_batches: list[Int[torch.Tensor, "batch_size"]],
    probe: Probe,
) -> float:
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for acts_BD, labels_B in zip(input_batches, label_batches):
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
        loss = torch.stack(losses).mean().item()

    return accuracy_all


def train_probe(
    train_input_batches: list[Float[torch.Tensor, "batch_size d_model"]],
    train_label_batches: list[Int[torch.Tensor, "batch_size"]],
    test_input_batches: list[Float[torch.Tensor, "batch_size d_model"]],
    test_label_batches: list[Int[torch.Tensor, "batch_size"]],
    dim: int,
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
        for acts_BD, labels_B in zip(train_input_batches, train_label_batches):
            logits_B = probe(acts_BD)
            loss = criterion(
                logits_B, labels_B.clone().detach().to(device=device, dtype=model_dtype)
            )
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        train_accuracy = test_probe(train_input_batches, train_label_batches, probe)

        test_accuracy = test_probe(test_input_batches, test_label_batches, probe)

        if epoch == epochs - 1 and verbose:
            print(
                f"\nEpoch {epoch + 1}/{epochs} Loss: {loss.item()}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}\n"
            )

    return probe, test_accuracy


def train_probe_on_activations(
    train_activations: dict[str, Float[torch.Tensor, "num_datapoints d_model"]],
    test_activations: dict[str, Float[torch.Tensor, "num_datapoints d_model"]],
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
        train_acts, train_labels = prepare_probe_data(
            train_activations, profession, probe_batch_size, select_top_k
        )

        test_acts, test_labels = prepare_probe_data(
            test_activations, profession, probe_batch_size, select_top_k
        )

        activation_dim = train_acts[0].shape[1]

        print(f"activation dim: {activation_dim}")

        probe, test_accuracy = train_probe(
            train_acts,
            train_labels,
            test_acts,
            test_labels,
            epochs=epochs,
            dim=activation_dim,
            device=device,
            model_dtype=model_dtype,
            lr=lr,
            verbose=False,
        )

        print(f"Test accuracy for {profession}: {test_accuracy}")

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

    return probes, test_accuracies
