import copy
from typing import Optional

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import math

import sae_bench_utils.dataset_info as dataset_info


class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@jaxtyped(typechecker=beartype)
def prepare_probe_data(
    all_activations: dict[str, Float[torch.Tensor, "num_datapoints_per_class ... d_model"]],
    class_name: str,
    perform_scr: bool = False,
) -> tuple[
    Float[torch.Tensor, "num_datapoints_per_class_x_2 ... d_model"],
    Int[torch.Tensor, "num_datapoints_per_class_x_2"],
]:
    """perform_scr is for the SCR metric. In this case, all_activations has 3 pairs of keys, or 6 total.
    It's a bit unfortunate to introduce coupling between the metrics, but most of the code is reused between them.
    The ... means we can have an optional seq_len dimension between num_datapoints_per_class and d_model.
    """
    positive_acts_BD = all_activations[class_name]
    device = positive_acts_BD.device

    num_positive = len(positive_acts_BD)

    if perform_scr:
        if class_name in dataset_info.PAIRED_CLASS_KEYS.keys():
            selected_negative_acts_BD = all_activations[dataset_info.PAIRED_CLASS_KEYS[class_name]]
        elif class_name in dataset_info.PAIRED_CLASS_KEYS.values():
            reversed_dict = {v: k for k, v in dataset_info.PAIRED_CLASS_KEYS.items()}
            selected_negative_acts_BD = all_activations[reversed_dict[class_name]]
        else:
            raise ValueError(f"Class {class_name} not found in paired class keys.")
    else:
        # Collect all negative class activations and labels
        selected_negative_acts_BD = []
        negative_keys = [k for k in all_activations.keys() if k != class_name]
        num_neg_classes = len(negative_keys)
        samples_per_class = math.ceil(num_positive / num_neg_classes)

        for negative_class_name in negative_keys:
            sample_indices = torch.randperm(len(all_activations[negative_class_name]))[
                :samples_per_class
            ]
            selected_negative_acts_BD.append(all_activations[negative_class_name][sample_indices])

        selected_negative_acts_BD = torch.cat(selected_negative_acts_BD)

    # Randomly select num_positive samples from negative class
    indices = torch.randperm(len(selected_negative_acts_BD))[:num_positive]
    selected_negative_acts_BD = selected_negative_acts_BD[indices]

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
def apply_topk_mask_zero_dims(
    acts_BD: Float[torch.Tensor, "batch_size d_model"],
    mask_D: Bool[torch.Tensor, "d_model"],
) -> Float[torch.Tensor, "batch_size k"]:
    masked_acts_BD = acts_BD.clone()
    masked_acts_BD[:, mask_D] = 0.0

    return masked_acts_BD


@jaxtyped(typechecker=beartype)
def apply_topk_mask_reduce_dim(
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
    max_iter: int = 1000,  # non-default sklearn value, increased due to convergence warnings
    C: float = 1.0,  # default sklearn value
    verbose: bool = False,
    l1_ratio: Optional[float] = None,
) -> tuple[LogisticRegression, float]:
    train_inputs = train_inputs.to(dtype=torch.float32)
    test_inputs = test_inputs.to(dtype=torch.float32)

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
    inputs = inputs.to(dtype=torch.float32)
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
    early_stopping_patience: int = 10,
) -> tuple[Probe, float]:
    """We have a GPU training function for training on all SAE features, which was very slow (1 minute+) on CPU.
    This is also used for SCR / TPP, which require probe weights."""
    device = train_inputs.device
    model_dtype = train_inputs.dtype

    print(f"Training probe with dim: {dim}, device: {device}, dtype: {model_dtype}")

    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_test_accuracy = 0.0
    best_probe = None
    patience_counter = 0
    for epoch in range(epochs):
        indices = torch.randperm(len(train_inputs))

        for i in range(0, len(train_inputs), batch_size):
            batch_indices = indices[i : i + batch_size]
            acts_BD = train_inputs[batch_indices]
            labels_B = train_labels[batch_indices]
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
            print(f"GPU probe training early stopping triggered after {epoch + 1} epochs")
            break

    return best_probe, best_test_accuracy


@jaxtyped(typechecker=beartype)
def train_probe_on_activations(
    train_activations: dict[str, Float[torch.Tensor, "train_dataset_size d_model"]],
    test_activations: dict[str, Float[torch.Tensor, "test_dataset_size d_model"]],
    select_top_k: Optional[int] = None,
    use_sklearn: bool = True,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    verbose: bool = False,
    early_stopping_patience: int = 10,
    perform_scr: bool = False,
    l1_penalty: Optional[float] = None,
) -> tuple[dict[str, LogisticRegression | Probe], dict[str, float]]:
    """Train a probe on the given activations and return the probe and test accuracies for each profession.
    use_sklearn is a flag to use sklearn's LogisticRegression model instead of a custom PyTorch model.
    We use sklearn by default. probe training on GPU is only for training a probe on all SAE features.
    """
    torch.set_grad_enabled(True)

    probes, test_accuracies = {}, {}

    for profession in train_activations.keys():
        train_acts, train_labels = prepare_probe_data(train_activations, profession, perform_scr)
        test_acts, test_labels = prepare_probe_data(test_activations, profession, perform_scr)

        if select_top_k is not None:
            activation_mask_D = get_top_k_mean_diff_mask(train_acts, train_labels, select_top_k)
            train_acts = apply_topk_mask_reduce_dim(train_acts, activation_mask_D)
            test_acts = apply_topk_mask_reduce_dim(test_acts, activation_mask_D)

        activation_dim = train_acts.shape[1]

        print(f"Num non-zero elements: {activation_dim}")

        if use_sklearn:
            probe, test_accuracy = train_sklearn_probe(
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                verbose=False,
            )
        else:
            probe, test_accuracy = train_probe_gpu(
                train_acts,
                train_labels,
                test_acts,
                test_labels,
                dim=activation_dim,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                verbose=verbose,
                early_stopping_patience=early_stopping_patience,
                l1_penalty=l1_penalty,
            )

        print(f"Test accuracy for {profession}: {test_accuracy}")

        probes[profession] = probe
        test_accuracies[profession] = test_accuracy

    return probes, test_accuracies
