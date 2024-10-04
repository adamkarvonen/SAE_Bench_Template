import os
import random
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn, optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

# Autocheck if the instance is a notebook or not (fixes weird bugs in colab)
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from evals.absorption.prompting import Formatter, SpellingPrompt, create_icl_prompt
from evals.absorption.util import DEFAULT_DEVICE, batchify
from evals.absorption.vocab import LETTERS


class LinearProbe(nn.Module):
    """
    Based on by https://github.com/jbloomAus/alphabetical_probe/blob/main/src/probes.py
    """

    def __init__(self, input_dim, num_outputs: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_outputs)

    def forward(self, x):
        return self.fc(x)

    @property
    def weights(self):
        return self.fc.weight

    @property
    def biases(self):
        return self.fc.bias


def _calc_pos_weights(y: torch.Tensor) -> torch.Tensor:
    num_pos_samples = y.sum(dim=0)
    num_neg_samples = len(y) - num_pos_samples
    return num_neg_samples / num_pos_samples


def train_multi_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples, num_probes), with values in [0, 1]
    num_probes: int | None = None,  # inferred from y_train if None
    batch_size: int = 4096,
    num_epochs: int = 100,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    show_progress: bool = True,
    optimizer: Literal["Adam", "SGD", "AdamW"] = "Adam",
    extra_loss_fn: (
        Callable[[LinearProbe, torch.Tensor, torch.Tensor], torch.Tensor] | None
    ) = None,
    verbose: bool = False,
    device: torch.device = DEFAULT_DEVICE,
) -> LinearProbe:
    """
    Train a multi-class one-vs-rest logistic regression probe on the given data.
    This is equivalent to training num_probes separate binary logistic regression probes.

    Args:
        x_train: tensor of shape (num_samples, input_dim)
        y_train: one_hot (or multi-hot) tensor of shape (num_samples, num_probes), with values in [0, 1]
        num_probes: number of probes to train simultaneously
        batch_size: batch size for training
        num_epochs: number of epochs to train for
        lr: learning rate
        weight_decay: weight decay
        show_progress: whether to show a progress bar
        device: device to train on
    """
    dtype = x_train.dtype
    num_probes = num_probes or y_train.shape[-1]
    dataset = TensorDataset(x_train.to(device), y_train.to(device, dtype=dtype))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    probe = LinearProbe(x_train.shape[-1], num_outputs=num_probes).to(
        device, dtype=dtype
    )

    _run_probe_training(
        probe,
        loader,
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=_calc_pos_weights(y_train)),
        num_epochs=num_epochs,
        lr=lr,
        end_lr=end_lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        optimizer_name=optimizer,
        extra_loss_fn=extra_loss_fn,
        verbose=verbose,
    )

    return probe


def train_binary_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples,), with values in [0, 1]
    batch_size: int = 256,
    num_epochs: int = 100,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    weight_decay: float = 1e-6,
    show_progress: bool = True,
    optimizer: Literal["Adam", "SGD", "AdamW"] = "Adam",
    extra_loss_fn: (
        Callable[[LinearProbe, torch.Tensor, torch.Tensor], torch.Tensor] | None
    ) = None,
    verbose: bool = False,
    device: torch.device = DEFAULT_DEVICE,
) -> LinearProbe:
    """
    Train a logistic regression probe on the given data. This is a thin wrapped around train_multi_probe.

    Args:
        x_train: tensor of shape (num_samples, input_dim)
        y_train: tensor of shape (num_samples,), with values in [0, 1]
        batch_size: batch size for training
        num_epochs: number of epochs to train for
        lr: learning rate
        weight_decay: weight decay
        show_progress: whether to show a progress bar
        device: device to train on
    """
    return train_multi_probe(
        x_train,
        y_train.unsqueeze(1),
        num_probes=1,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        end_lr=end_lr,
        weight_decay=weight_decay,
        show_progress=show_progress,
        optimizer=optimizer,
        extra_loss_fn=extra_loss_fn,
        verbose=verbose,
        device=device,
    )


def _get_exponential_decay_scheduler(
    optimizer: optim.Optimizer,  # type: ignore
    start_lr: float,
    end_lr: float,
    num_steps: int,
) -> optim.lr_scheduler.ExponentialLR:
    gamma = exp(log(end_lr / start_lr) / num_steps)
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def _run_probe_training(
    probe: LinearProbe,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_epochs: int,
    lr: float,
    end_lr: float,
    weight_decay: float,
    show_progress: bool,
    optimizer_name: Literal["Adam", "SGD", "AdamW"],
    extra_loss_fn: (
        Callable[[LinearProbe, torch.Tensor, torch.Tensor], torch.Tensor] | None
    ),
    verbose: bool,
) -> None:
    probe.train()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(probe.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)  # type: ignore
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler = _get_exponential_decay_scheduler(
        optimizer, start_lr=lr, end_lr=end_lr, num_steps=num_epochs
    )

    epoch_pbar = tqdm(range(num_epochs), disable=not show_progress, desc="Epochs")
    for epoch in epoch_pbar:
        epoch_sum_loss = 0
        batch_pbar = tqdm(
            loader,
            disable=not show_progress,
            leave=False,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for batch_embeddings, batch_labels in batch_pbar:
            optimizer.zero_grad()
            logits = probe(batch_embeddings)
            loss = loss_fn(logits, batch_labels)
            if extra_loss_fn is not None:
                loss += extra_loss_fn(probe, batch_embeddings, batch_labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_sum_loss += batch_loss
            batch_pbar.set_postfix({"Loss": f"{batch_loss:.8f}"})

        epoch_mean_loss = epoch_sum_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        epoch_pbar.set_postfix(
            {"Mean Loss": f"{epoch_mean_loss:.8f}", "LR": f"{current_lr:.2e}"}
        )

        if verbose:
            print(
                f"Epoch {epoch + 1}: Mean Loss: {epoch_mean_loss:.8f}, LR: {current_lr:.2e}"
            )

        scheduler.step()

    probe.eval()


def create_dataset_probe_training(
    vocab: list[str],
    formatter: Formatter,
    num_prompts_per_token: int,
    base_template: str,
    max_icl_examples: int = 10,
    train_test_fraction: float = 0.8,
    answer_class_fn: Callable[[str], int] = lambda answer: LETTERS.index(
        answer.strip().lower()
    ),
) -> tuple[list[tuple[SpellingPrompt, int]], list[tuple[SpellingPrompt, int]]]:
    """
    Create train and test datasets for probe training by generating prompts for each token in the given vocabulary.

    Args:
        vocab: List of tokens in the vocabulary.
        formatter: Formatter function for answers.
        num_prompts_per_token: Number of prompts to generate for each token.
        base_template: Template string for the base prompt.
        max_icl_examples: Maximum number of in-context learning examples to include.
        train_test_fraction: Fraction of vocabulary to use for training (default: 0.8).
        answer_class_fn: Function to determine the answer class from the answer string.
                         Default is to index into LETTERS for single-character answers.

    Returns:
        A tuple containing two lists of (SpellingPrompt, int) tuples for train and test sets respectively.
    """
    shuffled_vocab = random.sample(vocab, len(vocab))

    # Split into train and test vocabularies
    split_index = int(len(shuffled_vocab) * train_test_fraction)
    train_vocab = shuffled_vocab[:split_index]
    test_vocab = shuffled_vocab[split_index:]

    train_prompts, test_prompts = [], []

    def generate_prompts(token_list, examples, prompts_list):
        for token in tqdm(
            token_list,
            desc=f"Processing {'train' if prompts_list is train_prompts else 'test'} tokens",
        ):
            for _ in range(num_prompts_per_token):
                prompt = create_icl_prompt(
                    word=token,
                    examples=examples,
                    answer_formatter=formatter,
                    base_template=base_template,
                    max_icl_examples=max_icl_examples,
                )
                answer_class = answer_class_fn(prompt.answer)
                prompts_list.append((prompt, answer_class))

    generate_prompts(train_vocab, train_vocab, train_prompts)
    generate_prompts(test_vocab, test_vocab, test_prompts)

    return train_prompts, test_prompts


def gen_and_save_df_acts_probing(
    model: HookedTransformer,
    train_dataset: list[tuple[SpellingPrompt, int]],
    test_dataset: list[tuple[SpellingPrompt, int]],
    path: str | Path,
    hook_point: str,
    layer: int,
    batch_size: int = 64,
    position_idx: int = -2,
) -> tuple[pd.DataFrame, pd.DataFrame, np.memmap, np.memmap]:
    """
    Generate and save activations for probing tasks to the specified path

    Args:
        model: The model to use for generating activations.
        train_dataset: List of tuples containing SpellingPrompt objects and answer classes for training.
        test_dataset: List of tuples containing SpellingPrompt objects and answer classes for testing.
        path: Base path for saving outputs.
        hook_point: The model hook point to extract activations from.
        task_name: Name of the task for file naming.
        batch_size: Batch size for processing.
        position_idx: Index of the token position to extract activations from. Default is -2.

    Returns:
        A tuple containing the train and test task DataFrames and memory-mapped activation tensors.
    """
    d_model = model.cfg.d_model

    def process_dataset(dataset, prefix):
        df = pd.DataFrame(
            {
                "prompt": [prompt.base for prompt, _ in dataset],
                "HOOK_POINT": [hook_point] * len(dataset),
                "answer": [prompt.answer for prompt, _ in dataset],
                "answer_class": [answer_class for _, answer_class in dataset],
                "token": [prompt.word for prompt, _ in dataset],
            }
        )
        df.index.name = "index"

        memmap_path = os.path.join(task_dir, f"{prefix}_act_tensor.dat")
        act_tensor_memmap = np.memmap(
            memmap_path, dtype="float32", mode="w+", shape=(len(dataset), d_model)
        )

        with torch.no_grad():
            for i, batch in enumerate(
                batchify(dataset, batch_size, show_progress=True)
            ):
                batch_prompts = [prompt.base for prompt, _ in batch]
                cache = model.run_with_cache(batch_prompts, names_filter=[hook_point])[
                    1
                ]
                acts = (
                    cache[hook_point][:, position_idx, :]
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )
                start_idx = i * batch_size
                end_idx = start_idx + len(batch)
                act_tensor_memmap[start_idx:end_idx] = acts

        act_tensor_memmap.flush()
        df_path = os.path.join(task_dir, f"{prefix}_df.csv")
        df.to_csv(df_path, index=True)

        return df, act_tensor_memmap

    layer_path = f"layer_{layer}"
    task_dir = os.path.join(path, layer_path)
    os.makedirs(task_dir, exist_ok=True)

    train_df, train_act_tensor = process_dataset(train_dataset, "train")
    test_df, test_act_tensor = process_dataset(test_dataset, "test")

    return train_df, test_df, train_act_tensor, test_act_tensor


def train_linear_probe_for_task(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    train_activations: np.memmap,
    test_activations: np.memmap,
    num_classes: int = 26,
    batch_size: int = 4096,
    num_epochs: int = 50,
    lr: float = 1e-2,
    weight_decay=1e-4,
) -> tuple[LinearProbe, dict[str, torch.Tensor]]:
    """
    Train a linear probe for a specific task using the provided train and test DataFrames and activation tensors.
    """
    y_train = np.array(train_df["answer_class"].values)
    y_test = np.array(test_df["answer_class"].values)

    X_train_tensor = torch.from_numpy(train_activations).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_test_tensor = torch.from_numpy(test_activations).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)
    y_train_one_hot = one_hot(y_train_tensor, num_classes=num_classes)

    probe = train_multi_probe(
        x_train=X_train_tensor.detach().clone(),
        y_train=y_train_one_hot.detach().clone(),
        num_probes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        show_progress=True,
        verbose=False,
        device=device,
    )

    probe_data = {
        "X_train": X_train_tensor,
        "X_test": X_test_tensor,
        "y_train": y_train_tensor,
        "y_test": y_test_tensor,
    }

    return probe, probe_data


def save_probe_and_data(probe, probe_data, probing_path, layer):
    layer_path = f"layer_{layer}"
    task_dir = os.path.join(probing_path, layer_path)
    os.makedirs(task_dir, exist_ok=True)

    probe_path = os.path.join(task_dir, "probe.pth")
    torch.save(probe, probe_path)

    data_path = os.path.join(task_dir, "data.npz")
    np.savez(
        data_path,
        X_train=probe_data["X_train"].cpu().detach().numpy(),
        X_test=probe_data["X_test"].cpu().detach().numpy(),
        y_train=probe_data["y_train"].cpu().detach().numpy(),
        y_test=probe_data["y_test"].cpu().detach().numpy(),
    )


@dataclass
class ProbeStats:
    letter: str
    f1: float
    accuracy: float
    precision: float
    recall: float


def gen_probe_stats(
    probe: LinearProbe,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device = DEFAULT_DEVICE,
) -> list[ProbeStats]:
    """
    Generate statistics for a trained probe on validation data,
    treating each letter independently.

    Args:
        probe: The trained LinearProbe.
        X_val: Validation input tensor.
        y_val: Validation target tensor.
        device: The device to run computations on (default: CUDA if available, else CPU).

    Returns:
        A list of ProbeStats objects containing performance metrics for each letter.
    """

    def validator_fn(x: torch.Tensor) -> torch.Tensor:
        logits = probe(x.clone().detach().to(device))
        return (logits > threshold).float().cpu()

    results: list[ProbeStats] = []

    val_preds = validator_fn(X_val)
    y_val_cpu = y_val.cpu()

    for i, letter in enumerate(LETTERS):
        letter_preds = val_preds[:, i]
        letter_val_y = y_val_cpu == i
        results.append(
            ProbeStats(
                letter=letter,
                f1=float(f1_score(letter_val_y, letter_preds, average="binary")),
                accuracy=float(accuracy_score(letter_val_y, letter_preds)),
                precision=float(
                    precision_score(letter_val_y, letter_preds, average="binary")
                ),
                recall=float(
                    recall_score(letter_val_y, letter_preds, average="binary")
                ),
            )
        )
    return results
