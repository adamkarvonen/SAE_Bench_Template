"""
Shared helpers for experiments
"""

import re
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerFast

from evals.absorption.probing import (
    LinearProbe,
    create_dataset_probe_training,
    gen_and_save_df_acts_probing,
    save_probe_and_data,
    train_linear_probe_for_task,
)
from evals.absorption.prompting import (
    Formatter,
    first_letter_formatter,
)
from evals.absorption.vocab import get_alpha_tokens

DEFAULT_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / "absorption"
PROBES_DIR = RESULTS_DIR / "probes"


def dtype_to_str(dtype: torch.dtype | str) -> str:
    return str(dtype).replace("torch.", "")


def load_or_train_probe(
    model: HookedTransformer,
    base_template: str,
    pos_idx: int,
    layer: int = 0,
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> LinearProbe:
    probe_path = Path(probes_dir) / f"{model.cfg.model_name}" / f"layer_{layer}" / "probe.pth"
    if not probe_path.exists():
        print(f"Probe for layer {layer} not found, training...")
        train_and_save_probes(
            model,
            layers=[layer],
            probes_dir=probes_dir,
            base_template=base_template,
            pos_idx=pos_idx,
            device=device,
        )
    return load_probe(model.cfg.model_name, layer, probes_dir, dtype, device)


def load_probe(
    model_name: str | None,
    layer: int = 0,
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> LinearProbe:
    probe = torch.load(
        Path(probes_dir) / f"{model_name}" / f"layer_{layer}" / "probe.pth",
        map_location=device,
    ).to(dtype=dtype)
    return probe


def load_probe_data_split_or_train(
    model: HookedTransformer,
    base_template: str,
    pos_idx: int,
    layer: int = 0,
    split: Literal["train", "test"] = "test",
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    probe_path = Path(probes_dir) / f"{model.cfg.model_name}" / f"layer_{layer}" / "probe.pth"
    if not probe_path.exists():
        print(f"Probe for layer {layer} not found, training...")
        train_and_save_probes(
            model,
            layers=[layer],
            probes_dir=probes_dir,
            base_template=base_template,
            pos_idx=pos_idx,
            device=device,
        )
    return load_probe_data_split(
        model,  # type: ignore
        layer,
        split,
        probes_dir,
        dtype,
        device,
    )


@torch.inference_mode()
def load_probe_data_split(
    model: HookedTransformer,
    layer: int = 0,
    split: Literal["train", "test"] = "test",
    probes_dir: str | Path = PROBES_DIR,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    np_data = np.load(
        Path(probes_dir) / f"{model.cfg.model_name}" / f"layer_{layer}" / "data.npz",
    )
    df = pd.read_csv(
        Path(probes_dir) / f"{model.cfg.model_name}" / f"layer_{layer}" / f"{split}_df.csv",
        keep_default_na=False,
        na_values=[""],
    )
    activations = torch.from_numpy(np_data[f"X_{split}"]).to(device, dtype=dtype)
    labels = np_data[f"y_{split}"].tolist()
    return _parse_probe_data_split(
        model.tokenizer,
        activations,
        split_labels=labels,
        df=df,  # type: ignore
    )


def _parse_probe_data_split(
    tokenizer: PreTrainedTokenizerFast,
    split_activations: torch.Tensor,
    split_labels: list[int],
    df: pd.DataFrame,
) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    valid_act_indices = []
    vocab_with_labels = []
    raw_tokens_with_labels = [
        (df.iloc[idx]["token"], label) for idx, label in enumerate(split_labels)
    ]
    for idx, (token, label) in enumerate(raw_tokens_with_labels):
        # sometimes we have tokens that look like <0x6A>
        if not isinstance(token, str) or re.match(r"[\d<>]", token):
            continue
        vocab_with_labels.append((tokenizer.convert_tokens_to_string([token]), label))
        valid_act_indices.append(idx)
    activations = split_activations[valid_act_indices]
    if activations.shape[0] != len(vocab_with_labels):
        raise ValueError(
            f"Activations and vocab with labels have different lengths: "
            f"{activations.shape[0]} != {len(vocab_with_labels)}"
        )
    return activations.clone(), vocab_with_labels


def get_or_make_dir(
    experiment_dir: str | Path,
) -> Path:
    """
    Helper to create a directory for a specific task within an experiment directory.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def load_experiment_df(
    experiment_name: str,
    path: Path,
) -> pd.DataFrame:
    """
    Helper to load a DF or error if it doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the {experiment_name} experiment first."
        )
    return pd.read_parquet(path)


def load_df_or_run(
    fn: Callable[[], pd.DataFrame],
    path: Path,
    force: bool = False,
) -> pd.DataFrame:
    return load_dfs_or_run(lambda: [fn()], [path], force)[0]


def load_dfs_or_run(
    fn: Callable[[], Iterable[pd.DataFrame]],
    paths: Iterable[Path],
    force: bool = False,
) -> list[pd.DataFrame]:
    if force or not all(path.exists() for path in paths):
        dfs = fn()
        for df, path in zip(dfs, paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
    else:
        print(f"{paths} exist(s), loading from disk")
        dfs = [pd.read_parquet(path) for path in paths]
    return list(dfs)


def create_and_train_probe(
    model: HookedTransformer,
    formatter: Formatter,
    hook_point: str,
    probes_dir: str | Path,
    vocab: list[str],
    batch_size: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    base_template: str,
    pos_idx: int,
    num_prompts_per_token: int = 1,
):
    train_dataset, test_dataset = create_dataset_probe_training(
        vocab=vocab,
        formatter=formatter,
        num_prompts_per_token=num_prompts_per_token,
        base_template=base_template,
    )

    layer = int(hook_point.split(".")[1])

    train_df, test_df, train_activations, test_activations = gen_and_save_df_acts_probing(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        path=Path(probes_dir) / f"{model.cfg.model_name}" / f"layer_{layer}",
        hook_point=hook_point,
        batch_size=batch_size,
        position_idx=pos_idx,
    )

    num_classes = 26
    probe, probe_data = train_linear_probe_for_task(
        train_df=train_df,
        test_df=test_df,
        train_activations=train_activations,
        test_activations=test_activations,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
    )
    save_probe_and_data(
        probe,
        probe_data,
        probing_path=Path(probes_dir) / f"{model.cfg.model_name}" / f"layer_{layer}",
    )


def train_and_save_probes(
    model: HookedTransformer,
    layers: list[int],
    base_template: str,
    pos_idx: int,
    probes_dir: str | Path = PROBES_DIR,
    batch_size=64,
    num_epochs=50,
    lr=1e-2,
    device=DEFAULT_DEVICE,
):
    vocab = get_alpha_tokens(model.tokenizer)  # type: ignore
    for layer in tqdm(layers):
        hook_point = f"blocks.{layer}.hook_resid_post"
        create_and_train_probe(
            model=model,
            hook_point=hook_point,
            formatter=first_letter_formatter(),
            probes_dir=probes_dir,
            vocab=vocab,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            device=torch.device(device),
            base_template=base_template,
            pos_idx=pos_idx,
        )
