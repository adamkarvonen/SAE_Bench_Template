from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch import nn
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from evals.absorption.common import (
    RESULTS_DIR,
    PROBES_DIR,
    get_or_make_dir,
    load_df_or_run,
    load_dfs_or_run,
    load_or_train_probe,
    load_probe_data_split_or_train,
)
from evals.absorption.probing import LinearProbe, train_multi_probe
from evals.absorption.util import batchify
from evals.absorption.vocab import LETTERS

EPS = 1e-6
SPARSE_PROBING_EXPERIMENT_NAME = "k_sparse_probing"


class KSparseProbe(nn.Module):
    weight: torch.Tensor  # shape (k)
    bias: torch.Tensor  # scalar
    feature_ids: torch.Tensor  # shape (k)

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, feature_ids: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.feature_ids = feature_ids

    @property
    def k(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered_acts = x[:, self.feature_ids] if len(x.shape) == 2 else x[self.feature_ids]
        return filtered_acts @ self.weight + self.bias


def train_sparse_multi_probe(
    x_train: torch.Tensor,  # tensor of shape (num_samples, input_dim)
    y_train: torch.Tensor,  # tensor of shape (num_samples, num_probes), with values in [0, 1]
    device: torch.device,
    l1_decay: float = 0.01,  # l1 regularization strength
    num_probes: int | None = None,  # inferred from y_train if None
    batch_size: int = 4096,
    num_epochs: int = 50,
    lr: float = 0.01,
    end_lr: float = 1e-5,
    l2_decay: float = 1e-6,
    show_progress: bool = True,
    verbose: bool = False,
) -> LinearProbe:
    """
    Train a multi-probe with L1 regularization on the weights.
    """
    return train_multi_probe(
        x_train,
        y_train,
        num_probes=num_probes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        end_lr=end_lr,
        weight_decay=l2_decay,
        show_progress=show_progress,
        verbose=verbose,
        device=device,
        extra_loss_fn=lambda probe, _x, _y: l1_decay * probe.weights.abs().sum(dim=-1).mean(),
    )


def _get_sae_acts(
    sae: SAE,
    input_activations: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    batch_acts = []
    for batch in batchify(input_activations, batch_size):
        acts = sae.encode(batch.to(sae.device)).cpu()
        batch_acts.append(acts)
    return torch.cat(batch_acts)


def train_k_sparse_probes(
    sae: SAE,
    train_labels: list[tuple[str, int]],  # list of (token, letter number) pairs
    train_activations: torch.Tensor,  # n_vocab X d_model
    ks: Sequence[int],
) -> dict[int, dict[int, KSparseProbe]]:  # dict[k, dict[letter_id, probe]]
    """
    Train k-sparse probes for each k in ks.
    Returns a dict of dicts, where the outer dict is indexed by k and the inner dict is the label.
    """
    results: dict[int, dict[int, KSparseProbe]] = defaultdict(dict)
    with torch.no_grad():
        labels = {label for _, label in train_labels}
        sparse_train_y = torch.nn.functional.one_hot(torch.tensor([idx for _, idx in train_labels]))
        sae_feat_acts = _get_sae_acts(sae, train_activations)
    l1_probe = (
        train_sparse_multi_probe(
            sae_feat_acts.to(sae.device),
            sparse_train_y.to(sae.device),
            l1_decay=0.01,
            num_epochs=50,
            device=sae.device,
        )
        .float()
        .cpu()
    )
    with torch.no_grad():
        train_k_y = np.array([idx for _, idx in train_labels])
        with tqdm(total=len(ks) * len(labels), desc="training k-probes") as pbar:
            for k in ks:
                for label in labels:
                    # using topk and not abs() because we only want features that directly predict the label
                    sparse_feat_ids = l1_probe.weights[label].topk(k).indices.numpy()
                    train_k_x = sae_feat_acts[:, sparse_feat_ids].float().numpy()
                    # Use SKLearn here because it's much faster than torch if the data is small
                    sk_probe = LogisticRegression(max_iter=500, class_weight="balanced").fit(
                        train_k_x, (train_k_y == label).astype(np.int64)
                    )
                    probe = KSparseProbe(
                        weight=torch.tensor(sk_probe.coef_[0]).float(),
                        bias=torch.tensor(sk_probe.intercept_[0]).float(),
                        feature_ids=torch.tensor(sparse_feat_ids),
                    )
                    results[k][label] = probe
                    pbar.update(1)
    return results


@torch.inference_mode()
def sae_k_sparse_metadata(
    sae: SAE,
    probe: LinearProbe,
    k_sparse_probes: dict[int, dict[int, KSparseProbe]],
    sae_name: str,
    layer: int,
) -> pd.DataFrame:
    norm_probe_weights = probe.weights / torch.norm(probe.weights, dim=-1, keepdim=True)
    norm_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0, keepdim=True)
    norm_W_dec = sae.W_dec / torch.norm(sae.W_dec, dim=-1, keepdim=True)
    probe_dec_cos = (
        (norm_probe_weights.to(dtype=norm_W_dec.dtype, device=norm_W_dec.device) @ norm_W_dec.T)
        .cpu()
        .float()
    )
    probe_enc_cos = (
        (norm_probe_weights.to(dtype=norm_W_enc.dtype, device=norm_W_enc.device) @ norm_W_enc)
        .cpu()
        .float()
    )

    metadata: dict[str, float | str | float | np.ndarray] = {
        "layer": layer,
        "sae_name": sae_name,
    }
    rows = []
    for letter_i, letter in enumerate(LETTERS):
        for k, k_probes in k_sparse_probes.items():
            row = {**metadata}
            k_probe = k_probes[letter_i]
            row["letter"] = letter
            row["k"] = k
            row["feats"] = k_probe.feature_ids.numpy()
            row["cos_probe_sae_enc"] = probe_enc_cos[letter_i, k_probe.feature_ids].numpy()
            row["cos_probe_sae_dec"] = probe_dec_cos[letter_i, k_probe.feature_ids].numpy()
            row["weights"] = k_probe.weight.float().numpy()
            row["bias"] = k_probe.bias.item()
            rows.append(row)
    return pd.DataFrame(rows)


@torch.inference_mode()
def eval_probe_and_sae_k_sparse_raw_scores(
    sae: SAE,
    probe: LinearProbe,
    k_sparse_probes: dict[int, dict[int, KSparseProbe]],
    eval_labels: list[tuple[str, int]],  # list of (token, letter number) pairs
    eval_activations: torch.Tensor,  # n_vocab X d_model
) -> pd.DataFrame:
    probe = probe.to("cpu")

    # using a generator to avoid storing all the rows in memory
    def row_generator():
        for token_act, (token, answer_idx) in tqdm(
            zip(eval_activations, eval_labels), total=len(eval_labels)
        ):
            probe_scores = probe(token_act).tolist()
            row: dict[str, float | str | int | np.ndarray] = {
                "token": token,
                "answer_letter": LETTERS[answer_idx],
            }
            sae_acts = (
                _get_sae_acts(sae, token_act.unsqueeze(0).to(sae.device)).float().cpu()
            ).squeeze()
            for letter_i, (letter, probe_score) in enumerate(zip(LETTERS, probe_scores)):
                row[f"score_probe_{letter}"] = probe_score
                for k, k_probes in k_sparse_probes.items():
                    k_probe = k_probes[letter_i]
                    k_probe_score = k_probe(sae_acts)
                    sparse_acts = sae_acts[k_probe.feature_ids]
                    row[f"score_sparse_sae_{letter}_k_{k}"] = k_probe_score.item()
                    row[f"sum_sparse_sae_{letter}_k_{k}"] = sparse_acts.sum().item()
                    row[f"sparse_sae_{letter}_k_{k}_acts"] = sparse_acts.numpy()
            yield row

    return pd.DataFrame(row_generator())


def load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    sae_name: str,
    max_k_value: int,
    prompt_template: str,
    prompt_token_pos: int,
    probes_dir: Path | str,
    device: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if verbose:
        print("Loading probe and training data", flush=True)
    probe = load_or_train_probe(
        model=model,
        layer=layer,
        probes_dir=probes_dir,
        base_template=prompt_template,
        pos_idx=prompt_token_pos,
        device=device,
    )
    train_activations, train_data = load_probe_data_split_or_train(
        model,
        base_template=prompt_template,
        pos_idx=prompt_token_pos,
        probes_dir=probes_dir,
        layer=layer,
        split="train",
        device="cpu",
    )
    if verbose:
        print("Training k-sparse probes", flush=True)
    k_sparse_probes = train_k_sparse_probes(
        sae,
        train_data,
        train_activations,
        ks=list(range(1, max_k_value + 1)),
    )
    with torch.no_grad():
        if verbose:
            print("Loading validation data", flush=True)
        eval_activations, eval_data = load_probe_data_split_or_train(
            model,
            base_template=prompt_template,
            pos_idx=prompt_token_pos,
            probes_dir=probes_dir,
            layer=layer,
            split="test",
            device="cpu",
        )
        if verbose:
            print("Evaluating raw k-sparse probing scores", flush=True)
        df = eval_probe_and_sae_k_sparse_raw_scores(
            sae,
            probe,
            k_sparse_probes=k_sparse_probes,
            eval_labels=eval_data,
            eval_activations=eval_activations,
        )
        if verbose:
            print("Building metadata", flush=True)
        metadata = sae_k_sparse_metadata(
            sae,
            probe,
            k_sparse_probes,
            sae_name=sae_name,
            layer=layer,
        )
    return df, metadata


def build_metrics_df(results_df, metadata_df, max_k_value: int):
    aucs = []
    for letter in LETTERS:
        y = (results_df["answer_letter"] == letter).values
        pred_probe = results_df[f"score_probe_{letter}"].values
        auc_probe = metrics.roc_auc_score(y, pred_probe)
        f1_probe = metrics.f1_score(y, pred_probe > 0.0)
        recall_probe = metrics.recall_score(y, pred_probe > 0.0)
        precision_probe = metrics.precision_score(y, pred_probe > 0.0)
        auc_info = {
            "auc_probe": auc_probe,
            "f1_probe": f1_probe,
            "recall_probe": recall_probe,
            "precision_probe": precision_probe,
            "letter": letter,
            "layer": metadata_df["layer"].iloc[0],
            "sae_name": metadata_df["sae_name"].iloc[0],
        }

        for k in range(1, max_k_value + 1):
            pred_sae = results_df[f"score_sparse_sae_{letter}_k_{k}"].values
            auc_sae = metrics.roc_auc_score(y, pred_sae)
            f1 = metrics.f1_score(y, pred_sae > 0.0)
            recall = metrics.recall_score(y, pred_sae > 0.0)
            precision = metrics.precision_score(y, pred_sae > 0.0)
            auc_info[f"auc_sparse_sae_{k}"] = auc_sae
            sum_sae_pred = results_df[f"sum_sparse_sae_{letter}_k_{k}"].values
            auc_sum_sae = metrics.roc_auc_score(y, sum_sae_pred)
            f1_sum_sae = metrics.f1_score(y, sum_sae_pred > EPS)
            recall_sum_sae = metrics.recall_score(y, sum_sae_pred > EPS)
            precision_sum_sae = metrics.precision_score(y, sum_sae_pred > EPS)

            auc_info[f"f1_sparse_sae_{k}"] = f1
            auc_info[f"recall_sparse_sae_{k}"] = recall
            auc_info[f"precision_sparse_sae_{k}"] = precision
            auc_info[f"auc_sum_sparse_sae_{k}"] = auc_sum_sae
            auc_info[f"f1_sum_sparse_sae_{k}"] = f1_sum_sae
            auc_info[f"recall_sum_sparse_sae_{k}"] = recall_sum_sae
            auc_info[f"precision_sum_sparse_sae_{k}"] = precision_sum_sae

            meta_row = metadata_df[(metadata_df["letter"] == letter) & (metadata_df["k"] == k)]
            auc_info[f"sparse_sae_k_{k}_feats"] = meta_row["feats"].iloc[0]
            auc_info[f"cos_probe_sae_enc_k_{k}"] = meta_row["cos_probe_sae_enc"].iloc[0]
            auc_info[f"cos_probe_sae_dec_k_{k}"] = meta_row["cos_probe_sae_dec"].iloc[0]
            auc_info[f"sparse_sae_k_{k}_weights"] = meta_row["weights"].iloc[0]
            auc_info[f"sparse_sae_k_{k}_bias"] = meta_row["bias"].iloc[0]
            auc_info["layer"] = meta_row["layer"].iloc[0]
            auc_info["sae_name"] = meta_row["sae_name"].iloc[0]
        aucs.append(auc_info)
    return pd.DataFrame(aucs)


def add_feature_splits_to_metrics_df(
    df: pd.DataFrame,
    max_k_value: int,
    f1_jump_threshold: float = 0.03,
) -> None:
    """
    If a k-sparse probe has a F1 score that increases by `f1_jump_threshold` or more from the previous k-1, consider this to be feature splitting.
    """
    split_feats_by_letter = {}
    for letter in LETTERS:
        prev_best = -100
        df_letter = df[df["letter"] == letter]
        for k in range(1, max_k_value + 1):
            k_score = df_letter[f"f1_sparse_sae_{k}"].iloc[0]  # type: ignore
            k_feats = df_letter[f"sparse_sae_k_{k}_feats"].iloc[0].tolist()  # type: ignore
            if k_score > prev_best + f1_jump_threshold:
                prev_best = k_score
                split_feats_by_letter[letter] = k_feats
            else:
                break
    df["split_feats"] = df["letter"].apply(lambda letter: split_feats_by_letter.get(letter, []))
    df["num_split_features"] = df["split_feats"].apply(len) - 1


def get_sparse_probing_raw_results_filename(sae_name: str, layer: int) -> str:
    return f"layer_{layer}_{sae_name}_raw_results.parquet"


def get_sparse_probing_metadata_filename(sae_name: str, layer: int) -> str:
    return f"layer_{layer}_{sae_name}_metadata.parquet"


def get_sparse_probing_metrics_filename(sae_name: str, layer: int) -> str:
    return f"layer_{layer}_{sae_name}_metrics.parquet"


def run_k_sparse_probing_experiment(
    model: HookedTransformer,
    sae: SAE,
    layer: int,
    sae_name: str,
    max_k_value: int,
    prompt_template: str,
    prompt_token_pos: int,
    device: str,
    experiment_dir: Path | str = RESULTS_DIR / SPARSE_PROBING_EXPERIMENT_NAME,
    probes_dir: Path | str = PROBES_DIR,
    force: bool = False,
    f1_jump_threshold: float = 0.03,  # noqa: ARG001
    verbose: bool = True,
) -> pd.DataFrame:
    task_output_dir = get_or_make_dir(experiment_dir) / sae_name
    raw_results_path = task_output_dir / get_sparse_probing_raw_results_filename(sae_name, layer)
    metadata_results_path = task_output_dir / get_sparse_probing_metadata_filename(sae_name, layer)
    metrics_results_path = task_output_dir / get_sparse_probing_metrics_filename(sae_name, layer)

    def get_raw_results_df():
        return load_dfs_or_run(
            lambda: load_and_run_eval_probe_and_sae_k_sparse_raw_scores(
                sae,
                model,
                probes_dir=probes_dir,
                verbose=verbose,
                sae_name=sae_name,
                layer=layer,
                max_k_value=max_k_value,
                prompt_template=prompt_template,
                prompt_token_pos=prompt_token_pos,
                device=device,
            ),
            (raw_results_path, metadata_results_path),
            force=force,
        )

    metrics_df = load_df_or_run(
        lambda: build_metrics_df(*get_raw_results_df(), max_k_value=max_k_value),
        metrics_results_path,
        force=force,
    )
    add_feature_splits_to_metrics_df(
        metrics_df, max_k_value=max_k_value, f1_jump_threshold=f1_jump_threshold
    )
    return metrics_df
