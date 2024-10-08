from dataclasses import asdict
import gc
from sae_lens import SAE
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.sae import TopK

from evals.absorption import eval_config
from evals.absorption.feature_absorption import run_feature_absortion_experiment
from evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
from sae_bench_utils import formatting_utils, activation_collection
from transformer_lens import HookedTransformer


def run_eval(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
    force_rerun: bool = False,
):
    """config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility.
    selected_saes_dict: dict[str, list[str]] is a dict of SAE release name: list of SAE names to evaluate.
    Example: sae_bench_pythia70m_sweep_topk_ctx128_0730 :
    ['pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_10',
    'pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_12']"""
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

    print(f"Running evaluation for layer {config.layer}")

    for sae_release in selected_saes_dict:
        print(
            f"Running evaluation for SAE release: {sae_release}, SAEs: {selected_saes_dict[sae_release]}"
        )
        sae_id_to_name_map = sae_map_df.saes_map[sae_release]
        sae_name_to_id_map = {v: k for k, v in sae_id_to_name_map.items()}

        for sae_name in tqdm(
            selected_saes_dict[sae_release],
            desc="Running SAE evaluation on all selected SAEs",
        ):
            gc.collect()
            torch.cuda.empty_cache()

            sae_id = sae_name_to_id_map[sae_name]

            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
            sae = sae.to(device=device)
            sae = _fix_topk(sae, sae_name, sae_release)

            run_k_sparse_probing_experiment(
                model=model,
                sae=sae,
                layer=config.layer,
                sae_name=sae_name,
                force=force_rerun,
                max_k_value=config.max_k_value,
                f1_jump_threshold=config.f1_jump_threshold,
                prompt_template=config.prompt_template,
                prompt_token_pos=config.prompt_token_pos,
                device=device,
            )

            raw_df = run_feature_absortion_experiment(
                model=model,
                sae=sae,
                layer=config.layer,
                sae_name=sae_name,
                force=force_rerun,
                max_k_value=config.max_k_value,
                feature_split_f1_jump_threshold=config.f1_jump_threshold,
                prompt_template=config.prompt_template,
                prompt_token_pos=config.prompt_token_pos,
                batch_size=llm_batch_size,
                device=device,
            )
            agg_df = _aggregate_results_df(raw_df)

            results_dict["custom_eval_results"][sae_name] = {}
            for _, row in agg_df.iterrows():
                letter = row["letter"]
                results_dict["custom_eval_results"][sae_name][letter] = {
                    "num_absorption": int(row["num_absorption"]),
                    "absorption_rate": float(row["absorption_rate"]),
                    "num_probe_true_positives": float(row["num_probe_true_positives"]),
                    "num_feature_splits": int(row["num_split_feats"]),
                }

    results_dict["custom_eval_config"] = asdict(config)
    return results_dict


def _aggregate_results_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    agg_df = (
        df[["letter", "is_absorption"]]
        .groupby(["letter"])
        .sum()
        .reset_index()
        .merge(
            df[["letter", "num_probe_true_positives", "split_feats"]]
            .groupby(["letter"])
            .agg(
                {
                    "num_probe_true_positives": "mean",
                    "split_feats": lambda x: x.iloc[
                        0
                    ],  # Take the first split_feats list for each letter
                }
            )
            .reset_index()
        )
    )
    agg_df["num_split_feats"] = agg_df["split_feats"].apply(len)
    agg_df["num_absorption"] = agg_df["is_absorption"]
    agg_df["absorption_rate"] = (
        agg_df["num_absorption"] / agg_df["num_probe_true_positives"]
    )
    return agg_df


def _fix_topk(
    sae: SAE,
    sae_name: str,
    sae_release: str,
):
    if "topk" in sae_name:
        if isinstance(sae.activation_fn, TopK):
            return sae

        sae = formatting_utils.fix_topk_saes(sae, sae_release, sae_name, data_dir="../")

        assert isinstance(sae.activation_fn, TopK)
    return sae
