from dataclasses import asdict
import gc
import statistics
from sae_lens import SAE
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.sae import TopK

from evals.absorption import eval_config
from evals.absorption.feature_absorption import run_feature_absortion_experiment
from evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
from sae_bench_utils import formatting_utils, activation_collection, get_eval_uuid, get_sae_lens_version, get_sae_bench_version
from transformer_lens import HookedTransformer
import uuid
from datetime import datetime
import json
import os
import numpy as np


def run_eval(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
):
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_version = get_sae_bench_version()

    # TODO: Make this nicer.
    sae_map_df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T

    results_dict = {}

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
            sae = sae.to(device=device, dtype=llm_dtype)
            sae = _fix_topk(sae, sae_name, sae_release)

            k_sparse_probing_results = run_k_sparse_probing_experiment(
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

            sae_results = {}
            absorption_rates = []
            num_split_features = []
            for _, row in agg_df.iterrows():
                letter = row["letter"]
                absorption_rates.append(row["absorption_rate"])
                num_split_features.append(row["num_split_feats"])
                sae_results[f"absorption_first_letter_{letter}"] = {
                    "num_absorption": int(row["num_absorption"]),
                    "absorption_rate": float(row["absorption_rate"]),
                    "num_probe_true_positives": float(row["num_probe_true_positives"]),
                    "num_split_features": int(row["num_split_feats"]),
                }
            sae_results["mean_absorption_rate"] = statistics.mean(absorption_rates)
            sae_results["mean_num_split_features"] = statistics.mean(num_split_features)

            # Create artifacts subfolder
            artifacts_folder = os.path.join(output_path, "artifacts")
            os.makedirs(artifacts_folder, exist_ok=True)

            # Save k_sparse_probing_results as a separate JSON
            k_sparse_probing_file = f"{sae_release}_{sae_name}_k_sparse_probing.json"
            k_sparse_probing_file = k_sparse_probing_file.replace('/', '_')  # Replace '/' with '_' to avoid nested directories
            k_sparse_probing_path = os.path.join(artifacts_folder, k_sparse_probing_file)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(k_sparse_probing_path), exist_ok=True)
            k_sparse_probing_results.to_json(k_sparse_probing_path, orient='records', indent=4)

            sae_eval_result = {
                "eval_instance_id": eval_instance_id,
                "sae_lens_release": sae_release,
                "sae_lens_id": sae_id,
                "eval_type_id": "absorption",
                "sae_lens_version": sae_lens_version,
                "sae_bench_version": sae_bench_version,
                "date_time": datetime.now().isoformat(),
                "eval_config": asdict(config),
                "eval_results": sae_results,
                "eval_artifacts": {
                    "k_sparse_probing_results": os.path.relpath(k_sparse_probing_path, output_path)
                }
            }

            results_dict[f"{sae_release}_{sae_name}"] = sae_eval_result

            # Save individual SAE result
            sae_result_file = f"{sae_release}_{sae_name}_eval_results.json"
            sae_result_file = sae_result_file.replace('/', '_')
            sae_result_path = os.path.join(output_path, sae_result_file)
            
            with open(sae_result_path, 'w') as f:
                json.dump(sae_eval_result, f, indent=4)

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
    agg_df["absorption_rate"] = agg_df["num_absorption"] / agg_df["num_probe_true_positives"]
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


# This main function will produce the same results as the shift, tpp, and sparse probing main functions
if __name__ == "__main__":
    import time
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    start_time = time.time()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    config = eval_config.EvalConfig()

    # populate selected_saes_dict using config values
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

    # create output folder
    output_folder = "evals/absorption/results"
    os.makedirs(output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(config, config.selected_saes_dict, device, output_folder)

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")
