import os
import gc
import time
import random
import argparse
from tqdm import tqdm

import torch
from sae_lens import SAE
from nnsight import LanguageModel
import matplotlib.pyplot as plt

import sae_bench_utils.activation_collection as activation_collection
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils import general_utils as general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)

from evals.ravel.instance import create_filtered_dataset
from evals.ravel.eval_config import RAVELEvalConfig
from evals.ravel.uniprobe import run_feature_selection_probe
from evals.ravel.intervention import compute_disentanglement_AB_bidirectional


DEBUG_MODE = False
MODEL_DIR = "/share/u/can/models" # Set to None to disable model caching
REPO_DIR = "/share/u/can/SAE_Bench_Template"
RESULTS_DIR = "/share/u/can/SAE_Bench_Template/evals/ravel/results"
ARTIFACTS_DIR = "/share/u/can/SAE_Bench_Template/evals/ravel/artifacts"

eval_config = RAVELEvalConfig()
rng = random.Random(eval_config.random_seed)


# TODO test for MODEL_ID = 'pythia-70m'
def run_eval(
    eval_config: RAVELEvalConfig,
    selected_saes: list[tuple[str, str]],
    device: torch.device,
):
    # Instanciate evaluation run
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    # Load model
    if eval_config.model_name == "pythia-70m":
        model_id = "eleutherAI/pythia-70m-deduped"
        model_kwargs = {}
    elif eval_config.model_name == "gemma-2-2b":
        model_id = "google/gemma-2-2b"
        model_kwargs = {'low_cpu_mem_usage': True,'attn_implementation': 'eager'}
    else:
        raise ValueError(f"Invalid model name: {eval_config.model_name}")

    if eval_config.llm_dtype == "bfloat16":
        llm_dtype = torch.bfloat16
    elif eval_config.llm_dtype == "float32":
        llm_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {eval_config.llm_dtype}")

    model = LanguageModel(
        model_id,
        cache_dir=MODEL_DIR,
        device_map=device,
        torch_dtype=llm_dtype,
        dispatch=True,
        **model_kwargs,
    )
    model.requires_grad_(False)
    model.eval()

    # Initialize directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results_dict = {}

    # Iteratively evaluate each SAE
    for sae_release, sae_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        gc.collect()
        torch.cuda.empty_cache()

        # Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)
        if isinstance(sae_id, str):
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
        else:
            sae = sae_id
            sae_id = "custom_sae"

        sae = sae.to(device=device, dtype=llm_dtype)

        # # Initialize SAE specific directories
        # artifacts_folder = os.path.join(
        #     artifacts_base_folder, eval_type, eval_config.model_name, sae.cfg.hook_name
        # )
        # sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        # sae_result_file = sae_result_file.replace("/", "_")
        # sae_result_path = os.path.join(output_path, sae_result_file)


        if DEBUG_MODE:
            n_samples_per_attribute_class = 50
            top_n_entities = 10
            top_n_templates = 2
        else:
            n_samples_per_attribute_class = None
            top_n_entities = 400
            top_n_templates = 12


        dataset = create_filtered_dataset(
            model_id=model_id,
            chosen_entity="nobel_prize_winner",
            model=model,
            REPO_DIR=REPO_DIR,
            force_recompute=False,
            n_samples_per_attribute_class=n_samples_per_attribute_class,
            top_n_entities=top_n_entities,
            top_n_templates=top_n_templates,
        )


        # TODO save results
        all_attributes = dataset.get_attributes()
        print(f"All attributes: {all_attributes}")

        attribute_feature_dict = run_feature_selection_probe(
            model,
            sae,
            dataset,
            all_attributes=all_attributes,
            coeffs=[0.01, 0.1, 10, 100, 1000],
            max_samples_per_attribute=1024,
            layer=11,
            llm_batch_size=512,
        )


        results = compute_disentanglement_AB_bidirectional(
            model,
            sae,
            dataset,
            attribute_A="Field",
            attribute_B="Country of Birth",
            attribute_feature_dict=attribute_feature_dict,
            n_interventions=128,
            n_generated_tokens=8,
            llm_batch_size=16,
            tracer_kwargs={"scan": False, "validate": False},
        )

        


        def plot_disentanglement(results, title):
            thresholds = list(results.keys())
            print(results)
            cause_A = [results[t]["cause_A"] for t in thresholds]
            isolation_BtoA = [results[t]["isolation_BtoA"] for t in thresholds]
            cause_B = [results[t]["cause_B"] for t in thresholds]
            isolation_AtoB = [results[t]["isolation_AtoB"] for t in thresholds]
            disentanglement = [results[t]["disentanglement"] for t in thresholds]

            fig_dir = os.path.join(RESULTS_DIR, "disentanglement_plot.png")
            plt.figure(figsize=(12, 6))
            plt.plot(thresholds, cause_A, label="Cause A")
            plt.plot(thresholds, isolation_BtoA, label="Isolation B->A")
            plt.plot(thresholds, cause_B, label="Cause B")
            plt.plot(thresholds, isolation_AtoB, label="Isolation A->B")
            plt.plot(thresholds, disentanglement, label="Disentanglement")
            plt.xlabel("Threshold")
            plt.ylabel("Accuracy")
            plt.xscale("log")
            plt.title(title)
            plt.legend()
            plt.savefig(fig_dir)
            plt.show()


        plot_disentanglement(results, "Disentanglement score for Field and Country of Birth")



def create_config_and_selected_saes(
    args,
) -> tuple[RAVELEvalConfig, list[tuple[str, str]]]:
    config = RAVELEvalConfig(
        random_seed=args.random_seed,
        model_name=args.model_name,
    )

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run SCR or TPP evaluation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, default="pythia-70m-deduped", help="Model name")
    parser.add_argument("--layer", type=int, help="SAE layer")
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    return parser


if __name__ == "__main__":
    """
    Example Gemma-2-2B SAE Bench usage:
    python evals/ravel/main.py \
    --sae_regex_pattern "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824" \
    --sae_block_pattern "blocks.19.hook_resid_post__trainer_2" \
    --model_name "gemma-2-2b" \
    --layer 11 
    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]
    config.layer = args.layer

    # create output folder
    # os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")