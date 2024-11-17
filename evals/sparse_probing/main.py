import gc
import os
import shutil
import random
import time
from dataclasses import asdict
from pydantic import TypeAdapter
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime
from evals.sparse_probing.eval_config import SparseProbingEvalConfig
from evals.sparse_probing.eval_output import (
    EVAL_TYPE_ID_SPARSE_PROBING,
    SparseProbingEvalOutput,
    SparseProbingLlmMetrics,
    SparseProbingMetricCategories,
    SparseProbingResultDetail,
    SparseProbingSaeMetrics,
)
import evals.sparse_probing.probe_training as probe_training
import sae_bench_utils.activation_collection as activation_collection
import sae_bench_utils.dataset_info as dataset_info
import sae_bench_utils.dataset_utils as dataset_utils
import sae_bench_utils.general_utils as general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
    select_saes_multiple_patterns,
)


def average_test_accuracy(test_accuracies: dict[str, float]) -> float:
    return sum(test_accuracies.values()) / len(test_accuracies)


def get_dataset_activations(
    dataset_name: str,
    config: SparseProbingEvalConfig,
    model: HookedTransformer,
    llm_batch_size: int,
    layer: int,
    hook_point: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_data, test_data = dataset_utils.get_multi_label_train_test_data(
        dataset_name,
        config.probe_train_set_size,
        config.probe_test_set_size,
        config.random_seed,
    )

    chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]

    train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
    test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    train_data = dataset_utils.tokenize_data_dictionary(
        train_data, model.tokenizer, config.context_length, device
    )
    test_data = dataset_utils.tokenize_data_dictionary(
        test_data, model.tokenizer, config.context_length, device
    )

    all_train_acts_BLD = activation_collection.get_all_llm_activations(
        train_data, model, llm_batch_size, layer, hook_point, mask_bos_pad_eos_tokens=True
    )
    all_test_acts_BLD = activation_collection.get_all_llm_activations(
        test_data, model, llm_batch_size, layer, hook_point, mask_bos_pad_eos_tokens=True
    )

    return all_train_acts_BLD, all_test_acts_BLD


def run_eval_single_dataset(
    dataset_name: str,
    config: SparseProbingEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    hook_point: str,
    device: str,
    artifacts_folder: str,
    save_activations: bool,
) -> dict[str, float]:
    """config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility."""

    results_dict = {}

    activations_filename = f"{dataset_name}_activations.pt".replace("/", "_")

    activations_path = os.path.join(artifacts_folder, activations_filename)

    if not os.path.exists(activations_path):
        all_train_acts_BLD, all_test_acts_BLD = get_dataset_activations(
            dataset_name,
            config,
            model,
            config.llm_batch_size,
            layer,
            hook_point,
            device,
        )

        all_train_acts_BD = activation_collection.create_meaned_model_activations(
            all_train_acts_BLD
        )

        all_test_acts_BD = activation_collection.create_meaned_model_activations(all_test_acts_BLD)

        llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
            all_train_acts_BD,
            all_test_acts_BD,
            select_top_k=None,
        )

        llm_results = {"llm_test_accuracy": average_test_accuracy(llm_test_accuracies)}

        llm_test_accuracy = average_test_accuracy(llm_test_accuracies)

        for k in config.k_values:
            llm_top_k_probes, llm_top_k_test_accuracies = probe_training.train_probe_on_activations(
                all_train_acts_BD,
                all_test_acts_BD,
                select_top_k=k,
            )
            llm_results[f"llm_top_{k}_test_accuracy"] = average_test_accuracy(
                llm_top_k_test_accuracies
            )

        acts = {
            "train": all_train_acts_BLD,
            "test": all_test_acts_BLD,
            "llm_results": llm_results,
        }

        if save_activations:
            torch.save(acts, activations_path)
    else:
        print(f"Loading activations from {activations_path}")
        acts = torch.load(activations_path)
        all_train_acts_BLD = acts["train"]
        all_test_acts_BLD = acts["test"]
        llm_results = acts["llm_results"]

    all_train_acts_BD = activation_collection.create_meaned_model_activations(all_train_acts_BLD)
    all_test_acts_BD = activation_collection.create_meaned_model_activations(all_test_acts_BLD)

    all_sae_train_acts_BF = activation_collection.get_sae_meaned_activations(
        all_train_acts_BLD, sae, config.sae_batch_size
    )
    all_sae_test_acts_BF = activation_collection.get_sae_meaned_activations(
        all_test_acts_BLD, sae, config.sae_batch_size
    )

    # This is optional, checking the accuracy of a probe trained on the entire SAE activations
    # We use GPU here as sklearn.fit is slow on large input dimensions, all other probe training is done with sklearn.fit
    _, sae_test_accuracies = probe_training.train_probe_on_activations(
        all_sae_train_acts_BF,
        all_sae_test_acts_BF,
        select_top_k=None,
        use_sklearn=False,
        batch_size=250,
        epochs=100,
        lr=1e-2,
    )

    for llm_result_key, llm_result_value in llm_results.items():
        results_dict[llm_result_key] = llm_result_value

    results_dict["sae_test_accuracy"] = average_test_accuracy(sae_test_accuracies)

    for k in config.k_values:
        sae_top_k_probes, sae_top_k_test_accuracies = probe_training.train_probe_on_activations(
            all_sae_train_acts_BF,
            all_sae_test_acts_BF,
            select_top_k=k,
        )
        results_dict[f"sae_top_{k}_test_accuracy"] = average_test_accuracy(
            sae_top_k_test_accuracies
        )

    return results_dict


def run_eval_single_sae(
    config: SparseProbingEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    artifacts_folder: str,
    save_activations: bool = True,
) -> dict[str, float | dict[str, float]]:
    """hook_point: str is transformer lens format. example: f'blocks.{layer}.hook_resid_post'
    By default, we save activations for all datasets, and then reuse them for each sae.
    This is important to avoid recomputing activations for each SAE, and to ensure that the same activations are used for all SAEs.
    However, it can use 10s of GBs of disk space."""

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    os.makedirs(artifacts_folder, exist_ok=True)

    results_dict = {}

    dataset_results = {}
    for dataset_name in config.dataset_names:
        dataset_results[f"{dataset_name}_results"] = run_eval_single_dataset(
            dataset_name,
            config,
            sae,
            model,
            sae.cfg.hook_layer,
            sae.cfg.hook_name,
            device,
            artifacts_folder,
            save_activations,
        )

    results_dict = general_utils.average_results_dictionaries(dataset_results, config.dataset_names)

    for dataset_name, dataset_result in dataset_results.items():
        results_dict[f"{dataset_name}"] = dataset_result

    return results_dict


def run_eval(
    config: SparseProbingEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
):
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)

    If clean_up_activations is True, which means that the activations are deleted after the evaluation is done.
    You may want to use this because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of SAE name: evaluation results for that SAE."""
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    artifacts_base_folder = "artifacts"
    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    if config.llm_dtype == "bfloat16":
        llm_dtype = torch.bfloat16
    elif config.llm_dtype == "float32":
        llm_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {config.llm_dtype}")

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

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

        artifacts_folder = os.path.join(
            artifacts_base_folder,
            EVAL_TYPE_ID_SPARSE_PROBING,
            config.model_name,
            sae.cfg.hook_name,
        )

        sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        sae_result_file = sae_result_file.replace("/", "_")
        sae_result_path = os.path.join(output_path, sae_result_file)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Loading existing results from {sae_result_path}")
            with open(sae_result_path, "r") as f:
                eval_output = TypeAdapter(SparseProbingEvalOutput).validate_json(f.read())
        else:
            sparse_probing_results = run_eval_single_sae(
                config,
                sae,
                model,
                device,
                artifacts_folder,
                save_activations=save_activations,
            )
            eval_output = SparseProbingEvalOutput(
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=SparseProbingMetricCategories(
                    llm=SparseProbingLlmMetrics(
                        **{
                            k: v
                            for k, v in sparse_probing_results.items()
                            if k.startswith("llm_") and not isinstance(v, dict)
                        }
                    ),
                    sae=SparseProbingSaeMetrics(
                        **{
                            k: v
                            for k, v in sparse_probing_results.items()
                            if k.startswith("sae_") and not isinstance(v, dict)
                        }
                    ),
                ),
                eval_result_details=[
                    SparseProbingResultDetail(
                        dataset_name=dataset_name,
                        **result,
                    )
                    for dataset_name, result in sparse_probing_results.items()
                    if isinstance(result, dict)
                ],
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=sae_id,
                sae_lens_release_id=sae_release,
                sae_lens_version=sae_lens_version,
            )

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

    if clean_up_activations:
        if os.path.exists(artifacts_folder):
            shutil.rmtree(artifacts_folder)

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[SparseProbingEvalConfig, list[tuple[str, str]]]:
    config = SparseProbingEvalConfig(
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
    parser = argparse.ArgumentParser(description="Run sparse probing evaluation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, help="Model name")
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
    parser.add_argument(
        "--output_folder",
        type=str,
        default="evals/sparse_probing/results",
        help="Output folder",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations",
        action="store_true",
        help="Clean up activations after evaluation",
    )
    parser.add_argument(
        "--save_activations",
        action="store_false",
        help="Save the generated LLM activations for later use",
    )

    return parser


if __name__ == "__main__":
    """
    python evals/sparse_probing/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped


    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    config.llm_dtype = str(activation_collection.LLM_NAME_TO_DTYPE[config.model_name]).split(".")[
        -1
    ]

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        args.force_rerun,
        args.clean_up_activations,
        args.save_activations,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import baselines.identity_sae as identity_sae
#     import baselines.jumprelu_sae as jumprelu_sae
#     """
#     python evals/sparse_probing/main.py
#     """
#     device = formatting_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "evals/sparse_probing/results"

#     baseline_type = "identity_sae"
#     baseline_type = "jumprelu_sae"

#     model_name = "pythia-70m-deduped"
#     hook_layer = 4
#     d_model = 512

#     model_name = "gemma-2-2b"
#     hook_layer = 19
#     d_model = 2304

#     if baseline_type == "identity_sae":
#         sae = identity_sae.IdentitySAE(model_name, d_model=d_model, hook_layer=hook_layer)
#         selected_saes = [(f"{model_name}_layer_{hook_layer}_identity_sae", sae)]
#     elif baseline_type == "jumprelu_sae":
#         repo_id = "google/gemma-scope-2b-pt-res"
#         filename = "layer_20/width_16k/average_l0_71/params.npz"
#         sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, 20)
#         selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]
#     else:
#         raise ValueError(f"Invalid baseline type: {baseline_type}")

#     config = SparseProbingEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#     )

#     config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
#     config.llm_dtype = str(activation_collection.LLM_NAME_TO_DTYPE[config.model_name]).split(".")[
#         -1
#     ]

#     # create output folder
#     os.makedirs(output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         output_folder,
#         force_rerun=True,
#         clean_up_activations=False,
#         save_activations=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")
