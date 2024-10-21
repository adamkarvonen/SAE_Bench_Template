import json

import torch

import evals.shift_and_tpp.eval_config as eval_config
import evals.shift_and_tpp.main as shift_and_tpp
import sae_bench_utils.formatting_utils as formatting_utils
import sae_bench_utils.testing_utils as testing_utils

tpp_results_filename = "tests/test_data/pythia-70m-deduped_tpp_layer_4_expected_eval_results.json"
scr_results_filename = "tests/test_data/pythia-70m-deduped_scr_layer_4_expected_eval_results.json"


# def test_end_to_end_matching_seed():
#     """Estimated runtime: 1 minute"""
#     if torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     print(f"Using device: {device}")

#     test_config = eval_config.EvalConfig()
#     test_config.sae_releases = [
#         "sae_bench_pythia70m_sweep_topk_ctx128_0730",
#     ]

#     test_config.dataset_names = ["bias_in_bios"]
#     test_config.model_name = "pythia-70m-deduped"
#     test_config.layer = 4
#     test_config.trainer_ids = [10]
#     test_config.include_checkpoints = False
#     test_config.random_seed = 42
#     tolerance = 0.0

#     # populate selected_saes_dict using config values
#     for release in test_config.sae_releases:
#         if "gemma-scope" in release:
#             test_config.selected_saes_dict[release] = (
#                 formatting_utils.find_gemmascope_average_l0_sae_names(test_config.layer)
#             )
#         else:
#             test_config.selected_saes_dict[release] = formatting_utils.filter_sae_names(
#                 sae_names=release,
#                 layers=[test_config.layer],
#                 include_checkpoints=test_config.include_checkpoints,
#                 trainer_ids=test_config.trainer_ids,
#             )

#         print(f"SAE release: {release}, SAEs: {test_config.selected_saes_dict[release]}")

#     run_results = sparse_probing.run_eval(test_config, test_config.selected_saes_dict, device)

#     with open(results_filename, "r") as f:
#         expected_results = json.load(f)

#     testing_utils.compare_dicts_within_tolerance(run_results, expected_results, tolerance)


def test_scr_end_to_end_same_seed():
    """Estimated runtime: 1 minute"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = eval_config.EvalConfig()
    test_config.sae_releases = [
        "sae_bench_pythia70m_sweep_topk_ctx128_0730",
    ]

    test_config.dataset_names = ["LabHC/bias_in_bios_class_set1"]
    test_config.model_name = "pythia-70m-deduped"
    test_config.layer = 4
    test_config.trainer_ids = [10]
    test_config.include_checkpoints = False
    test_config.random_seed = 42
    test_config.n_values = [2, 20]
    test_config.sae_batch_size = 250
    tolerance = 0.04

    test_config.spurious_corr = True
    test_config.column1_vals_list = [("professor", "nurse")]

    # populate selected_saes_dict using config values
    for release in test_config.sae_releases:
        if "gemma-scope" in release:
            test_config.selected_saes_dict[release] = (
                formatting_utils.find_gemmascope_average_l0_sae_names(test_config.layer)
            )
        else:
            test_config.selected_saes_dict[release] = formatting_utils.filter_sae_names(
                sae_names=release,
                layers=[test_config.layer],
                include_checkpoints=test_config.include_checkpoints,
                trainer_ids=test_config.trainer_ids,
            )

        print(f"SAE release: {release}, SAEs: {test_config.selected_saes_dict[release]}")

    run_results = shift_and_tpp.run_eval(test_config, test_config.selected_saes_dict, device)

    # This is required because when saving tuples are converted to lists
    run_results["custom_eval_config"]["column1_vals_list"] = [["professor", "nurse"]]

    with open(scr_results_filename, "r") as f:
        expected_results = json.load(f)

    testing_utils.compare_dicts_within_tolerance(run_results, expected_results, tolerance)


def test_tpp_end_to_end_different_seed():
    """Estimated runtime: 1 minute"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = eval_config.EvalConfig()
    test_config.sae_releases = [
        "sae_bench_pythia70m_sweep_topk_ctx128_0730",
    ]

    test_config.dataset_names = ["LabHC/bias_in_bios_class_set1"]
    test_config.model_name = "pythia-70m-deduped"
    test_config.layer = 4
    test_config.trainer_ids = [10]
    test_config.include_checkpoints = False
    test_config.random_seed = 44
    test_config.n_values = [2, 20]
    test_config.sae_batch_size = 250
    tolerance = 0.02

    test_config.spurious_corr = False

    # populate selected_saes_dict using config values
    for release in test_config.sae_releases:
        if "gemma-scope" in release:
            test_config.selected_saes_dict[release] = (
                formatting_utils.find_gemmascope_average_l0_sae_names(test_config.layer)
            )
        else:
            test_config.selected_saes_dict[release] = formatting_utils.filter_sae_names(
                sae_names=release,
                layers=[test_config.layer],
                include_checkpoints=test_config.include_checkpoints,
                trainer_ids=test_config.trainer_ids,
            )

        print(f"SAE release: {release}, SAEs: {test_config.selected_saes_dict[release]}")

    run_results = shift_and_tpp.run_eval(test_config, test_config.selected_saes_dict, device)

    with open(tpp_results_filename, "r") as f:
        expected_results = json.load(f)

    testing_utils.compare_dicts_within_tolerance(run_results, expected_results, tolerance)
