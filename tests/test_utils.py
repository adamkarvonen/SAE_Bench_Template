import sae_bench_utils.formatting_utils as formatting_utils
import sae_bench_utils.testing_utils as testing_utils


def test_average_results():
    # Prepare test data
    results_dict = {
        "dataset1_results": {
            "sae1": {"accuracy": 0.8, "loss": 0.5},
            "sae2": {"accuracy": 0.75, "loss": 0.6},
        },
        "dataset2_results": {
            "sae1": {"accuracy": 0.85, "loss": 0.4},
            "sae2": {"accuracy": 0.7, "loss": 0.65},
        },
    }
    dataset_names = ["dataset1", "dataset2"]

    # Expected output
    expected_output = {
        "sae1": {"accuracy": 0.825, "loss": 0.45},
        "sae2": {"accuracy": 0.725, "loss": 0.625},
    }

    # Call the function
    output = formatting_utils.average_results_dictionaries(results_dict, dataset_names)

    testing_utils.compare_dicts_within_tolerance(output, expected_output, tolerance=1e-6)
