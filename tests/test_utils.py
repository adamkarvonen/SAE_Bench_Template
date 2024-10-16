import sae_bench_utils.formatting_utils as formatting_utils
import sae_bench_utils.indexing_utils as indexing_utils
import sae_bench_utils.testing_utils as testing_utils
import torch


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

    testing_utils.compare_dicts_within_tolerance(
        output, expected_output, tolerance=1e-6
    )


def test_indexing_utils():
    x = torch.arange(40).reshape((2, 20))
    x[0, 10] += 50  # 2nd highest value
    x[0, 11] += 100  # highest value
    x[1, 1] += (
        150  # not inside buffer (it's less than 3 from the start of the sequence)
    )
    top_indices = indexing_utils.get_k_largest_indices(
        x, k=2, buffer=3, no_overlap=False
    )
    assert top_indices.tolist() == [[0, 11], [0, 10]]
    top_indices_no_overlap = indexing_utils.get_k_largest_indices(
        x, k=2, buffer=3, no_overlap=True
    )
    assert top_indices_no_overlap.tolist() == [[0, 11], [1, 16]]

    # TODO - add test here (should get high values that are still strictly below 10)
    # iw_indices = get_iw_sample_indices(x, k=5, buffer=3, threshold=10)
    # # print(x[iw_indices[:, 0], iw_indices[:, 1]])

    x_top_values_with_context = indexing_utils.index_with_buffer(
        x, top_indices, buffer=3
    )
    assert x_top_values_with_context[0].tolist() == [
        8,
        9,
        10 + 50,
        11 + 100,
        12,
        13,
        14,
    ]  # highest value in the middle
    assert x_top_values_with_context[1].tolist() == [
        7,
        8,
        9,
        10 + 50,
        11 + 100,
        12,
        13,
    ]  # 2nd highest value in the middle
