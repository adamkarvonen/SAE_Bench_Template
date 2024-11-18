import sae_bench_utils
import sae_bench_utils.general_utils as general_utils
import sae_bench_utils.indexing_utils as indexing_utils
import sae_bench_utils.testing_utils as testing_utils
import torch
import pytest
import re
import uuid


def test_average_results():
    # Prepare test data
    results_dict = {
        "dataset1_results": {"accuracy": 0.8, "loss": 0.5},
        "dataset2_results": {"accuracy": 0.85, "loss": 0.4},
    }
    dataset_names = ["dataset1", "dataset2"]

    # Expected output
    expected_output = {"accuracy": 0.825, "loss": 0.45}

    # Call the function
    output = general_utils.average_results_dictionaries(results_dict, dataset_names)

    testing_utils.compare_dicts_within_tolerance(output, expected_output, tolerance=1e-6)


def test_get_eval_uuid():
    # Test that the function returns a valid UUID
    eval_uuid = sae_bench_utils.get_eval_uuid()
    assert isinstance(eval_uuid, str)
    assert uuid.UUID(eval_uuid)  # This will raise an exception if the string is not a valid UUID


def test_get_sae_lens_version():
    # Test that the function returns a string
    version = sae_bench_utils.get_sae_lens_version()
    assert isinstance(version, str)
    # Check if it's either a version number or "Unknown"
    assert version == "Unknown" or re.match(r"^\d+\.\d+\.\d+", version)


def test_get_sae_bench_version():
    # Test that the function returns a string
    version = sae_bench_utils.get_sae_bench_version()
    assert isinstance(version, str)
    # Check if it's either a git hash (40 characters) or "Unknown"
    assert version == "Unknown" or (
        len(version) == 40 and all(c in "0123456789abcdef" for c in version)
    )


@pytest.mark.parametrize("num_calls", [1, 10, 100])
def test_get_eval_uuid_uniqueness(num_calls):
    # Test that multiple calls to get_eval_uuid() return unique values
    uuids = [sae_bench_utils.get_eval_uuid() for _ in range(num_calls)]
    assert len(set(uuids)) == num_calls, "Not all generated UUIDs are unique"


def test_indexing_utils():
    x = torch.arange(40).reshape((2, 20))
    x[0, 10] += 50  # 2nd highest value
    x[0, 11] += 100  # highest value
    x[1, 1] += 150  # not inside buffer (it's less than 3 from the start of the sequence)
    top_indices = indexing_utils.get_k_largest_indices(x, k=2, buffer=3, no_overlap=False)
    assert top_indices.tolist() == [[0, 11], [0, 10]]
    top_indices_no_overlap = indexing_utils.get_k_largest_indices(x, k=2, buffer=3, no_overlap=True)
    assert top_indices_no_overlap.tolist() == [[0, 11], [1, 16]]

    # TODO - add test here (should get high values that are still strictly below 10)
    # iw_indices = get_iw_sample_indices(x, k=5, buffer=3, threshold=10)
    # # print(x[iw_indices[:, 0], iw_indices[:, 1]])

    x_top_values_with_context = indexing_utils.index_with_buffer(x, top_indices, buffer=3)
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
