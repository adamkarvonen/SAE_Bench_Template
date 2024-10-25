import sae_bench_utils
import sae_bench_utils.formatting_utils as formatting_utils
import sae_bench_utils.testing_utils as testing_utils
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
    output = formatting_utils.average_results_dictionaries(results_dict, dataset_names)

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
