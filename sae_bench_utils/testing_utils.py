from datetime import datetime
import uuid
from typing import Dict, Any, Optional
from beartype import beartype

import json
from datetime import datetime
import uuid
from typing import Dict, Optional
from beartype import beartype
from argparse import ArgumentParser
from typing import Optional, Set

@beartype
def validate_eval_output_format(
    output_path: str,
    eval_type: str,
    additional_required_config: Optional[Dict[str, type]] = None,
    additional_required_results: Optional[Dict[str, type]] = None
) -> None:
    """Validates that an eval output JSON file matches the required format from eval_template.ipynb
    
    Args:
        output_path: Path to the JSON file containing the output to validate
        eval_type: Expected eval_type_id value
        additional_required_config: Additional required fields in eval_config and their types
        additional_required_results: Additional required fields in eval_results and their types
    
    Raises:
        AssertionError: If any validation fails
        ValueError: If UUID or datetime format is invalid
        FileNotFoundError: If the specified JSON file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    try:
        with open(output_path, 'r') as f:
            output_dict = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified JSON file does not exist: {output_path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"The file at {output_path} is not valid JSON")

    # Required top-level fields from template
    required_fields = {
        "eval_instance_id": str,
        "sae_lens_release": str,
        "sae_lens_id": str, 
        "eval_type_id": str,
        "sae_lens_version": str,
        "sae_bench_version": str,
        "date_time": str,
        "eval_config": dict,
        "eval_results": dict
    }
    
    # Validate all required fields exist with correct types
    for field, expected_type in required_fields.items():
        assert field in output_dict, f"Missing required field: {field}, current keys: {output_dict.keys()}"
        assert isinstance(output_dict[field], expected_type), \
            f"Field {field} has wrong type. Expected {expected_type}, got {type(output_dict[field])}"
    
    # Validate eval_type matches
    assert output_dict["eval_type_id"] == eval_type, \
        f"eval_type_id mismatch. Expected {eval_type}, got {output_dict['eval_type_id']}"
    
    # Validate UUID format
    try:
        uuid.UUID(output_dict["eval_instance_id"])
    except ValueError as e:
        raise ValueError(f"Invalid UUID format: {output_dict['eval_instance_id']}") from e
    
    # Validate datetime format
    try:
        datetime.strptime(output_dict["date_time"], "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {output_dict['date_time']}") from e
    
    # Validate additional required config fields if specified
    if additional_required_config:
        for field, expected_type in additional_required_config.items():
            assert field in output_dict["eval_config"], \
                f"Missing required config field: {field}"
            assert isinstance(output_dict["eval_config"][field], expected_type), \
                f"Config field {field} has wrong type. Expected {expected_type}, got {type(output_dict['eval_config'][field])}"
    
    # Validate additional required results fields if specified
    if additional_required_results:
        for field, expected_type in additional_required_results.items():
            assert field in output_dict["eval_results"], \
                f"Missing required results field: {field}"
            assert isinstance(output_dict["eval_results"][field], expected_type), \
                f"Results field {field} has wrong type. Expected {expected_type}, got {type(output_dict['eval_results'][field])}"


def validate_eval_cli_interface(
    parser: ArgumentParser,
    eval_config_cls: Optional[object] = None,
    additional_required_args: Optional[Set[str]] = None
) -> None:
    """Validates that an eval's CLI interface meets the requirements from eval_template.ipynb
    
    Args:
        parser: The ArgumentParser instance to validate
        eval_config_cls: The eval's config dataclass (optional). If provided, verifies CLI args match config fields
        additional_required_args: Any additional required arguments specific to this eval
        
    Raises:
        AssertionError: If validation fails with details about what's missing/incorrect
    """
    # Get all argument names (excluding help)
    all_args = {action.dest for action in parser._actions if action.dest != 'help'}
    
    # Required common arguments from template
    common_args = {
        "sae_regex_pattern",
        "sae_block_pattern",
        "output_folder",
        "force_rerun",
    }
    
    # Add any eval-specific required args
    if additional_required_args:
        common_args.update(additional_required_args)
    
    # Check all required args are present
    missing_args = common_args - all_args
    assert not missing_args, f"Missing required CLI arguments: {missing_args}"
    
    # If config class provided, verify CLI args match config fields
    if eval_config_cls:
        config_fields = {field for field in eval_config_cls.__dataclass_fields__}
        # model_name is a special case that's both common and in config
        config_fields.add("model_name")
        
        # Get args that should match config (excluding common args)
        eval_specific_args = all_args - common_args
        
        # Check for mismatches between CLI args and config
        missing_config_args = config_fields - eval_specific_args
        extra_cli_args = eval_specific_args - config_fields
        
        assert not missing_config_args, \
            f"Config fields missing from CLI args: {missing_config_args}"
        assert not extra_cli_args, \
            f"CLI args not present in config: {extra_cli_args}"

    # Verify help text exists for all arguments
    for action in parser._actions:
        if action.dest != 'help':
            assert action.help is not None and action.help != '', \
                f"Missing help text for argument: {action.dest}"


def compare_dicts_within_tolerance(
    actual,
    expected,
    tolerance: float,
    path: str = "",
    all_diffs=None,
    ignore_keys: tuple[str] = ("random_seed",),
    keys_to_compare: Optional[list[str]] = None,
):
    """
    Recursively compare two nested dictionaries and assert that all numeric values
    are within the specified tolerance. Print global mean and max difference at root call.

    :param actual: The actual dictionary of results
    :param expected: The expected dictionary of results
    :param tolerance: The allowed tolerance for floating point comparisons
    :param path: The current path in the nested structure (used for error messages)
    :param all_diffs: List to collect all differences (used internally for recursion)
    :param ignore_keys: Tuple of keys to ignore in the comparison
    :param keys_to_compare: Optional list of keys to compare. If provided, only compare
                          values whose leaf key name matches one in this list
    """

    if all_diffs is None:
        all_diffs = []

    assert isinstance(
        actual, type(expected)
    ), f"Type mismatch at {path}: {type(actual)} != {type(expected)}"

    if not isinstance(actual, dict) and keys_to_compare is not None:
        if path.split(".")[-1] not in keys_to_compare:
            return

    if isinstance(actual, dict):
        assert set(actual.keys()) == set(
            expected.keys()
        ), f"Key mismatch at {path}: {set(actual.keys())} != {set(expected.keys())}"
        for key in actual:
            new_path = f"{path}.{key}" if path else str(key)

            if key in ignore_keys:
                continue

            compare_dicts_within_tolerance(
                actual[key],
                expected[key],
                tolerance,
                new_path,
                all_diffs,
                ignore_keys,
                keys_to_compare,
            )
    elif isinstance(actual, (int, float)):
        diff = abs(actual - expected)
        all_diffs.append(diff)
    else:
        assert actual == expected, f"Value mismatch at {path}: {actual} != {expected}"

    # Print global mean and max difference only at the root call
    if path == "":
        if all_diffs:
            mean_diff = sum(all_diffs) / len(all_diffs)
            max_diff = max(all_diffs)

            print(f"Global mean difference: {mean_diff}")
            print(f"Global max difference: {max_diff}")

            assert (
                max_diff <= tolerance
            ), f"Value mismatch at {path}: {actual} not within {tolerance} of {expected}"

        else:
            print("No numeric differences found.")
