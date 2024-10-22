from typing import Optional


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
