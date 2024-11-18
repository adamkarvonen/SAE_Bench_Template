from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm.auto import tqdm
import re
from tabulate import tabulate


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append((release, sae_name, expected_var_explained, expected_l0))

    return all_loadable_saes


def get_saes_from_regex(sae_regex_pattern: str, sae_id_pattern: str) -> list[tuple[str, str]]:
    """
    Filter and retrieve SAEs based on regex patterns for release names and SAE IDs.

    This function searches through all loadable SAEs and returns those that match
    the provided regex patterns for both the release name and the SAE ID.

    Args:
        sae_regex_pattern (str): A regex pattern to match against SAE release names.
        sae_id_pattern (str): A regex pattern to match against SAE IDs.

    Returns:
        list[tuple[str, str]]: A list of tuples, where each tuple contains
        (release_name, sae_id) for SAEs matching both regex patterns.

    Example:
        >>> get_saes_from_regex(r"sae_bench_pythia.*", r"blocks\.4\.hook_resid_pre.*")
        [('sae_bench_pythia70m_sweep_standard_ctx128_0712', 'blocks.4.hook_resid_pre__trainer_0'),
         ('sae_bench_pythia70m_sweep_standard_ctx128_0712', 'blocks.4.hook_resid_pre__trainer_1'), ...]
    """
    sae_regex_compiled = re.compile(sae_regex_pattern)
    sae_id_compiled = re.compile(sae_id_pattern)
    all_saes = all_loadable_saes()
    filtered_saes = [
        sae
        for sae in all_saes
        if sae_regex_compiled.fullmatch(sae[0]) and sae_id_compiled.fullmatch(sae[1])
    ]

    # exclude the expected_var_explained and expected_l0 values
    filtered_saes = [(sae[0], sae[1]) for sae in filtered_saes]
    return filtered_saes


metadata_rows = [
    [data.model, data.release, data.repo_id, len(data.saes_map)]
    for data in get_pretrained_saes_directory().values()
]


# Print all SAE releases, sorted by base model
def print_all_sae_releases():
    """
    Print a table of all SAE releases, sorted by base model.
    """
    metadata_rows = [
        [data.model, data.release, data.repo_id, len(data.saes_map)]
        for data in get_pretrained_saes_directory().values()
    ]

    print(
        tabulate(
            sorted(metadata_rows, key=lambda x: x[0]),
            headers=["model", "release", "repo_id", "n_saes"],
            tablefmt="simple_outline",
        )
    )


def print_release_details(release_name: str):
    """
    Print details of a specific SAE release.

    Args:
    release_name (str): The name of the release to display details for.
    """

    def format_value(value):
        if isinstance(value, dict):
            if not value:
                return "{}"
            return "{{{0!r}: {1!r}, ...}}".format(*next(iter(value.items())))
        return repr(value)

    release = get_pretrained_saes_directory()[release_name]

    print(
        tabulate(
            [[k, format_value(v)] for k, v in release.__dict__.items()],
            headers=["Field", "Value"],
            tablefmt="simple_outline",
        )
    )


def select_saes_multiple_patterns(
    sae_regex_patterns: list[str],
    sae_block_patterns: list[str],
) -> list[tuple[str, str]]:
    assert len(sae_regex_patterns) == len(sae_block_patterns), "Length mismatch"

    selected_saes = []
    for sae_regex_pattern, sae_block_pattern in zip(sae_regex_patterns, sae_block_patterns):
        selected_saes.extend(get_saes_from_regex(sae_regex_pattern, sae_block_pattern))
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return selected_saes
