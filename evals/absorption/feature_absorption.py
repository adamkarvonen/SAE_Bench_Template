from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from evals.absorption.common import (
    RESULTS_DIR,
    PROBES_DIR,
    get_or_make_dir,
    load_df_or_run,
    load_experiment_df,
    load_probe,
)
from evals.absorption.k_sparse_probing import (
    SPARSE_PROBING_EXPERIMENT_NAME,
    add_feature_splits_to_metrics_df,
    get_sparse_probing_metrics_filename,
    get_sparse_probing_raw_results_filename,
)
from evals.absorption.feature_absorption_calculator import FeatureAbsorptionCalculator
from evals.absorption.probing import LinearProbe
from evals.absorption.prompting import (
    first_letter_formatter,
)
from evals.absorption.vocab import LETTERS, get_alpha_tokens

FEATURE_ABSORPTION_EXPERIMENT_NAME = "feature_absorption"

# the cosine similarity between the top projecting feature and the probe must be at least this high
ABSORPTION_PROBE_COS_THRESHOLD = 0.025

# the top projecting feature must contribute at least this much to the total probe projection to count as absorption
ABSORPTION_PROBE_PROJECTION_PROPORTION_THRESHOLD = 0.4


@dataclass
class StatsAndLikelyFalseNegativeResults:
    probe_true_positives: int
    split_feats_true_positives: int
    split_feats: list[int]
    potential_false_negatives: list[str]


def calculate_projection_and_cos_sims(
    calculator: FeatureAbsorptionCalculator,
    sae: SAE,
    probe: LinearProbe,
    layer: int,
    likely_negs: dict[str, StatsAndLikelyFalseNegativeResults],
) -> pd.DataFrame:
    results = []
    for letter, stats in tqdm(likely_negs.items()):
        assert calculator.model.tokenizer is not None
        absorption_results = calculator.calculate_absorption(
            sae,
            layer=layer,
            words=stats.potential_false_negatives,
            probe_direction=probe.weights[LETTERS.index(letter)],
            main_feature_ids=stats.split_feats,
            show_progress=False,
        )
        for sample in absorption_results.word_results:
            top_feat_score = sample.top_projection_feature_scores[0]
            second_feat_score = sample.top_projection_feature_scores[1]
            result = {
                "letter": letter,
                "token": sample.word,
                "prompt": sample.prompt,
                "num_probe_true_positives": stats.probe_true_positives,
                "split_feats": stats.split_feats,
                "split_feat_acts": [
                    score.activation for score in sample.main_feature_scores
                ],
                "split_feat_probe_cos": [
                    score.probe_cos_sim for score in sample.main_feature_scores
                ],
                "top_projection_feat": top_feat_score.feature_id,
                "top_probe_projection": top_feat_score.probe_projection,
                "top_projection_feat_probe_cos": top_feat_score.probe_cos_sim,
                "second_projection_feat": second_feat_score.feature_id,
                "second_probe_projection": second_feat_score.probe_projection,
                "second_projection_feat_probe_cos": second_feat_score.probe_cos_sim,
                "probe_projections": [
                    score.probe_projection
                    for score in sample.top_projection_feature_scores
                ],
                "projection_feats": [
                    score.feature_id for score in sample.top_projection_feature_scores
                ],
                "projection_feat_acts": [
                    score.activation for score in sample.top_projection_feature_scores
                ],
                "projection_feat_probe_cos": [
                    score.probe_cos_sim
                    for score in sample.top_projection_feature_scores
                ],
                "is_absorption": sample.is_absorption,
            }
            results.append(result)
    result_df = pd.DataFrame(results)
    return result_df


def get_stats_and_likely_false_negative_tokens(
    metrics_df: pd.DataFrame,
    sae_name: str,
    layer: int,
    sparse_probing_task_output_dir: Path,
) -> dict[str, StatsAndLikelyFalseNegativeResults]:
    """
    Examine the k-sparse probing results and look for false-negative cases where the k top feats don't fire but our LR probe does
    """
    results: dict[str, StatsAndLikelyFalseNegativeResults] = {}
    raw_df = load_experiment_df(
        SPARSE_PROBING_EXPERIMENT_NAME,
        sparse_probing_task_output_dir
        / get_sparse_probing_raw_results_filename(sae_name, layer),
    )
    for letter in LETTERS:
        split_feats = metrics_df[metrics_df["letter"] == letter]["split_feats"].iloc(  # type: ignore
            0
        )[
            0
        ]
        k = len(split_feats)
        potential_false_negatives = raw_df[
            (raw_df["answer_letter"] == letter)
            & (raw_df[f"score_probe_{letter}"] > 0)
            & (raw_df[f"score_sparse_sae_{letter}_k_{k}"] <= 0)
        ]["token"].tolist()
        num_split_feats_true_positives = raw_df[
            (raw_df["answer_letter"] == letter)
            & (raw_df[f"score_probe_{letter}"] > 0)
            & (raw_df[f"score_sparse_sae_{letter}_k_{k}"] > 0)
        ].shape[0]
        num_probe_true_positives = raw_df[
            (raw_df["answer_letter"] == letter) & (raw_df[f"score_probe_{letter}"] > 0)
        ].shape[0]
        results[letter] = StatsAndLikelyFalseNegativeResults(
            probe_true_positives=num_probe_true_positives,
            split_feats_true_positives=num_split_feats_true_positives,
            split_feats=split_feats,
            potential_false_negatives=potential_false_negatives,
        )
    return results


def load_and_run_calculate_projections_and_cos_sims(
    model: HookedTransformer,
    sae: SAE,
    calculator: FeatureAbsorptionCalculator,
    metrics_df: pd.DataFrame,
    sae_name: str,
    layer: int,
    probes_dir: Path | str,
    sparse_probing_task_output_dir: Path,
) -> pd.DataFrame:
    probe = load_probe(
        model_name=model.cfg.model_name, layer=layer, probes_dir=probes_dir
    )
    likely_negs = get_stats_and_likely_false_negative_tokens(
        metrics_df, sae_name, layer, sparse_probing_task_output_dir
    )
    return calculate_projection_and_cos_sims(
        calculator, sae, probe, likely_negs=likely_negs, layer=layer
    )


def run_feature_absortion_experiment(
    model: HookedTransformer,
    sae: SAE,
    layer: int,
    sae_name: str,
    max_k_value: int,
    prompt_template: str,
    prompt_token_pos: int,
    experiment_dir: Path | str = RESULTS_DIR / SPARSE_PROBING_EXPERIMENT_NAME,
    sparse_probing_experiment_dir: Path | str = RESULTS_DIR
    / SPARSE_PROBING_EXPERIMENT_NAME,
    probes_dir: Path | str = PROBES_DIR,
    force: bool = False,
    feature_split_f1_jump_threshold: float = 0.03,
    batch_size: int = 10,
) -> pd.DataFrame:
    """
    NOTE: this experiments requires the results of the k-sparse probing experiments. Make sure to run them first.
    """
    task_output_dir = get_or_make_dir(experiment_dir) / sae_name
    sparse_probing_task_output_dir = (
        get_or_make_dir(sparse_probing_experiment_dir) / sae_name
    )

    vocab = get_alpha_tokens(model.tokenizer)  # type: ignore
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=vocab,
        max_icl_examples=10,
        base_template=prompt_template,
        answer_formatter=first_letter_formatter(),
        word_token_pos=prompt_token_pos,
        probe_cos_sim_threshold=ABSORPTION_PROBE_COS_THRESHOLD,
        probe_projection_proportion_threshold=ABSORPTION_PROBE_PROJECTION_PROPORTION_THRESHOLD,
        batch_size=batch_size,
    )
    metrics_df = load_experiment_df(
        SPARSE_PROBING_EXPERIMENT_NAME,
        sparse_probing_task_output_dir
        / get_sparse_probing_metrics_filename(sae_name, layer),
    )
    add_feature_splits_to_metrics_df(
        metrics_df,
        max_k_value=max_k_value,
        f1_jump_threshold=feature_split_f1_jump_threshold,
    )
    df_path = task_output_dir / f"layer_{layer}_{sae_name}.parquet"
    df = load_df_or_run(
        lambda: load_and_run_calculate_projections_and_cos_sims(
            model,
            sae,
            calculator,
            metrics_df,
            sae_name=sae_name,
            layer=layer,
            probes_dir=probes_dir,
            sparse_probing_task_output_dir=sparse_probing_task_output_dir,
        ),
        df_path,
        force=force,
    )
    return df
