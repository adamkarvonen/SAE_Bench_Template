from sae_lens import SAE
from transformer_lens import HookedTransformer

from evals.absorption.feature_absorption import (
    StatsAndLikelyFalseNegativeResults,
    calculate_projection_and_cos_sims,
)
from evals.absorption.feature_absorption_calculator import FeatureAbsorptionCalculator
from evals.absorption.probing import LinearProbe
from evals.absorption.prompting import (
    VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS,
    first_letter_formatter,
)


def test_calculate_projection_and_cos_sims_gives_sane_results(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    fake_probe = LinearProbe(768, 26)
    calculator = FeatureAbsorptionCalculator(
        gpt2_model,
        icl_word_list=["dog", "cat", "fish", "bird"],
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
        answer_formatter=first_letter_formatter(),
    )
    # format: dict[letter: (num_true_positives, [split_feature_ids], [probable_feature_absorption_words])]
    likely_negs: dict[str, StatsAndLikelyFalseNegativeResults] = {
        "a": StatsAndLikelyFalseNegativeResults(
            10, 10, [1, 2, 3], [" Animal", " apple"]
        ),
        "b": StatsAndLikelyFalseNegativeResults(100, 100, [12], [" banana", " bear"]),
    }
    df = calculate_projection_and_cos_sims(
        calculator, gpt2_l4_sae, fake_probe, layer=4, likely_negs=likely_negs
    )
    assert df.columns.values.tolist() == [
        "letter",
        "token",
        "prompt",
        "num_probe_true_positives",
        "split_feats",
        "split_feat_acts",
        "split_feat_probe_cos",
        "top_projection_feat",
        "top_probe_projection",
        "top_projection_feat_probe_cos",
        "second_projection_feat",
        "second_probe_projection",
        "second_projection_feat_probe_cos",
        "probe_projections",
        "projection_feats",
        "projection_feat_acts",
        "projection_feat_probe_cos",
        "is_absorption",
    ]
