from dataclasses import dataclass

import torch
from sae_lens import SAE
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

from evals.absorption.prompting import (
    Formatter,
    SpellingPrompt,
    create_icl_prompt,
    first_letter_formatter,
)
from evals.absorption.util import batchify

EPS = 1e-8


@dataclass
class FeatureScore:
    feature_id: int
    activation: float
    probe_cos_sim: float

    @property
    def probe_projection(self) -> float:
        return self.activation * self.probe_cos_sim


@dataclass
class WordAbsorptionResult:
    word: str
    prompt: str
    probe_projection: float
    main_feature_scores: list[FeatureScore]
    top_projection_feature_scores: list[FeatureScore]
    is_absorption: bool


@dataclass
class AbsorptionResults:
    main_feature_ids: list[int]
    word_results: list[WordAbsorptionResult]


@dataclass
class FeatureAbsorptionCalculator:
    """
    Feature absorption calculator for spelling tasks.

    Absorption is defined by the following criteria:
    - The main features for a concept do not fire
    - The top feature is aligned with a probe trained on that concept
    - The top feature contributes a significant portion of the total activation probe projection
    """

    model: HookedTransformer
    icl_word_list: list[str]
    max_icl_examples: int | None = None
    base_template: str = "{word}:"
    answer_formatter: Formatter = first_letter_formatter()
    example_separator: str = "\n"
    shuffle_examples: bool = True
    # the position to read activations from (depends on the template)
    word_token_pos: int = -2
    batch_size: int = 10
    topk_feats: int = 10

    # the cosine similarity between the top projecting feature and the probe must be at least this high to count as absorption
    probe_cos_sim_threshold: float = 0.025
    # the probe projection of the top projecting feature must contribute at least this much to the total probe projection to count as absorption
    probe_projection_proportion_threshold: float = 0.4

    @torch.inference_mode()
    def _filter_prompts(
        self,
        prompts: list[SpellingPrompt],
        sae: SAE,
        main_feature_ids: list[int],
    ) -> list[SpellingPrompt]:
        """
        Filter out any prompts where the main features are already active.
        NOTE: All prompts must have the same token length
        """
        self._validate_prompts_are_same_length(prompts)
        results: list[SpellingPrompt] = []
        for batch in batchify(prompts, batch_size=self.batch_size):
            sae_in = self.model.run_with_cache([p.base for p in batch])[1][
                sae.cfg.hook_name
            ]
            sae_acts = sae.encode(sae_in)
            split_feats_active = (
                sae_acts[:, self.word_token_pos, main_feature_ids]
                .sum(dim=-1)
                .float()
                .tolist()
            )
            for prompt, res in zip(batch, split_feats_active):
                if res < EPS:
                    results.append(prompt)
        return results

    def _build_prompts(self, words: list[str]) -> list[SpellingPrompt]:
        return [
            create_icl_prompt(
                word,
                examples=self.icl_word_list,
                base_template=self.base_template,
                answer_formatter=self.answer_formatter,
                example_separator=self.example_separator,
                max_icl_examples=self.max_icl_examples,
                shuffle_examples=self.shuffle_examples,
            )
            for word in words
        ]

    def _is_absorption(
        self,
        probe_projection: float,
        main_feature_scores: list[FeatureScore],
        top_projection_feature_scores: list[FeatureScore],
    ) -> bool:
        # if any of the main features fired, this isn't absorption
        if not all(score.activation < EPS for score in main_feature_scores):
            return False
        # If the top firing feature isn't aligned with the probe, this isn't absorption
        if (
            top_projection_feature_scores[0].probe_cos_sim
            < self.probe_cos_sim_threshold
        ):
            return False
        # If the probe isn't even activated, this can't be absorption
        if probe_projection < 0:
            return False
        # If the top firing feature doesn't contribute much to the total probe projection, this isn't absorption
        proj_proportion = (
            top_projection_feature_scores[0].probe_projection / probe_projection
        )
        if proj_proportion < self.probe_projection_proportion_threshold:
            return False
        return True

    @torch.inference_mode()
    def calculate_absorption(
        self,
        sae: SAE,
        words: list[str],
        probe_direction: torch.Tensor,
        main_feature_ids: list[int],
        layer: int,
        filter_prompts: bool = True,
        show_progress: bool = True,
    ) -> AbsorptionResults:
        """
        This method calculates the absorption for each word in the list of words. If `max_ablation_samples` is provided,
        this method will randomly sample that many words to calculate absorption for as a performance optimization.
        If `filter_prompts` is True, this method will filter out any prompts where the main features are already active, as these cannot be absorption.
        """
        if probe_direction.ndim != 1:
            raise ValueError("probe_direction must be 1D")
        # make sure the probe direction is a unit vector
        probe_direction = probe_direction / probe_direction.norm()
        prompts = self._build_prompts(words)
        if filter_prompts:
            prompts = self._filter_prompts(prompts, sae, main_feature_ids)
        results: list[WordAbsorptionResult] = []
        cos_sims = (
            torch.nn.functional.cosine_similarity(
                probe_direction.to(sae.device), sae.W_dec, dim=-1
            )
            .float()
            .cpu()
        )
        hook_point = f"blocks.{layer}.hook_resid_post"
        for batch_prompts in batchify(prompts, batch_size=self.batch_size):
            batch_acts = self.model.run_with_cache(
                [p.base for p in batch_prompts],
                names_filter=[hook_point],
            )[1][hook_point][:, self.word_token_pos, :]
            batch_sae_acts = sae.encode(batch_acts)
            batch_sae_probe_projections = batch_sae_acts * cos_sims.to(
                batch_sae_acts.device
            )
            batch_probe_projections = batch_acts @ probe_direction.to(
                device=batch_sae_acts.device, dtype=batch_sae_acts.dtype
            )
            for i, prompt in enumerate(tqdm(batch_prompts, disable=not show_progress)):
                sae_acts = batch_sae_acts[i]
                act_probe_proj = batch_probe_projections[i].cpu().item()
                sae_act_probe_proj = batch_sae_probe_projections[i]
                with torch.inference_mode():
                    # sort by negative ig score
                    top_proj_feats = sae_act_probe_proj.topk(
                        self.topk_feats
                    ).indices.tolist()
                    main_feature_scores = _get_feature_scores(
                        main_feature_ids,
                        probe_cos_sims=cos_sims,
                        sae_acts=sae_acts,
                    )
                    top_projection_feature_scores = _get_feature_scores(
                        top_proj_feats,
                        probe_cos_sims=cos_sims,
                        sae_acts=sae_acts,
                    )
                    is_absorption = self._is_absorption(
                        probe_projection=act_probe_proj,
                        top_projection_feature_scores=top_projection_feature_scores,
                        main_feature_scores=main_feature_scores,
                    )
                    results.append(
                        WordAbsorptionResult(
                            word=prompt.word,
                            prompt=prompt.base,
                            probe_projection=act_probe_proj,
                            main_feature_scores=main_feature_scores,
                            top_projection_feature_scores=top_projection_feature_scores,
                            is_absorption=is_absorption,
                        )
                    )
        return AbsorptionResults(
            main_feature_ids=main_feature_ids,
            word_results=results,
        )

    def _validate_prompts_are_same_length(self, prompts: list[SpellingPrompt]):
        "Validate that all prompts have the same token length"
        token_lens = {len(self.model.to_tokens(p.base)[0]) for p in prompts}
        if len(token_lens) > 1:
            raise ValueError(
                "All prompts must have the same token length! Variable-length prompts are not yet supported."
            )


def _get_feature_scores(
    feature_ids: list[int],
    probe_cos_sims: torch.Tensor,
    sae_acts: torch.Tensor,
) -> list[FeatureScore]:
    return [
        FeatureScore(
            feature_id=feature_id,
            probe_cos_sim=probe_cos_sims[feature_id].item(),
            activation=sae_acts[feature_id].item(),
        )
        for feature_id in feature_ids
    ]
