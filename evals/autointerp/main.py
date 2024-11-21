import asyncio
import gc
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Iterator, Literal, TypeAlias, Optional
import os
import time
import argparse
import json
from datetime import datetime

import torch
from openai import OpenAI
from sae_lens import SAE
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer

from evals.autointerp.eval_config import AutoInterpEvalConfig
from evals.autointerp.eval_output import (
    EVAL_TYPE_ID_AUTOINTERP,
    AutoInterpEvalOutput,
    AutoInterpMetricCategories,
    AutoInterpMetrics,
)

from sae_bench_utils.indexing_utils import (
    get_iw_sample_indices,
    get_k_largest_indices,
    index_with_buffer,
)
import sae_bench_utils.dataset_utils as dataset_utils
import sae_bench_utils.activation_collection as activation_collection
import sae_bench_utils.general_utils as general_utils


from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
    select_saes_multiple_patterns,
)

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


def display_messages(messages: Messages) -> str:
    return tabulate(
        [m.values() for m in messages], tablefmt="simple_grid", maxcolwidths=[None, 120]
    )


def str_bool(b: bool) -> str:
    return "Y" if b else ""


class Example:
    """
    Data for a single example sequence.
    """

    def __init__(
        self,
        toks: list[int],
        acts: list[float],
        act_threshold: float,
        model: HookedTransformer,
    ):
        self.toks = toks
        self.str_toks = model.to_str_tokens(torch.tensor(self.toks))
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)  # this is what we predict in the scoring phase

    def to_str(self, mark_toks: bool = False) -> str:
        return (
            "".join(
                f"<<{tok}>>" if (mark_toks and is_active) else tok
                for tok, is_active in zip(self.str_toks, self.toks_are_active)  # type: ignore
            )
            .replace("�", "")
            .replace("\n", "↵")
            # .replace(">><<", "")
        )


class Examples:
    """
    Data for multiple example sequences. Includes methods for shuffling seuqences, and displaying them.
    """

    def __init__(self, examples: list[Example], shuffle: bool = False) -> None:
        self.examples = examples
        if shuffle:
            random.shuffle(self.examples)
        else:
            self.examples = sorted(self.examples, key=lambda x: max(x.acts), reverse=True)

    def display(self, predictions: list[int] | None = None) -> str:
        """
        Displays the list of sequences. If `predictions` is provided, then it'll include a column for both "is_active"
        and these predictions of whether it's active. If not, then neither of those columns will be included.
        """
        return tabulate(
            [
                [max(ex.acts), ex.to_str(mark_toks=True)]
                if predictions is None
                else [
                    max(ex.acts),
                    str_bool(ex.is_active),
                    str_bool(i + 1 in predictions),
                    ex.to_str(mark_toks=False),
                ]
                for i, ex in enumerate(self.examples)
            ],
            headers=["Top act"]
            + ([] if predictions is None else ["Active?", "Predicted?"])
            + ["Sequence"],
            tablefmt="simple_outline",
            floatfmt=".3f",
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


class AutoInterp:
    """
    This is a start-to-end class for generating explanations and optionally scores. It's easiest to implement it as a
    single class for the time being because there's data we'll need to fetch that'll be used in both the generation and
    scoring phases.
    """

    def __init__(
        self,
        cfg: AutoInterpEvalConfig,
        model: HookedTransformer,
        sae: SAE,
        tokenized_dataset: Tensor,
        sparsity: Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.tokenized_dataset = tokenized_dataset
        self.device = device
        self.api_key = api_key
        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            assert self.cfg.n_latents is not None
            sparsity *= cfg.total_tokens
            alive_latents = (
                torch.nonzero(sparsity > self.cfg.dead_latent_threshold).squeeze(1).tolist()
            )
            if len(alive_latents) < self.cfg.n_latents:
                self.latents = alive_latents
                print(
                    f"\n\n\nWARNING: Found only {len(alive_latents)} alive latents, which is less than {self.cfg.n_latents}\n\n\n"
                )
            else:
                self.latents = random.sample(alive_latents, k=self.cfg.n_latents)
        self.n_latents = len(self.latents)

    async def run(self, explanations_override: dict[int, str] = {}) -> dict[int, dict[str, Any]]:
        """
        Runs both generation & scoring phases. Returns a dict where keys are latent indices, and values are dicts with:

            "explanation": str, the explanation generated for this latent
            "predictions": list[int], the predicted activating indices
            "correct seqs": list[int], the true activating indices
            "score": float, the fraction of correct predictions (including positive and negative)
            "logs": str, the logs for this latent
        """
        generation_examples, scoring_examples = self.gather_data()
        latents_with_data = sorted(generation_examples.keys())
        n_dead = self.n_latents - len(latents_with_data)
        if n_dead > 0:
            print(
                f"Found data for {len(latents_with_data)}/{self.n_latents} alive latents; {n_dead} dead"
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            tasks = [
                self.run_single_feature(
                    executor,
                    latent,
                    generation_examples[latent],
                    scoring_examples[latent],
                    explanations_override.get(latent, None),
                )
                for latent in latents_with_data
            ]
            results = {}
            for future in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Calling API (for gen & scoring)",
            ):
                result = await future
                if result:
                    results[result["latent"]] = result

        return results

    async def run_single_feature(
        self,
        executor: ThreadPoolExecutor,
        latent: int,
        generation_examples: Examples,
        scoring_examples: Examples,
        explanation_override: str | None = None,
    ) -> dict[str, Any] | None:
        # Generation phase
        gen_prompts = self.get_generation_prompts(generation_examples)
        (explanation_raw,), logs = await asyncio.get_event_loop().run_in_executor(
            executor,
            self.get_api_response,
            gen_prompts,
            self.cfg.max_tokens_in_explanation,
        )
        explanation = self.parse_explanation(explanation_raw)
        results = {
            "latent": latent,
            "explanation": explanation,
            "logs": f"Generation phase\n{logs}\n{generation_examples.display()}",
        }

        # Scoring phase
        if self.cfg.scoring:
            scoring_prompts = self.get_scoring_prompts(
                explanation=explanation_override or explanation,
                scoring_examples=scoring_examples,
            )
            (predictions_raw,), logs = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.get_api_response,
                scoring_prompts,
                self.cfg.max_tokens_in_prediction,
            )
            predictions = self.parse_predictions(predictions_raw)
            if predictions is None:
                return None
            score = self.score_predictions(predictions, scoring_examples)
            results |= {
                "predictions": predictions,
                "correct seqs": [
                    i for i, ex in enumerate(scoring_examples, start=1) if ex.is_active
                ],
                "score": score,
                "logs": results["logs"]
                + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
            }

        return results

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> list[int] | None:
        predictions_split = (
            predictions.strip().rstrip(".").replace("and", ",").replace("None", "").split(",")
        )
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if predictions_list == []:
            return []
        if not all(pred.strip().isdigit() for pred in predictions_list):
            return None
        predictions_ints = [int(pred.strip()) for pred in predictions_list]
        return predictions_ints

    def score_predictions(self, predictions: list[int], scoring_examples: Examples) -> float:
        classifications = [i in predictions for i in range(1, len(scoring_examples) + 1)]
        correct_classifications = [ex.is_active for ex in scoring_examples]
        return sum([c == cc for c, cc in zip(classifications, correct_classifications)]) / len(
            classifications
        )

    def get_api_response(
        self, messages: Messages, max_tokens: int, n_completions: int = 1
    ) -> tuple[list[str], str]:
        """Generic API usage function for OpenAI"""
        for message in messages:
            assert message.keys() == {"content", "role"}
            assert message["role"] in ["system", "user", "assistant"]

        client = OpenAI(api_key=self.api_key)

        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # type: ignore
            n=n_completions,
            max_tokens=max_tokens,
            stream=False,
        )
        response = [choice.message.content.strip() for choice in result.choices]

        logs = tabulate(
            [m.values() for m in messages + [{"role": "assistant", "content": response[0]}]],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )

        return response, logs

    def get_generation_prompts(self, generation_examples: Examples) -> Messages:
        assert len(generation_examples) > 0, "No generation examples found"

        examples_as_str = "\n".join(
            [f"{i+1}. {ex.to_str(mark_toks=True)}" for i, ex in enumerate(generation_examples)]
        )

        SYSTEM_PROMPT = """We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words."""
        if self.cfg.use_demos_in_explanation:
            SYSTEM_PROMPT += """ Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy"."""
        else:
            SYSTEM_PROMPT += (
                """Your response should be in the form "This neuron activates on..."."""
            )
        USER_PROMPT = f"""The activating documents are given below:\n\n{examples_as_str}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def get_scoring_prompts(self, explanation: str, scoring_examples: Examples) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join(
            [f"{i+1}. {ex.to_str(mark_toks=False)}" for i, ex in enumerate(scoring_examples)]
        )

        example_response = sorted(
            random.sample(
                range(1, 1 + self.cfg.n_ex_for_scoring),
                k=self.cfg.n_correct_for_scoring,
            )
        )
        example_response_str = ", ".join([str(i) for i in example_response])
        SYSTEM_PROMPT = f"""We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown {self.cfg.n_ex_for_scoring} example sequences in random order. You will have to return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like "{example_response_str}". Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with "None". You should include nothing else in your response other than comma-separated numbers or the word "None" - this is important."""
        USER_PROMPT = f"Here is the explanation: this neuron fires on {explanation}.\n\nHere are the examples:\n\n{examples_as_str}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def gather_data(self) -> tuple[dict[int, Examples], dict[int, Examples]]:
        """
        Stores top acts / random seqs data, which is used for generation & scoring respectively.
        """
        dataset_size, seq_len = self.tokenized_dataset.shape

        acts = activation_collection.collect_sae_activations(
            self.tokenized_dataset,
            self.model,
            self.sae,
            self.cfg.llm_batch_size,
            self.sae.cfg.hook_layer,
            self.sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
            selected_latents=self.latents,
            activation_dtype=torch.bfloat16,  # reduce memory usage, we don't need full precision when sampling activations
        )

        generation_examples = {}
        scoring_examples = {}

        for i, latent in tqdm(enumerate(self.latents), desc="Collecting examples for LLM judge"):
            # (1/3) Get random examples (we don't need their values)
            rand_indices = torch.stack(
                [
                    torch.randint(0, dataset_size, (self.cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(
                self.tokenized_dataset, rand_indices, buffer=self.cfg.buffer
            )

            # (2/3) Get top-scoring examples
            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(
                self.tokenized_dataset, top_indices, buffer=self.cfg.buffer
            )
            top_values = index_with_buffer(acts[..., i], top_indices, buffer=self.cfg.buffer)
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Get importance-weighted examples, using a threshold so they're disjoint from top examples
            # Also, if we don't have enough values, then we assume this is a dead feature & continue
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[:, self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = get_iw_sample_indices(
                acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
            )
            iw_toks = index_with_buffer(self.tokenized_dataset, iw_indices, buffer=self.cfg.buffer)
            iw_values = index_with_buffer(acts[..., i], iw_indices, buffer=self.cfg.buffer)

            # Get random values to use for splitting
            rand_top_ex_split_indices = torch.randperm(self.cfg.n_top_ex)
            top_gen_indices = rand_top_ex_split_indices[: self.cfg.n_top_ex_for_generation]
            top_scoring_indices = rand_top_ex_split_indices[self.cfg.n_top_ex_for_generation :]
            rand_iw_split_indices = torch.randperm(self.cfg.n_iw_sampled_ex)
            iw_gen_indices = rand_iw_split_indices[: self.cfg.n_iw_sampled_ex_for_generation]
            iw_scoring_indices = rand_iw_split_indices[self.cfg.n_iw_sampled_ex_for_generation :]

            def create_examples(all_toks: Tensor, all_acts: Tensor | None = None) -> list[Example]:
                if all_acts is None:
                    all_acts = torch.zeros_like(all_toks).float()
                return [
                    Example(
                        toks=toks,
                        acts=acts,
                        act_threshold=act_threshold,
                        model=self.model,
                    )
                    for (toks, acts) in zip(all_toks.tolist(), all_acts.tolist())
                ]

            # Get the generation & scoring examples
            generation_examples[latent] = Examples(
                create_examples(top_toks[top_gen_indices], top_values[top_gen_indices])
                + create_examples(iw_toks[iw_gen_indices], iw_values[iw_gen_indices]),
            )
            scoring_examples[latent] = Examples(
                create_examples(top_toks[top_scoring_indices], top_values[top_scoring_indices])
                + create_examples(iw_toks[iw_scoring_indices], iw_values[iw_scoring_indices])
                + create_examples(rand_toks),
                shuffle=True,
            )

        return generation_examples, scoring_examples


def run_eval_single_sae(
    config: AutoInterpEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    artifacts_folder: str,
    api_key: str,
    sae_sparsity: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.set_grad_enabled(False)

    os.makedirs(artifacts_folder, exist_ok=True)

    tokens_filename = (
        f"{config.model_name}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    )
    tokens_path = os.path.join(artifacts_folder, tokens_filename)

    if os.path.exists(tokens_path):
        tokenized_dataset = torch.load(tokens_path).to(device)
    else:
        tokenized_dataset = dataset_utils.load_and_tokenize_dataset(
            config.dataset_name, config.llm_context_size, config.total_tokens, model.tokenizer
        ).to(device)
        torch.save(tokenized_dataset, tokens_path)

    print(f"Loaded tokenized dataset of shape {tokenized_dataset.shape}")

    if sae_sparsity is None:
        sae_sparsity = activation_collection.get_feature_activation_sparsity(
            tokenized_dataset,
            model,
            sae,
            config.llm_batch_size,
            sae.cfg.hook_layer,
            sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )

    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        tokenized_dataset=tokenized_dataset,
        sparsity=sae_sparsity,
        api_key=api_key,
        device=device,
    )
    results = asyncio.run(autointerp.run())
    return results


def run_eval(
    config: AutoInterpEvalConfig,
    selected_saes: list[tuple[str, str]] | list[tuple[str, SAE]],
    device: str,
    api_key: str,
    output_path: str,
    force_rerun: bool = False,
    save_logs_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)
    """
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    artifacts_base_folder = "artifacts"
    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    model: HookedTransformer = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for sae_release, sae_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        gc.collect()
        torch.cuda.empty_cache()

        # Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)
        if isinstance(sae_id, str):
            sae, _, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )
        else:
            sae = sae_id
            sae_id = "custom_sae"
            sparsity = None

        sae = sae.to(device=device, dtype=llm_dtype)

        artifacts_folder = os.path.join(artifacts_base_folder, EVAL_TYPE_ID_AUTOINTERP)

        sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        sae_result_file = sae_result_file.replace("/", "_")
        sae_result_path = os.path.join(output_path, sae_result_file)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Loading existing results from {sae_result_path}")
            with open(sae_result_path, "r") as f:
                eval_output = json.load(f)
        else:
            sae_eval_result = run_eval_single_sae(
                config, sae, model, device, artifacts_folder, api_key, sparsity
            )

            # Save nicely formatted logs to a text file, helpful for debugging.
            if save_logs_path is not None:
                # Get summary results for all latents, as well logs for the best and worst-scoring latents
                headers = [
                    "latent",
                    "explanation",
                    "predictions",
                    "correct seqs",
                    "score",
                ]
                logs = "Summary table:\n" + tabulate(
                    [[sae_eval_result[latent][h] for h in headers] for latent in sae_eval_result],
                    headers=headers,
                    tablefmt="simple_outline",
                )
                worst_result = min(sae_eval_result.values(), key=lambda x: x["score"])
                best_result = max(sae_eval_result.values(), key=lambda x: x["score"])
                logs += f"\n\nWorst scoring idx {worst_result['latent']}, score = {worst_result['score']}\n{worst_result['logs']}"
                logs += f"\n\nBest scoring idx {best_result['latent']}, score = {best_result['score']}\n{best_result['logs']}"
                # Save the results to a file
                with open(save_logs_path, "a") as f:
                    f.write(logs)

            # Put important results into the results dict
            score = sum([r["score"] for r in sae_eval_result.values()]) / len(sae_eval_result)

            eval_output = AutoInterpEvalOutput(
                eval_config=config,
                eval_id=eval_instance_id,
                datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                eval_result_metrics=AutoInterpMetricCategories(
                    autointerp=AutoInterpMetrics(autointerp_score=score)
                ),
                eval_result_details=[],
                eval_result_unstructured=sae_eval_result,
                sae_bench_commit_hash=sae_bench_commit_hash,
                sae_lens_id=sae_id,
                sae_lens_release_id=sae_release,
                sae_lens_version=sae_lens_version,
            )

            results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

            eval_output.to_json_file(sae_result_path, indent=2)

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[AutoInterpEvalConfig, list[tuple[str, str]]]:
    config = AutoInterpEvalConfig(
        model_name=args.model_name,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run auto interp evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/autointerp",
        help="Output folder",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )

    return parser


if __name__ == "__main__":
    """
    python evals/autointerp/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped \
    --api_key <API_KEY>

    python evals/autointerp/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_20/width_16k/average_l0_139" \
    --model_name gemma-2-2b \
    --api_key <API_KEY>

    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.api_key,
        args.output_folder,
        args.force_rerun,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     """
#     python evals/autointerp/main.py
#     NOTE: We don't use argparse here. This requires a file openai_api_key.txt to be present in the root directory.
#     """

#     import custom_saes.identity_sae as identity_sae
#     import custom_saes.jumprelu_sae as jumprelu_sae

#     device = general_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results/autointerp"

#     with open("openai_api_key.txt", "r") as f:
#         api_key = f.read().strip()

#     model_name = "gemma-2-2b"
#     hook_layer = 20

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     config = AutoInterpEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#     )

#     config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
#     config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

#     # create output folder
#     os.makedirs(output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         api_key,
#         output_folder,
#         force_rerun=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")
