import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Literal, TypeAlias

import torch
from openai import OpenAI
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm

from evals.autointerp.config import AutoInterpConfig
from evals.autointerp.sae_encode import encode_subset
from sae_bench_utils.indexing_utils import get_iw_sample_indices, get_k_largest_indices, index_with_buffer

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


def display_messages(messages: Messages) -> str:
    return tabulate([m.values() for m in messages], tablefmt="simple_grid", maxcolwidths=[None, 120])


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
        model: HookedSAETransformer,
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
            headers=["Top act"] + ([] if predictions is None else ["Active?", "Predicted?"]) + ["Sequence"],
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
        cfg: AutoInterpConfig,
        model: HookedSAETransformer,
        sae: SAE,
        sparsity: Tensor,
        device: str,
        api_key: str,
    ):
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.device = device
        self.api_key = api_key
        self.batch_size = cfg.total_tokens // sae.cfg.context_size
        self.act_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            streaming=True,
            store_batch_size_prompts=self.batch_size,
            device=str(self.device),
        )
        if cfg.latents is not None:
            self.latents = cfg.latents
        else:
            assert self.cfg.n_latents is not None
            alive_latents = torch.nonzero(sparsity > self.cfg.dead_latent_threshold).squeeze(1).tolist()
            assert len(alive_latents) >= self.cfg.n_latents, "Error: not enough alive latents to sample from"
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
            print(f"Found data for {len(latents_with_data)}/{self.n_latents} alive latents; {n_dead} dead")

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
                "correct seqs": [i for i, ex in enumerate(scoring_examples, start=1) if ex.is_active],
                "score": score,
                "logs": results["logs"] + f"\nScoring phase\n{logs}\n{scoring_examples.display(predictions)}",
            }

        return results

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> list[int] | None:
        predictions_split = predictions.strip().rstrip(".").replace("and", ",").replace("None", "").split(",")
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
        return sum([c == cc for c, cc in zip(classifications, correct_classifications)]) / len(classifications)

    def get_api_response(self, messages: Messages, max_tokens: int, n_completions: int = 1) -> tuple[list[str], str]:
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

        examples_as_str = "\n".join([f"{i+1}. {ex.to_str(mark_toks=True)}" for i, ex in enumerate(generation_examples)])

        SYSTEM_PROMPT = """We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words."""
        if self.cfg.use_demos_in_explanation:
            SYSTEM_PROMPT += """ Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy"."""
        else:
            SYSTEM_PROMPT += """Your response should be in the form "This neuron activates on..."."""
        USER_PROMPT = f"""The activating documents are given below:\n\n{examples_as_str}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def get_scoring_prompts(self, explanation: str, scoring_examples: Examples) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join([f"{i+1}. {ex.to_str(mark_toks=False)}" for i, ex in enumerate(scoring_examples)])

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
        # Get all activations, split up into batches
        tokens = self.act_store.get_batch_tokens()
        batch_size, seq_len = tokens.shape
        acts = torch.empty((0, seq_len, self.n_latents), device=self.device)
        for _tokens in tqdm(
            tokens.split(split_size=self.cfg.batch_size, dim=0),
            desc="Forward passes to get activation values",
        ):
            sae_in = self.act_store.get_activations(_tokens).squeeze(2).to(self.device)
            acts = torch.concat([acts, encode_subset(self.sae, sae_in, latents=torch.tensor(self.latents))], dim=0)

        generation_examples = {}
        scoring_examples = {}

        for i, latent in enumerate(self.latents):
            # (1/3) Get random examples (we don't need their values)
            rand_indices = torch.stack(
                [
                    torch.randint(0, batch_size, (self.cfg.n_random_ex_for_scoring,)),
                    torch.randint(
                        self.cfg.buffer,
                        seq_len - self.cfg.buffer,
                        (self.cfg.n_random_ex_for_scoring,),
                    ),
                ],
                dim=-1,
            )
            rand_toks = index_with_buffer(tokens, rand_indices, buffer=self.cfg.buffer)

            # (2/3) Get top-scoring examples
            top_indices = get_k_largest_indices(
                acts[..., i],
                k=self.cfg.n_top_ex,
                buffer=self.cfg.buffer,
                no_overlap=self.cfg.no_overlap,
            )
            top_toks = index_with_buffer(tokens, top_indices, buffer=self.cfg.buffer)
            top_values = index_with_buffer(acts[..., i], top_indices, buffer=self.cfg.buffer)
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()

            # (3/3) Get importance-weighted examples, using a threshold so they're disjoint from top examples
            # Also, if we don't have enough values, then we assume this is a dead feature & continue
            threshold = top_values[:, self.cfg.buffer].min().item()
            acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
            if acts_thresholded[self.cfg.buffer : -self.cfg.buffer].max() < 1e-6:
                continue
            iw_indices = get_iw_sample_indices(acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer)
            iw_toks = index_with_buffer(tokens, iw_indices, buffer=self.cfg.buffer)
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


def run_eval(
    config: AutoInterpConfig,
    selected_saes_dict: dict[str, list[str]],  # dict of SAE release name: list of SAE names to evaluate
    device: str,
    api_key: str,
    save_logs_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Runs autointerp eval. Returns results as a dict with the following structure:

        custom_eval_config - dict of config parameters used for this evaluation
        custom_eval_results - nested dict of {sae_name: {"score": score}}
    """
    results_dict = {}

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    results_dict = {"custom_eval_results": {}, "custom_eval_config": asdict(config)}

    model: HookedSAETransformer = HookedSAETransformer.from_pretrained(config.model_name, device=device)

    for release, sae_names in selected_saes_dict.items():
        saes_map = get_pretrained_saes_directory()[release].saes_map
        for sae_name in sae_names:
            # Load in SAE, and randomly choose a number of latents to use for this autointerp instance
            sae_id = saes_map[sae_name]
            sae, _, sparsity = SAE.from_pretrained(release, sae_id, device=str(device))

            # Get autointerp results
            autointerp = AutoInterp(cfg=config, model=model, sae=sae, sparsity=sparsity, api_key=api_key, device=device)
            results = asyncio.run(autointerp.run())

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
                    [[results[latent][h] for h in headers] for latent in results],
                    headers=headers,
                    tablefmt="simple_outline",
                )
                worst_result = min(results.values(), key=lambda x: x["score"])
                best_result = max(results.values(), key=lambda x: x["score"])
                logs += f"\n\nWorst scoring idx {worst_result['latent']}, score = {worst_result['score']}\n{worst_result['logs']}"
                logs += f"\n\nBest scoring idx {best_result['latent']}, score = {best_result['score']}\n{best_result['logs']}"
                # Save the results to a file
                with open(save_logs_path, "a") as f:
                    f.write(logs)

            # Put important results into the results dict
            score = sum([r["score"] for r in results.values()]) / len(results)
            results_dict["custom_eval_results"][sae_name] = {"score": score}

    return results_dict
