import torch
import numpy as np
import pandas as pd
import re
import os
import time
import pickle
import os
from transformer_lens import HookedTransformer
from sae_lens import SAE
import itertools
from itertools import permutations
import torch.nn.functional as F
import gc
import json
from tqdm import tqdm
from datasets import load_dataset
from functools import partial
from jaxtyping import Float
from typing import Any, Optional
import requests
from requests.exceptions import HTTPError
import time


from evals.unlearning.utils.var import (
    GEMMA_INST_FORMAT,
    MIXTRAL_INST_FORMAT,
    PRE_WMDP_BIO,
    PRE_QUESTION_FORMAT,
)
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features

all_permutations = list(permutations([0, 1, 2, 3]))


def load_dataset_with_retries(
    dataset_path: str, dataset_name: str, split: str, retries: int = 5, delay: int = 20
):
    """
    Tries to load the dataset with a specified number of retries and delay between attempts.

    Raises:
    - HTTPError: If the dataset cannot be loaded after the given number of retries.
    """
    for attempt in range(retries):
        try:
            dataset = load_dataset(dataset_path, dataset_name, split=split)
            return dataset  # Successful load
        except HTTPError as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                print(f"Failed to load dataset after {retries} attempts.")
                raise


def calculate_MCQ_metrics(
    model: HookedTransformer,
    mcq_batch_size: int,
    dataset_name: str = "wmdp-bio",
    target_metric: Optional[str] = None,
    question_subset: Optional[list[int]] = None,
    question_subset_file: Optional[str] = None,
    permutations: list[list[int]] = [[0, 1, 2, 3]],
    verbose: bool = True,
    without_question: bool = False,
    prompt_format: Optional[str] = None,
    split: str = "all",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Calculate metrics for a multiple-choice question (MCQ) dataset using a given model.

    Parameters:
    ----------
    model : HookedTransformer
    dataset_name : str, default='wmdp-bio' - Or the dataset_name of MMLU
    target_metric : Optional[str] - Name of the metric used to select a subset of questions
    question_subset : Optional[List[int]] - A list of indices specifying the subset of questions to be used
    question_subset_file : Optional[str] - Path to a file containing the indices for a subset of the questions to be used. Overrides question_subset if provided
    permutations : List[List[int]], default=[[0, 1, 2, 3]] - List of permutations to be applied to the question indices
    verbose : bool, default=True
    without_question : bool, default=False - Evaluate the model without instruction and question if True
    prompt_format : Optional[str] - The format of the prompt to be used. Can be None, 'GEMMA_INST_FORMAT' or 'MIXTRAL_INST_FORMAT'
    split : str, default='all'
    **kwargs : Any - Additional arguments

    Returns:
    -------
    metrics : Dict[str, Any] - A dictionary containing the calculated metrics for the dataset.
    """

    metrics = {}

    # Load dataset
    assert isinstance(dataset_name, str)
    if dataset_name == "wmdp-bio":
        pre_question = PRE_WMDP_BIO
        dataset = load_dataset_with_retries("cais/wmdp", "wmdp-bio", split="test")
    else:
        pre_question = PRE_QUESTION_FORMAT.format(subject=dataset_name.replace("_", " "))
        # pre_question = 'The following are multiple choice questions (with answers) about history'
        dataset = load_dataset_with_retries("cais/mmlu", dataset_name, split="test")

    answers = [x["answer"] for x in dataset]
    questions = [x["question"] for x in dataset]
    choices_list = [x["choices"] for x in dataset]

    # Select subset of questions
    assert target_metric in [
        None,
        "correct",
        "correct-iff-question",
        "correct_no_tricks",
        "all",
    ], "target_metric not recognised"
    assert split in ["all", "train", "test"], "split not recognised"
    if target_metric is not None:
        model_name = model.cfg.model_name
        full_dataset_name = (
            f'mmlu-{dataset_name.replace("_", "-")}' if dataset_name != "wmdp-bio" else dataset_name
        )
        question_subset_file = (
            f"data/question_ids/{model_name}/{split}/{full_dataset_name}_{target_metric}.csv"
        )

    if question_subset_file is not None:
        question_subset = np.genfromtxt(question_subset_file, ndmin=1, dtype=int)

    # Only keep desired subset of questions
    if question_subset is not None:
        answers = [answers[i] for i in question_subset if i < len(answers)]
        questions = [questions[i] for i in question_subset if i < len(questions)]
        choices_list = [choices_list[i] for i in question_subset if i < len(choices_list)]

    # changing prompt_format
    if model.cfg.model_name in ["gemma-2-9b-it", "gemma-2-2b-it"]:
        prompt_format = "GEMMA_INST_FORMAT"
    else:
        raise Exception("Model prompt format not found.")

    if permutations is None:
        prompts = [
            convert_wmdp_data_to_prompt(
                question,
                choices,
                prompt_format=prompt_format,
                without_question=without_question,
                pre_question=pre_question,
            )
            for question, choices in zip(questions, choices_list)
        ]
    else:
        prompts = [
            [
                convert_wmdp_data_to_prompt(
                    question,
                    choices,
                    prompt_format=prompt_format,
                    permute_choices=p,
                    without_question=without_question,
                    pre_question=pre_question,
                )
                for p in permutations
            ]
            for question, choices in zip(questions, choices_list)
        ]
        prompts = [item for sublist in prompts for item in sublist]

        answers = [[p.index(answer) for p in permutations] for answer in answers]
        answers = [item for sublist in answers for item in sublist]

    actual_answers = answers

    batch_size = np.minimum(len(prompts), mcq_batch_size)
    n_batches = len(prompts) // batch_size

    if len(prompts) > batch_size * n_batches:
        n_batches = n_batches + 1

    if isinstance(model, HookedTransformer):
        output_probs = get_output_probs_abcd(
            model, prompts, batch_size=batch_size, n_batches=n_batches, verbose=verbose
        )
    else:
        output_probs = get_output_probs_abcd_hf(
            model,
            model.tokenizer,
            prompts,
            batch_size=batch_size,
            n_batches=n_batches,
            verbose=verbose,
        )

    predicted_answers = output_probs.argmax(dim=1)
    predicted_probs = output_probs.max(dim=1)[0]

    n_predicted_answers = len(predicted_answers)

    actual_answers = torch.tensor(actual_answers)[:n_predicted_answers].to("cuda")

    predicted_prob_of_correct_answers = output_probs[
        torch.arange(len(actual_answers)), actual_answers
    ]

    is_correct = (actual_answers == predicted_answers).to(torch.float)
    mean_correct = is_correct.mean()

    metrics["mean_correct"] = float(mean_correct.item())
    metrics["total_correct"] = int(np.sum(is_correct.cpu().numpy()))
    metrics["is_correct"] = is_correct.cpu().numpy()

    metrics["output_probs"] = output_probs.to(torch.float16).cpu().numpy()
    # metrics['actual_answers'] = actual_answers.cpu().numpy()

    # metrics['predicted_answers'] = predicted_answers.cpu().numpy()
    # metrics['predicted_probs'] = predicted_probs.to(torch.float16).cpu().numpy()
    # metrics['predicted_probs_of_correct_answers'] = predicted_prob_of_correct_answers.to(torch.float16).cpu().numpy()
    # metrics['mean_predicted_prob_of_correct_answers'] = float(np.mean(predicted_prob_of_correct_answers.to(torch.float16).cpu().numpy()))
    # metrics['mean_predicted_probs'] = float(np.mean(predicted_probs.to(torch.float16).cpu().numpy()))

    # unique, counts = np.unique(metrics['predicted_answers'], return_counts=True)
    # metrics['value_counts'] = dict(zip([int(x) for x in unique], [int(x) for x in counts]))

    # metrics['sum_abcd'] = metrics['output_probs'].sum(axis=1)

    return metrics


def get_output_probs_abcd(model, prompts, batch_size=2, n_batches=100, verbose=True):
    """
    Calculates probability of selecting A, B, C, & D for a given input prompt
    and language model. Returns tensor of shape (len(prompts), 4).
    """

    spaces_and_single_models = [
        "gemma-2b-it",
        "gemma-2b",
        "gemma-2-9b",
        "gemma-2-9b-it",
        "gemma-2-2b-it",
        "gemma-2-2b",
    ]
    if model.cfg.model_name in spaces_and_single_models:
        answer_strings = ["A", "B", "C", "D", " A", " B", " C", " D"]
    elif model.cfg.model_name in ["Mistral-7B-v0.1"]:
        answer_strings = ["A", "B", "C", "D"]
    else:
        raise Exception("Model name not hardcoded in this function.")

    answer_tokens = model.to_tokens(answer_strings, prepend_bos=False).flatten()

    # batch_size = 1

    with torch.no_grad():
        output_probs = []

        for i in tqdm(range(n_batches), disable=not verbose):
            prompt_batch = prompts[i * batch_size : i * batch_size + batch_size]
            current_batch_size = len(prompt_batch)

            token_batch = model.to_tokens(prompt_batch, padding_side="right").to("cuda")
            token_lens = [len(model.to_tokens(x)[0]) for x in prompt_batch]
            next_token_indices = torch.tensor([x - 1 for x in token_lens]).to("cuda")

            vals = model(token_batch, return_type="logits")
            vals = vals[torch.arange(current_batch_size).to("cuda"), next_token_indices].softmax(-1)
            # vals = torch.vstack([x[i] for x, i in zip(vals, next_token_indices)]).softmax(-1)
            # vals = vals[0, -1].softmax(-1)
            vals = vals[:, answer_tokens]
            if model.cfg.model_name in spaces_and_single_models:
                vals = vals.reshape(-1, 2, 4).max(dim=1)[0]
            output_probs.append(vals)

        output_probs = torch.vstack(output_probs)

    return output_probs


def convert_wmdp_data_to_prompt(
    question,
    choices,
    prompt_format=None,
    pre_question=PRE_WMDP_BIO,
    permute_choices=None,
    without_question=False,
):
    """
    Takes in the question and choices for WMDP data and converts it to a prompt,
    including a pre-question prompt, question, answers with A, B, C & D, followed
    by "Answer:"

    datapoint: datapoint containing question and choices
    prompt_format: can be None (default), GEMMA_INST_FORMAT or MIXTRAL_INST_FORMAT
    """

    pre_answers = ["A. ", "B. ", "C. ", "D. "]
    pre_answers = ["\n" + x for x in pre_answers]
    post_answers = "\nAnswer:"

    if permute_choices is not None:
        choices = [choices[i] for i in permute_choices]

    answers = r"".join([item for pair in zip(pre_answers, choices) for item in pair])

    if prompt_format is None:
        if without_question:
            prompt = r"".join([answers, post_answers])[1:]  # slice it to remove the '\n'
        else:
            prompt = r"".join([pre_question, question, answers, post_answers])

    elif prompt_format == "GEMMA_INST_FORMAT":
        if without_question:
            prompt = answers[1:]  # slice it to remove the '\n'
        else:
            prompt = r"".join([pre_question, question, answers])

        prompt = GEMMA_INST_FORMAT.format(prompt=prompt)
        prompt = prompt + "Answer: ("

    elif prompt_format == "MIXTRAL_INST_FORMAT":
        if without_question:
            prompt = answers[1:]  # slice it to remove the '\n'
        else:
            prompt = r"".join([pre_question, question, answers, post_answers])
        prompt = MIXTRAL_INST_FORMAT.format(prompt=prompt)
        # prompt = prompt + "Answer:"

    else:
        raise Exception("Prompt format not recognised.")

    return prompt


def get_per_token_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]
    return -predicted_log_probs


def get_output_probs_abcd_hf(model, tokenizer, prompts, batch_size=1, n_batches=100, verbose=True):
    spaces_and_single_models = ["gemma-2b-it", "gemma-2b"]
    # answer_strings = ["A", "B", "C", "D"]
    answer_strings = [" A", " B", " C", " D"]
    istart = 0

    # answer_tokens = model.to_tokens(answer_strings, prepend_bos=False).flatten()
    answer_tokens = torch.tensor([tokenizer(x)["input_ids"][1:] for x in answer_strings]).to("cuda")

    with torch.no_grad():
        output_probs = []

        for i in tqdm(range(n_batches), disable=not verbose):
            prompt_batch = prompts[i * batch_size : i * batch_size + batch_size]
            current_batch_size = len(prompt_batch)
            token_batch = [
                torch.tensor(tokenizer(x)["input_ids"][istart:]).to("cuda") for x in prompt_batch
            ]
            next_token_indices = torch.tensor([len(x) - 1 for x in token_batch]).to("cuda")
            max_len = np.max([len(x) for x in token_batch])
            token_batch = [
                torch.concatenate(
                    (x, torch.full((max_len - len(x),), tokenizer.pad_token_id).to("cuda"))
                )
                for x in token_batch
            ]
            token_batch = torch.vstack(token_batch)

            logits = model(token_batch).logits
            vals = logits[torch.arange(current_batch_size), next_token_indices]
            vals = vals.softmax(-1)[:, answer_tokens]

            # if model.cfg.model_name in spaces_and_single_models:
            # vals = vals.reshape(-1, 2, 4).max(dim=1)[0]
            output_probs.append(vals)

        output_probs = torch.vstack(output_probs)
    return output_probs[:, :, 0]


def modify_model(model, sae, **ablate_params):
    model.reset_hooks()

    # Select intervention function
    if ablate_params["intervention_method"] == "scale_feature_activation":
        # ablation_method = anthropic_remove_resid_SAE_features
        raise NotImplementedError
    elif ablate_params["intervention_method"] == "remove_from_residual_stream":
        # ablation_method = remove_resid_SAE_features
        raise NotImplementedError
    elif ablate_params["intervention_method"] == "clamp_feature_activation":
        ablation_method = anthropic_clamp_resid_SAE_features
    elif ablate_params["intervention_method"] == "clamp_feature_activation_jump":
        # ablation_method = anthropic_clamp_jump_relu_resid_SAE_features
        raise NotImplementedError
    elif ablate_params["intervention_method"] == "clamp_feature_activation_random":
        # ablation_method = partial(anthropic_clamp_resid_SAE_features, random=True)
        raise NotImplementedError

    # Hook function
    features_to_ablate = ablate_params["features_to_ablate"]

    if (
        isinstance(ablate_params["features_to_ablate"], int)
        or isinstance(features_to_ablate, np.int64)
        or isinstance(features_to_ablate, np.float64)
    ):
        features_to_ablate = [ablate_params["features_to_ablate"]]
        ablate_params["features_to_ablate"] = features_to_ablate

    hook_params = dict(ablate_params)
    del hook_params["intervention_method"]

    ablate_hook_func = partial(ablation_method, sae=sae, **hook_params)
    # features_to_ablate=features_to_ablate,
    # multiplier=ablate_params['multiplier']
    # )

    # Hook point
    if "custom_hook_point" not in ablate_params or ablate_params["custom_hook_point"] is None:
        hook_point = sae.cfg.hook_name
    else:
        hook_point = ablate_params["custom_hook_point"]

    model.add_hook(hook_point, ablate_hook_func)


def compute_loss_added(
    model, sae, activation_store, n_batch=2, split="all", verbose=False, **ablate_params
):
    """
    Computes loss added for model and SAE intervention
    """

    activation_store.iterable_dataset = iter(activation_store.dataset)

    # only take even batches for train and odd batches for test
    if split in ["train", "test"]:
        n_batch *= 2

    with torch.no_grad():
        loss_diffs = []

        for i in tqdm(range(n_batch), disable=not verbose):
            tokens = activation_store.get_batch_tokenized_data()

            # skip the irrelevant batch
            if split == "train" and i % 2 == 0:
                continue
            elif split == "test" and i % 2 == 1:
                continue

            # Compute baseline loss
            model.reset_hooks()
            baseline_loss = model(tokens, return_type="loss")

            gc.collect()
            torch.cuda.empty_cache()

            # Calculate modified loss
            model.reset_hooks()
            modify_model(model, sae, **ablate_params)
            modified_loss = model(tokens, return_type="loss")

            gc.collect()
            torch.cuda.empty_cache()

            model.reset_hooks()

            loss_diff = modified_loss.item() - baseline_loss.item()
            loss_diffs.append(loss_diff)

        return np.mean(loss_diffs)


def get_baseline_metrics(
    model: HookedTransformer,
    mcq_batch_size: int,
    dataset_name,
    metric_param,
    recompute=False,
    split="all",
    output_dir="./data/baseline_metrics",
):
    """
    Compute the baseline metrics or retrieve if pre-computed and saved
    """

    model.reset_hooks()

    full_dataset_name = (
        f'mmlu-{dataset_name.replace("_", "-")}' if dataset_name != "wmdp-bio" else dataset_name
    )
    model_name = model.cfg.model_name
    q_type = metric_param["target_metric"]

    baseline_metrics_file = os.path.join(
        output_dir, f"{model_name}/{split}/{full_dataset_name}_{q_type}.json"
    )
    os.makedirs(os.path.dirname(baseline_metrics_file), exist_ok=True)

    if not recompute and os.path.exists(baseline_metrics_file):
        # Load the json
        with open(baseline_metrics_file, "r") as f:
            baseline_metrics = json.load(f)

        # Convert lists to arrays for ease of use
        for key, value in baseline_metrics.items():
            if isinstance(value, list):
                baseline_metrics[key] = np.array(value)

        return baseline_metrics

    else:
        baseline_metrics = calculate_MCQ_metrics(
            model, mcq_batch_size, dataset_name=dataset_name, split=split, **metric_param
        )

        metrics = baseline_metrics.copy()

        # Convert lists to arrays for ease of use
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()

        with open(baseline_metrics_file, "w") as f:
            json.dump(metrics, f)

        return baseline_metrics


def modify_and_calculate_metrics(
    model: HookedTransformer,
    mcq_batch_size: int,
    sae: SAE,
    dataset_names=["wmdp-bio"],
    metric_params={"wmdp-bio": {"target_metric": "correct"}},
    n_batch_loss_added=2,
    activation_store=None,
    split="all",
    verbose=False,
    **ablate_params,
):
    metrics_for_current_ablation = {}

    if "loss_added" in dataset_names:
        loss_added = compute_loss_added(
            model,
            sae,
            activation_store,
            n_batch=n_batch_loss_added,
            split=split,
            verbose=verbose,
            **ablate_params,
        )

        metrics_for_current_ablation["loss_added"] = loss_added
        dataset_names = [x for x in dataset_names if x != "loss_added"]

    model.reset_hooks()
    modify_model(model, sae, **ablate_params)

    for dataset_name in dataset_names:
        if dataset_name in metric_params:
            metric_param = metric_params[dataset_name]
        else:
            metric_param = {"target_metric": "correct", "verbose": verbose}

        dataset_metrics = calculate_MCQ_metrics(
            model, mcq_batch_size, dataset_name=dataset_name, split=split, **metric_param
        )
        metrics_for_current_ablation[dataset_name] = dataset_metrics

    model.reset_hooks()

    return metrics_for_current_ablation


def generate_ablate_params_list(main_ablate_params, sweep):
    combinations = [
        dict(zip(sweep.keys(), values)) for values in itertools.product(*sweep.values())
    ]

    cfg_list = []
    for combo in combinations:
        specific_inputs = main_ablate_params.copy()
        specific_inputs.update(combo)
        cfg_list.append(specific_inputs)
    return cfg_list


def calculate_metrics_list(
    model: HookedTransformer,
    mcq_batch_size: int,
    sae: SAE,
    main_ablate_params,
    sweep,
    dataset_names=["wmdp-bio"],
    metric_params={"wmdp-bio": {"target_metric": "correct"}},
    n_batch_loss_added=2,
    activation_store=None,
    split="all",
    target_metric="correct",
    verbose=False,
    save_metrics=False,
    save_metrics_dir=None,
    retain_threshold=None,
):
    """
    Calculate metrics for combinations of ablations
    """

    metrics_list = []

    # First get baseline metrics and ensure that target question ids exist
    baseline_metrics = {}

    for dataset_name in [x for x in dataset_names if x != "loss_added"]:
        # Ensure that target question ids exist
        save_target_question_ids(
            model, mcq_batch_size, dataset_name, output_dir="./data/question_ids"
        )

        if dataset_name in metric_params:
            metric_param = metric_params[dataset_name]
        else:
            metric_param = {"target_metric": target_metric, "verbose": False}

        # metrics[dataset_name] = dataset_metrics

        baseline_metric = get_baseline_metrics(
            model, mcq_batch_size, dataset_name, metric_param, split=split
        )

        baseline_metrics[dataset_name] = baseline_metric

    if "loss_added" in dataset_names:
        baseline_metrics["loss_added"] = 0

    metrics_list.append(baseline_metrics)

    # Now do all ablatation combinations and get metrics each time
    ablate_params_list = generate_ablate_params_list(main_ablate_params, sweep)

    for ablate_params in tqdm(ablate_params_list):
        # check if metrics already exist
        intervention_method = ablate_params["intervention_method"]
        multiplier = ablate_params["multiplier"]
        n_features = len(ablate_params["features_to_ablate"])
        layer = sae.cfg.hook_layer

        save_file_name = f"{intervention_method}_multiplier{multiplier}_nfeatures{n_features}_layer{layer}_retainthres{retain_threshold}.pkl"
        full_path = os.path.join(save_metrics_dir, save_file_name)

        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                ablated_metrics = pickle.load(f)
            metrics_list.append(ablated_metrics)
            continue

        ablated_metrics = modify_and_calculate_metrics(
            model,
            mcq_batch_size,
            sae,
            dataset_names=dataset_names,
            metric_params=metric_params,
            n_batch_loss_added=n_batch_loss_added,
            activation_store=activation_store,
            split=split,
            verbose=verbose,
            **ablate_params,
        )
        metrics_list.append(ablated_metrics)

        if save_metrics:
            modified_ablate_metrics = ablated_metrics.copy()
            modified_ablate_metrics["ablate_params"] = ablate_params

            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "wb") as f:
                pickle.dump(modified_ablate_metrics, f)

    return metrics_list


def convert_list_of_dicts_to_dict_of_lists(list_of_dicts):
    # Initialize an empty dictionary to hold the lists
    dict_of_lists = {}

    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)

    return dict_of_lists


def create_df_from_metrics(metrics_list):
    df_data = []

    dataset_names = list(metrics_list[0].keys())

    if "loss_added" in dataset_names:
        dataset_names.remove("loss_added")

    if "ablate_params" in dataset_names:
        dataset_names.remove("ablate_params")

    for metric in metrics_list:
        if "loss_added" in metric:
            loss_added = metric["loss_added"]
        else:
            loss_added = np.NaN
        mean_correct = [metric[dataset_name]["mean_correct"] for dataset_name in dataset_names]
        mean_predicted_probs = [
            metric[dataset_name]["mean_predicted_probs"] for dataset_name in dataset_names
        ]

        metric_data = np.concatenate(([loss_added], mean_correct, mean_predicted_probs))

        df_data.append(metric_data)

    df_data = np.array(df_data)

    columns = ["loss_added"] + dataset_names + [x + "_prob" for x in dataset_names]
    df = pd.DataFrame(df_data, columns=columns)

    return df


def save_target_question_ids(
    model: HookedTransformer,
    mcq_batch_size: int,
    dataset_name: str,
    output_dir: str = "../data/question_ids",
    train_ratio: float = 0.5,
):
    """
    Find and save the question ids where the model
    1. correct: all permutations correct
    2. correct-iff-question: all permutations correct iff with instruction and questions
    3. correct-no-tricks: all permutations correct and without tricks
    """

    full_dataset_name = (
        f'mmlu-{dataset_name.replace("_", "-")}' if dataset_name != "wmdp-bio" else dataset_name
    )
    model_name = model.cfg.model_name

    # Check if the files already exist
    file_paths = [
        os.path.join(output_dir, f"{model_name}/{split}/{full_dataset_name}_{q_type}.csv")
        for q_type in ["correct", "correct-iff-question", "correct-no-tricks"]
        for split in ["train", "test", "all"]
    ]

    if all(os.path.exists(file_path) for file_path in file_paths):
        print(
            f"All target question ids for {model_name} on {dataset_name} already exist. No need to generate target ids."
        )
        return

    print(f"Saving target question ids for {model_name} on {dataset_name}...")

    metrics = calculate_MCQ_metrics(
        model, mcq_batch_size, dataset_name, permutations=all_permutations
    )
    metrics_wo_question = calculate_MCQ_metrics(
        model, mcq_batch_size, dataset_name, permutations=all_permutations, without_question=True
    )

    # find all permutations correct
    all_types = {
        "correct": (correct_ids := _find_all_permutation_correct_ans(metrics)),
        "correct-iff-question": _find_correct_iff_question(correct_ids, metrics_wo_question),
        "correct-no-tricks": _find_correct_no_tricks(correct_ids, dataset_name),
    }

    for q_type, q_ids in all_types.items():
        train, test = _split_train_test(q_ids, train_ratio=train_ratio)
        splits = {"train": train, "test": test, "all": q_ids}

        for split, ids in splits.items():
            file_name = os.path.join(
                output_dir, f"{model.cfg.model_name}/{split}/{full_dataset_name}_{q_type}.csv"
            )
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            np.savetxt(file_name, ids, fmt="%d")
            print(f"{file_name} saved, with {len(ids)} questions")


def _find_all_permutation_correct_ans(metrics):
    each_question_acc = metrics["is_correct"].reshape(-1, 24)
    questions_correct = each_question_acc.sum(axis=1) == 24
    correct_question_id = np.where(questions_correct)[0]

    return correct_question_id


def _find_correct_iff_question(correct_questions, metrics_wo_question):
    each_question_acc_wo_question = metrics_wo_question["is_correct"].reshape(-1, 24)
    correct_wo_question = np.where(each_question_acc_wo_question.sum(axis=1) == 24)[0]
    questions_correct_iff_question = list(set(correct_questions) - set(correct_wo_question))

    return np.array(questions_correct_iff_question)


def load_dataset_from_name(dataset_name: str):
    if dataset_name == "wmdp-bio":
        dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
    else:
        dataset = load_dataset("cais/mmlu", dataset_name, split="test")
    return dataset


def _find_correct_no_tricks(correct_questions, dataset_name):
    dataset = load_dataset_from_name(dataset_name)
    choices_list = [x["choices"] for x in dataset]

    def matches_pattern(s):
        pattern = r"^(Both )?(A|B|C|D) (and|&) (A|B|C|D)$"
        return bool(re.match(pattern, s)) or s == "All of the above"

    correct_no_tricks = []
    for question_id in correct_questions:
        if not any(matches_pattern(choice) for choice in choices_list[question_id]):
            correct_no_tricks.append(question_id)

    return np.array(correct_no_tricks)


def _split_train_test(questions_ids, train_ratio=0.5):
    """shuffle then split the questions ids into train and test"""
    questions_ids = np.random.permutation(questions_ids)
    split = int(len(questions_ids) * train_ratio)
    return questions_ids[:split], questions_ids[split:]
