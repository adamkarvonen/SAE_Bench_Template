import os
import re
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset

from evals.unlearning.utils.metrics import calculate_MCQ_metrics, all_permutations


def save_target_question_ids(
    model: HookedTransformer,
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
    metrics = calculate_MCQ_metrics(model, dataset_name, permutations=all_permutations)
    metrics_wo_question = calculate_MCQ_metrics(
        model, dataset_name, permutations=all_permutations, without_question=True
    )

    # find all permutations correct
    all_types = {
        "correct": (correct_ids := _find_all_permutation_correct_ans(metrics)),
        "correct-iff-question": _find_correct_iff_question(correct_ids, metrics_wo_question),
        "correct-no-tricks": _find_correct_no_tricks(correct_ids, dataset_name),
    }

    full_dataset_name = (
        f'mmlu-{dataset_name.replace("_", "-")}' if dataset_name != "wmdp-bio" else dataset_name
    )

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


def save_train_test_all(
    dataset_name: str,
    model_name: str = "gemma-2b-it",
    train_ratio: float = 0.5,
    output_dir: str = "../data/question_ids",
):
    """
    Randomly split all the questions ids into train and test, then save them
    """
    dataset = load_dataset_from_name(dataset_name)
    q_ids = np.arange(len(dataset))

    train, test = _split_train_test(q_ids, train_ratio=train_ratio)
    splits = {"train": train, "test": test, "all": q_ids}

    full_dataset_name = (
        f'mmlu-{dataset_name.replace("_", "-")}' if dataset_name != "wmdp-bio" else dataset_name
    )
    q_type = "all"

    for split, ids in splits.items():
        file_name = os.path.join(
            output_dir, f"{model_name}/{split}/{full_dataset_name}_{q_type}.csv"
        )
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.savetxt(file_name, ids, fmt="%d")
        print(f"{file_name} saved, with {len(ids)} questions")
