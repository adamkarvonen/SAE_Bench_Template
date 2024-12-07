"""
RAVEL Entity Prompt Data Module

This module provides functionality for handling and processing entity prompt data
for the RAVEL evaluation benchmark.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pickle as pkl

import torch
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoTokenizer

from evals.ravel.validation import evaluate_completion
from evals.ravel.eval_config import RAVELEvalConfig
from evals.ravel.generation import generate_batched


# Set random seed for reproducibility
eval_config = RAVELEvalConfig()
rng = random.Random(eval_config.random_seed)


@dataclass
class AttributePrompt:
    """Represents an attribute with its associated prompt templates."""

    attribute_class: str
    templates: List[str]


@dataclass
class Prompt:
    """Represents a single prompt with its associated data."""

    text: str
    template: str
    attribute_class: str
    attribute_label: str
    entity: str
    context_split: str
    entity_split: str
    input_ids: Optional[torch.Tensor] = None
    final_entity_token_pos: Optional[int] = (
        None  # Position of the final entity token in the input_ids, as counted from the end (negative index)
    )
    attention_mask: Optional[torch.Tensor] = None
    completion: Optional[str] = None
    is_correct: Optional[bool] = None


class RAVELInstance:
    def __init__(self):
        self.prompts = {}  # prompt text -> Prompt object
        self.entity_attributes = {}  # entity -> attribute -> value
        self.template_splits = {}  # template -> 'train'/'val'
        self.entity_splits = {}  # entity -> 'train'/'val'
        self.attribute_prompts = []  # templates per attribute

    @classmethod
    def from_files(
        cls,
        entity_type: str,
        data_dir: str,
        tokenizer: AutoTokenizer,
        n_samples_per_attribute_class: Optional[int] = None,
        max_prompt_length: int = 64,
    ) -> "RAVELInstance":
        instance = cls()

        # Load data files
        with open(
            os.path.join(data_dir, "base", f"ravel_{entity_type}_attribute_to_prompts.json")
        ) as f:
            attribute_prompts_dict = json.load(f)
        with open(os.path.join(data_dir, "base", f"ravel_{entity_type}_prompt_to_split.json")) as f:
            instance.template_splits = json.load(f)
        with open(
            os.path.join(data_dir, "base", f"ravel_{entity_type}_entity_attributes.json")
        ) as f:
            instance.entity_attributes = json.load(f)
        with open(os.path.join(data_dir, "base", f"ravel_{entity_type}_entity_to_split.json")) as f:
            instance.entity_splits = json.load(f)

        # Create prompts
        for i, x in tqdm(enumerate(instance.entity_attributes), total=len(instance.entity_attributes)):
            for a, ts in attribute_prompts_dict.items():
                for t in ts:
                    text = t % x
                    encoded = tokenizer(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=max_prompt_length,
                        padding_side="left",
                        truncation=False,
                    )

                    remainder = t.split("%s")[1]
                    encoded_remainder = tokenizer(remainder, truncation=False)
                    final_pos = -len(encoded_remainder["input_ids"])

                    instance.prompts[text] = Prompt(
                        text=text,
                        template=t,
                        attribute_class=a,
                        attribute_label=instance.entity_attributes[x][a],
                        entity=x,
                        context_split=instance.template_splits[t],
                        entity_split=instance.entity_splits[x],
                        input_ids=encoded["input_ids"].squeeze(),
                        attention_mask=encoded["attention_mask"].squeeze(),
                        final_entity_token_pos=final_pos,
                    )

            if n_samples_per_attribute_class and i >= n_samples_per_attribute_class:
                break

        instance.attribute_prompts = [
            AttributePrompt(attribute_class=k, templates=v)
            for k, v in attribute_prompts_dict.items()
        ]

        return instance

    def get_prompts_by_split(self, context_split: str) -> List[Prompt]:
        return [prompt for prompt in self.prompts.values() if prompt.context_split == context_split]

    def get_entities(self, split: Optional[str] = None) -> List[str]:
        if split is None:
            return list(self.entity_splits.keys())
        return [
            entity for entity, entity_split in self.entity_splits.items() if entity_split == split
        ]

    def get_attributes(self) -> List[str]:
        return [ap.attribute_class for ap in self.attribute_prompts]

    def get_prompt_by_text(self, text: str) -> Prompt:
        assert text in self.prompts, f'Prompt with text "{text}" not found'
        return self.prompts[text]

    def get_prompts_by_template(self, template: str) -> List[Prompt]:
        return [p for p in self.prompts.values() if p.template == template]

    def get_prompts_by_attribute(
        self, attribute: str, n_samples: Optional[int] = None
    ) -> List[Prompt]:
        prompts = [p for p in self.prompts.values() if p.attribute_class == attribute]
        if n_samples:
            if n_samples > len(prompts):
                print(f"Warning: Requested {n_samples} samples but only {len(prompts)} available")
            return prompts[:n_samples]
        return prompts

    def get_prompts_by_entity(self, entity: str) -> List[Prompt]:
        return [p for p in self.prompts.values() if p.entity == entity]

    def generate_completions(
        self, model: LanguageModel, tokenizer: AutoTokenizer, max_new_tokens: int, llm_batch_size: int = 32, **kwargs
    ) -> None:
        prompts = list(self.prompts.values())
        token_ids = torch.stack([prompt.input_ids for prompt in prompts])
        attention_masks = torch.stack([prompt.attention_mask for prompt in prompts])

        completions = generate_batched(
            model,
            tokenizer,
            input_ids_BL=token_ids,
            attention_mask_BL=attention_masks,
            max_new_tokens=max_new_tokens,
            llm_batch_size=llm_batch_size,
            **kwargs,
        )

        for prompt, completion in zip(prompts, completions):
            prompt.completion = completion

    def evaluate_completion(self, prompt: Prompt, completion: str) -> bool:
        return evaluate_completion(
            text=prompt.text,
            expected_label=self.entity_attributes[prompt.entity][prompt.attribute_class],
            completion=completion,
        )

    def evaluate_correctness(self):
        for prompt in self.prompts.values():
            if prompt.completion is not None:
                prompt.is_correct = self.evaluate_completion(prompt, prompt.completion)

    def get_accuracy_stats(self):
        stats = {}
        for prompt in self.prompts.values():
            if prompt.is_correct is not None:
                key = (prompt.entity, prompt.template)
                if key not in stats:
                    stats[key] = {"correct": 0, "total": 0}
                stats[key]["total"] += 1
                if prompt.is_correct:
                    stats[key]["correct"] += 1
        return stats

    def calculate_average_accuracy(self):
        correct = sum(1 for p in self.prompts.values() if p.is_correct)
        total = len(self.prompts)
        return correct / total if total > 0 else 0

    def _filter_data(self, filtered_prompts: Dict[str, Prompt]) -> "RAVELInstance":
        instance = RAVELInstance()
        instance.prompts = filtered_prompts

        entities = set(p.entity for p in filtered_prompts.values())
        attributes = set(p.attribute_class for p in filtered_prompts.values())
        templates = set(p.template for p in filtered_prompts.values())

        instance.entity_attributes = {
            e: attrs for e, attrs in self.entity_attributes.items() if e in entities
        }
        instance.template_splits = {
            t: split for t, split in self.template_splits.items() if t in templates
        }
        instance.entity_splits = {
            e: split for e, split in self.entity_splits.items() if e in entities
        }
        instance.attribute_prompts = [
            AttributePrompt(
                attribute_class=ap.attribute_class,
                templates=[t for t in ap.templates if t in templates],
            )
            for ap in self.attribute_prompts
            if ap.attribute_class in attributes
        ]
        return instance

    def filter_correct(self):
        correct_prompts = {text: p for text, p in self.prompts.items() if p.is_correct}
        return self._filter_data(correct_prompts)

    def filter_prompts_by_template_format(self):
        return {text: p for text, p in self.prompts.items() if p.template.count("%s") == 1}

    def filter_top_entities_and_templates(
        self, top_n_entities=400, top_n_templates_per_attribute=12
    ):
        stats = self.get_accuracy_stats()

        # Get top entities
        entity_scores = {}
        for (entity, _), stat in stats.items():
            entity_scores[entity] = entity_scores.get(entity, 0) + stat["correct"]
        kept_entities = set(
            sorted(entity_scores, key=entity_scores.get, reverse=True)[:top_n_entities]
        )

        # Get top templates
        template_scores = {}
        for (_, template), stat in stats.items():
            template_scores[template] = template_scores.get(template, 0) + stat["correct"]

        kept_templates = set()
        for attr in set(p.attribute_class for p in self.prompts.values()):
            attr_templates = [t for t in self.attribute_prompts if t.attribute_class == attr][
                0
            ].templates
            kept_templates.update(
                sorted(
                    [t for t in attr_templates if t in template_scores],
                    key=template_scores.get,
                    reverse=True,
                )[:top_n_templates_per_attribute]
            )

        filtered_prompts = {
            text: p
            for text, p in self.prompts.items()
            if p.entity in kept_entities and p.template in kept_templates
        }
        return self._filter_data(filtered_prompts)

    def downsample(self, n: int) -> "RAVELInstance":
        sampled_keys = rng.sample(list(self.prompts.keys()), n)
        sampled_prompts = {k: self.prompts[k] for k in sampled_keys}
        return self._filter_data(sampled_prompts)

    def __len__(self) -> int:
        return len(self.prompts)


def create_filtered_dataset(
    model_id: str,
    chosen_entity: str,
    model,
    force_recompute: bool = False,
    llm_batch_size: int = 512,
    top_n_entities: int = 400,
    top_n_templates: int = 12,
    max_prompt_length: int = 64,
    n_samples_per_attribute_class: Optional[int] = None,
    full_dataset_downsample: int = 8192,
    artifact_dir: str = "evals/ravel/data/",
):
    """
    Creates and saves filtered dataset of correct model completions.

    Args:
        model_id: Identifier for model
        chosen_entity: Entity type to analyze
        model: Language model instance
        force_recompute: Whether to recompute even if cached file exists
        prompt_max_length: Maximum length for prompts
        batch_size: Batch size for generation
        top_n_entities: Number of top entities to keep
        top_n_templates: Number of top templates per attribute to keep
        full_dataset_downsample: Number of prompts to sample from full dataset

    Returns:
        filtered_data: Dataset containing correct completions
        accuracy: Average accuracy of model completions
    """
    os.makedirs(os.path.join(artifact_dir, model_id), exist_ok=True)
    filename = os.path.join(artifact_dir, f"{model_id}/{chosen_entity}_instance.pkl")

    if force_recompute or not os.path.exists(filename):
        # Load and sample data
        print("Tokenizing full dataset")
        full_dataset = RAVELInstance.from_files(
            entity_type=chosen_entity,
            tokenizer=model.tokenizer,
            data_dir='evals/ravel/data/',
            max_prompt_length=max_prompt_length,
            n_samples_per_attribute_class=n_samples_per_attribute_class,
        )
        sampled_dataset = full_dataset.downsample(full_dataset_downsample)
        print(f"Number of prompts sampled: {len(full_dataset.prompts)}")

        # Generate and evaluate completions
        sampled_dataset.generate_completions(
            model,
            model.tokenizer,
            max_new_tokens=8,
            llm_batch_size=llm_batch_size,
        )
        sampled_dataset.evaluate_correctness()

        # Filter data
        dataset = sampled_dataset.filter_correct()
        dataset = dataset.filter_top_entities_and_templates(
            top_n_entities=top_n_entities, top_n_templates_per_attribute=top_n_templates
        )

        # Calculate metrics
        accuracy = sampled_dataset.calculate_average_accuracy()
        print(f"Average accuracy: {accuracy:.2%}")
        print(f"Prompts remaining: {len(dataset)}")
        print(
            f"Entities after filtering: {len(set([p.entity for p in list(dataset.prompts.values())]))}"
        )

        # Save results
        with open(filename, "wb") as f:
            pkl.dump(dataset, f)

    else:
        print("Loading cached data")
        dataset = pkl.load(open(filename, "rb"))
        accuracy = dataset.calculate_average_accuracy()

    return dataset