from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import BaseEvalConfig
from typing import List

DEBUG_MODE = False

@dataclass
class RAVELEvalConfig(BaseEvalConfig):
    # Dataset
    entity_class: str = Field(
        default="nobel_prize_winner",
        title="Chosen Entity",
        description="Entity to be evaluated.",
    )
    attribute_class_A: str = Field(
        default="Field",
        title="Attribute Class A",
        description="Attribute class A.",
    )
    attribute_class_B: str = Field(
        default="Country of Birth",
        title="Attribute Class B",
        description="Attribute class B.",
    )
    n_samples_per_attribute_class: int = Field(
        default=None,
        title="Number of Samples per Attribute Class",
        description="Number of samples per attribute class. If None, all samples are used.",
    )
    top_n_entities: int = Field(
        default=400,
        title="Number of distinct entities in the dataset",
        description="Number of entities in the dataset, filtered by prediction accuracy over attributes / templates."
    )
    top_n_templates: int = Field(
        default=12,
        title="Number of distinct templates in the dataset",
        description="Number of templates in the dataset, filtered by prediction accuracy over entities."
    )
    force_dataset_recompute: bool = Field(
        default=False,
        title="Force Dataset Recompute",
        description="Force recomputation of the dataset, ie. generating model predictions for attribute values, evaluating, and downsampling.",
    )

    # Language model and SAE
    model_name: str = Field(
        default="gemma-2-2b",
        title="Model Name",
        description="Model name",
    )
    layer: int = Field(
        default=11,
        title="Layer",
        description="Layer",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Data Type",
        description="LLM data type",
    )
    llm_batch_size: int = Field(
        default=32,
        title="LLM Batch Size",
        description="LLM batch size, inference only",
    )
    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="SAE batch size, inference only",
    )

    # Probe
    probe_coefficients: List[int] = Field(
        default=[0.01, 0.1, 10, 100, 1000],
        title="Probe Coefficients",
        description="Probe coefficients determining the number of patched features.",
    )
    max_samples_per_attribute: int = Field(
        default=1024,
        title="Max Samples per Attribute",
        description="Indirect definition of probe training datset size, which contains half target attribute and half balanced mix of non-target attributes.",
    )

    # Intervention
    n_interventions: int = Field(
        default=128,
        title="Number of Interventions",
        description="Number of interventions per attribute feature threshold, ie. number of experiments to compute a single cause / isolation score.",
    )
    n_generated_tokens: int = Field(
        default=8,
        title="Number of Generated Tokens",
        description="Number of tokens to generate for each intervention. 8 was used in the RAVEL paper",
    )
    inv_batch_size: int = Field(
        default=16,
        title="Intervention Batch Size",
        description="Intervention batch size, inference only",
    )

    # Misc
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )



    if DEBUG_MODE:
        n_samples_per_attribute_class = 50
        top_n_entities = 10
        top_n_templates = 2

        n_interventions = 10
        llm_batch_size = 5
