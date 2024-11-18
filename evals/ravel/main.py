from matplotlib import pyplot as plt
from typing import List
from transformers import BatchEncoding
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

import os
import gc
import time
import random
import argparse
import pickle as pkl
from typing import List, Tuple, Dict, Any, Union, Literal, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sae_lens import SAE
from sae_lens.sae import TopK
from nnsight import LanguageModel

import sae_bench_utils.activation_collection as activation_collection
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils import general_utils as general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)

from instance import RAVELInstance, Prompt
from validation import evaluate_completion
from eval_config import RAVELEvalConfig


DEBUG_MODE = False
MODEL_DIR = "/share/u/can/models" # Set to None to disable model caching
REPO_DIR = "/share/u/can/SAE_Bench_Template"
RESULTS_DIR = "/share/u/can/SAE_Bench_Template/evals/ravel/results"
ARTIFACTS_DIR = "/share/u/can/SAE_Bench_Template/evals/ravel/artifacts"


eval_config = RAVELEvalConfig()
rng = random.Random(eval_config.random_seed)
device = torch.device("cuda:0")


# TODO test for MODEL_ID = 'pythia-70m'
def run_eval(
    eval_config: RAVELEvalConfig,
    selected_saes: list[tuple[str, str]],
    device: torch.device,
):
    # Instanciate evaluation run
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    # Load model
    if eval_config.model_name == "pythia-70m":
        model_id = "eleutherAI/pythia-70m-deduped"
        model_kwargs = {}
    elif eval_config.model_name == "gemma-2-2b":
        model_id = "google/gemma-2-2b"
        model_kwargs = {'low_cpu_mem_usage': True,'attn_implementation': 'eager'}
    else:
        raise ValueError(f"Invalid model name: {eval_config.model_name}")

    if eval_config.llm_dtype == "bfloat16":
        llm_dtype = torch.bfloat16
    elif eval_config.llm_dtype == "float32":
        llm_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {eval_config.llm_dtype}")

    model = LanguageModel(
        model_id,
        cache_dir=MODEL_DIR,
        device_map=device,
        torch_dtype=llm_dtype,
        dispatch=True,
        **model_kwargs,
    )
    model.requires_grad_(False)
    model.eval()

    # Initialize directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results_dict = {}

    for sae_release, sae_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        gc.collect()
        torch.cuda.empty_cache()

        # Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)
        if isinstance(sae_id, str):
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
        else:
            sae = sae_id
            sae_id = "custom_sae"

        sae = sae.to(device=device, dtype=llm_dtype)

        # # Initialize directories
        # artifacts_folder = os.path.join(
        #     artifacts_base_folder, eval_type, eval_config.model_name, sae.cfg.hook_name
        # )
        # sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        # sae_result_file = sae_result_file.replace("/", "_")
        # sae_result_path = os.path.join(output_path, sae_result_file)




        # # Load and filter RAVEL dataset


        # %%
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
            DATA_DIR = os.path.join(REPO_DIR, "evals/ravel/data/")
            os.makedirs(os.path.join(DATA_DIR, model_id), exist_ok=True)
            filename = os.path.join(DATA_DIR, f"{model_id}/{chosen_entity}_instance.pkl")

            if force_recompute or not os.path.exists(filename):
                # Load and sample data
                print("Tokenizing full dataset")
                full_dataset = RAVELInstance.from_files(
                    chosen_entity,
                    DATA_DIR,
                    model.tokenizer,
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
                if not DEBUG_MODE:
                    with open(filename, "wb") as f:
                        pkl.dump(dataset, f)

            else:
                print("Loading cached data")
                dataset = pkl.load(open(filename, "rb"))
                accuracy = dataset.calculate_average_accuracy()

            return dataset


        if DEBUG_MODE:
            n_samples_per_attribute_class = 50
            top_n_entities = 10
            top_n_templates = 2
        else:
            n_samples_per_attribute_class = None
            top_n_entities = 400
            top_n_templates = 12


        dataset = create_filtered_dataset(
            model_id=model_id,
            chosen_entity="nobel_prize_winner",
            model=model,
            force_recompute=True,
            n_samples_per_attribute_class=n_samples_per_attribute_class,
            top_n_entities=top_n_entities,
            top_n_templates=top_n_templates,
        )


        # def load_SAELensSAE(
        #     layer: int,
        #     expansion_factor: int = 2,
        #     k: Literal[20, 40, 80, 160, 320, 640] = 40,
        #     variant: Literal["standard", "topk"] = "topk",
        #     llm_name: Literal["gemma-2-2b", "pythia70m"] = "gemma-2-2b",
        #     ctx: int = 128,
        #     device: str = "cuda",
        # ) -> Tuple[SAELensSAE, dict, Optional[Tensor]]:
        #     """
        #     Load a pre-trained SAE from SAELens
        #     """
        #     k_to_trainer = {20: 0, 40: 1, 80: 2, 160: 3, 320: 4, 640: 5}
        #     trainer = k_to_trainer[k]
        #     # assert llm_name == 'gemma-2-2b', "only gemma-2-2b is supported for now"
        #     if llm_name == "gemma-2-2b":
        #         release = f"sae_bench_{llm_name}_sweep_{variant}_ctx{ctx}_ef{expansion_factor}_0824"
        #         sae_name_prefix = f"{llm_name}_sweep_{variant}_ctx{ctx}_ef{expansion_factor}_0824"
        #     elif llm_name == "pythia70m":
        #         if variant == "standard":
        #             suffix = "0712"
        #         else:
        #             suffix = "0730"
        #         release = f"sae_bench_{llm_name}_sweep_{variant}_ctx{ctx}_{suffix}"
        #         sae_name_prefix = f"{llm_name}_sweep_{variant}_ctx{ctx}_{suffix}"
        #     sae_name_suffix = f"resid_post_layer_{layer}/trainer_{trainer}"
        #     sae_df = formatting_utils.make_available_sae_df(for_printing=False)
        #     sae_name = f"{sae_name_prefix}/{sae_name_suffix}"
        #     sae_id_to_name_map = sae_df.saes_map[release]
        #     sae_name_to_id_map = {v: k for k, v in sae_id_to_name_map.items()}
        #     sae_id = sae_name_to_id_map[sae_name]
        #     sae, cfg_dict, sparsity = SAELensSAE.from_pretrained(
        #         release=release,
        #         sae_id=sae_id,
        #         device=device,
        #     )
        #     sae = sae.to(device=device)
        #     if variant == "topk":
        #         assert isinstance(
        #             sae.activation_fn, TopK
        #         ), "This sae is not a topk sae, you probably have an old sae_lens version"
        #     if llm_name == "gemma-2-2b":
        #         assert (
        #             cfg_dict["activation_fn_kwargs"]["k"] == k
        #         ), f"Expected k={k}, got k={cfg_dict['activation_fn_kwargs']['k']}"
        #     sae.requires_grad_(False)
        #     return sae, cfg_dict, sparsity

        # sae, cfg_dict, sparsity = load_SAELensSAE(layer=11, k=40, llm_name=MODEL_ID, ctx=128, device="cuda")

        # ### TODO Doublecheck the dataset selection used in the paper for training the probe
            
        # TODO Urgent: Doublecheck BOS token handling
        def get_attribute_activations_nnsight(
            model,
            dataset,
            all_attributes,
            max_samples_per_attribute=1024,
            layer=11,
            llm_batch_size=512,
            tracer_kwargs={"scan": False, "validate": False},
        ):
            """
            Get model activations for attribute classification.

            Returns:
                Tuple of (activations tensor, binary labels list)
            """
            # Randomly sample prompts for each attribute
            all_attribute_activations_BD = {}
            for attr in all_attributes:
                attr_prompts = dataset.get_prompts_by_attribute(
                    attr, n_samples=max_samples_per_attribute
                )
                print(f"Number of prompts for {attr}: {len(attr_prompts)}")

                # Prepare inputs
                input_ids, attn_masks, entity_pos = [], [], []
                for p in attr_prompts:
                    input_ids.append(p.input_ids)
                    attn_masks.append(p.attention_mask)
                    entity_pos.append(p.final_entity_token_pos)
                input_ids = torch.stack(input_ids).to(device)
                attn_masks = torch.stack(attn_masks).to(device)
                entity_positions = torch.tensor(entity_pos)

                batch_activations = []
                for batch_begin in range(0, len(input_ids), llm_batch_size):
                    input_ids_BL = input_ids[batch_begin : batch_begin + llm_batch_size]
                    attn_masks_BL = attn_masks[batch_begin : batch_begin + llm_batch_size]
                    entity_pos_B = entity_positions[batch_begin : batch_begin + llm_batch_size]
                    encoding_BL = BatchEncoding({
                        "input_ids": input_ids_BL,
                        "attention_mask": attn_masks_BL,
                    })

                    # Get activations
                    with torch.no_grad(), model.trace(encoding_BL, **tracer_kwargs):
                        llm_act_BLD = model.model.layers[layer].output[0]
                        batch_arange = torch.arange(llm_act_BLD.shape[0])
                        llm_act_BD = llm_act_BLD[batch_arange, entity_pos_B]
                        llm_act_BD = llm_act_BD.save()
                    batch_activations.append(llm_act_BD)

                all_attribute_activations_BD[attr] = torch.cat(batch_activations, dim=0)
            return all_attribute_activations_BD

        def prepare_attribute_probe_data(
            chosen_attribute: str,
            all_attribute_activations_BD: torch.Tensor,
            max_samples_per_attribute: int = 1024,
        ):

            # Select balanced samples
            balanced_attribute_acts = all_attribute_activations_BD[chosen_attribute]
            n_chosen = len(all_attribute_activations_BD[chosen_attribute])
            for attr, acts in all_attribute_activations_BD.items():
                if attr == chosen_attribute:
                    continue
                n_samples = min(max_samples_per_attribute, len(acts))
                balanced_attribute_acts = torch.cat((balanced_attribute_acts, acts[:n_samples]), dim=0)

            # Shuffle and create labels
            n_total = balanced_attribute_acts.shape[0]
            shuffled_indices = torch.randperm(n_total)
            balanced_attribute_acts = balanced_attribute_acts[shuffled_indices]
            labels = torch.zeros(n_total)
            labels[:n_chosen] = 1
            labels = labels[shuffled_indices]

            return balanced_attribute_acts, labels


        # from ravel code
        # NOTE requires importing scikit-learn

        


        def select_features_with_classifier(sae, inputs, labels, coeff=None):
            if coeff is None:
                coeff = [0.01, 0.1, 10, 100, 1000]
            coeff_to_select_features = {}
            for c in tqdm(coeff):
                with torch.no_grad():
                    X_transformed = sae.encode(inputs).to(dtype=torch.float32).cpu().numpy()
                    lsvc = LinearSVC(C=c, penalty="l1", dual=False, max_iter=5000, tol=0.01).fit(
                        X_transformed, labels
                    )
                    selector = SelectFromModel(lsvc, prefit=True)
                    kept_dim = np.where(selector.get_support())[0]
                    coeff_to_select_features[c] = kept_dim
            return coeff_to_select_features


        def run_feature_selection_probe(
            model,
            sae,
            dataset,
            all_attributes,
            coeffs=[0.01, 0.1, 10, 100, 1000],
            max_samples_per_attribute=1024,
            layer=11,
            llm_batch_size=512,
        ):

            # Cache activations
            all_attribute_activations_BD = get_attribute_activations_nnsight(
                model,
                dataset,
                all_attributes,
                max_samples_per_attribute=max_samples_per_attribute,
                layer=layer,
                llm_batch_size=llm_batch_size,
            )

            # Select features for each attribute by trainin a linear probe on all latents at once
            selected_features = {}  # {attr: {c: [kept_dims]}}
            for attr in tqdm(all_attributes, desc="Running feature selection probe for each attribute"):
                balanced_attribute_acts, labels = prepare_attribute_probe_data(
                    attr, all_attribute_activations_BD, max_samples_per_attribute=max_samples_per_attribute
                )
                selected_features[attr] = select_features_with_classifier(
                    sae, balanced_attribute_acts, labels, coeff=coeffs
                )

            return selected_features


        # TODO save results
        all_attributes = dataset.get_attributes()
        print(f"All attributes: {all_attributes}")

        attribute_feature_dict = run_feature_selection_probe(
            model,
            sae,
            dataset,
            all_attributes=all_attributes,
            coeffs=[0.01, 0.1, 10, 100, 1000],
            max_samples_per_attribute=1024,
            layer=11,
            llm_batch_size=512,
        )


        def get_prompt_pairs(dataset, base_attribute_A, attribute_B, n_interventions):
            """
            Selects pairs of base_prompts and source_prompts for the cause and isolation evaluations.
            Base_prompts always contain attribute A templates.
            The cause evaluation requires source_prompts from attribute A templates, attribute values in base and source should differ.
            The isolation evaluation requires source_prompts from attribute B templates.
            """
            all_A_prompts = dataset.get_prompts_by_attribute(base_attribute_A)
            all_B_prompts = dataset.get_prompts_by_attribute(attribute_B)

            base_A_prompts = rng.sample(all_A_prompts, n_interventions)
            source_B_prompts = rng.sample(all_B_prompts, n_interventions)

            def recursively_match_prompts(base_prompt, source_prompts):
                source_prompt = rng.choice(source_prompts)
                if (
                    source_prompt.attribute_label != base_prompt.attribute_label
                ):  # This is obvious for the base_attribute != source_attribute case
                    return source_prompt
                else:
                    return recursively_match_prompts(base_prompt, source_prompts)

            source_A_prompts = []
            for p in base_A_prompts:
                source_A_prompts.append(recursively_match_prompts(p, all_A_prompts))

            return base_A_prompts, source_A_prompts, source_B_prompts





        layer = sae.cfg.hook_layer


        def format_prompt_batch(prompts: List[Prompt]) -> (BatchEncoding, torch.Tensor):
            ids = torch.stack([p.input_ids for p in prompts]).to(device)
            attn = torch.stack([p.attention_mask for p in prompts]).to(device)
            final_entity_pos = torch.tensor([p.final_entity_token_pos for p in prompts]).to(device)
            encoding = BatchEncoding(
                {
                    "input_ids": ids,
                    "attention_mask": attn,
                }
            )
            return encoding, final_entity_pos


        def create_inverted_latent_mask(
            selected_feature_idxs: torch.Tensor, sae_dict_size: int
        ) -> torch.Tensor:
            inverted_latent_mask = torch.ones(sae_dict_size)
            inverted_latent_mask[selected_feature_idxs] = 0
            inverted_latent_mask = inverted_latent_mask.bool()
            return inverted_latent_mask


        def generate_with_intervention(
            nnsight_model: LanguageModel,
            layer: int,
            sae: SAE,
            base_prompts: List[Prompt],
            source_prompts: List[Prompt],
            sae_latent_idxs: torch.Tensor,
            n_generated_tokens: int = 8,
            llm_batch_size: int = 5,
            tracer_kwargs={"scan": False, "validate": False},
        ):
            inverted_latent_mask = create_inverted_latent_mask(sae_latent_idxs, sae_dict_size=sae.cfg.d_sae)
            ## Iterate over batches
            generated_output_tokens = []
            for batch_idx in range(0, len(base_prompts), llm_batch_size):
                base_batch = base_prompts[batch_idx : batch_idx + llm_batch_size]
                source_batch = source_prompts[batch_idx : batch_idx + llm_batch_size]

                base_encoding, base_entity_pos = format_prompt_batch(base_batch)
                source_encoding, source_entity_pos = format_prompt_batch(source_batch)

                # Get source activations
                with nnsight_model.trace(source_encoding, **tracer_kwargs):
                    source_llm_act_BLD = nnsight_model.model.layers[layer].output[0].save()

                batch_arange = torch.arange(source_llm_act_BLD.shape[0])
                source_llm_act_BD = source_llm_act_BLD[batch_arange, source_entity_pos]
                source_sae_act_BS = sae.encode(source_llm_act_BD)
                source_sae_act_BS[:, inverted_latent_mask] = 0
                source_decoded_act_BD = sae.decode(source_sae_act_BS)

                # Generate from base prompts with interventions
                with nnsight_model.generate(
                    base_encoding, max_new_tokens=n_generated_tokens, **tracer_kwargs
                ):
                    # Cache original activations at the final token position of the attribute
                    residual_stream_module = nnsight_model.model.layers[layer]
                    base_act_BLD = residual_stream_module.output[0]
                    base_act_BD = base_act_BLD[batch_arange, base_entity_pos]

                    # Map base activations of attribute-specific features to residual stream space
                    base_sae_act_BS = sae.encode(base_act_BD)
                    base_sae_act_BS[:, inverted_latent_mask] = 0
                    base_decoded_act_BD = sae.decode(base_sae_act_BS)

                    # Intervene on the residual stream by replacing the activations of the attribute-specific features base -> source
                    base_act_BLD[batch_arange, base_entity_pos] += source_decoded_act_BD
                    base_act_BLD[batch_arange, base_entity_pos] -= base_decoded_act_BD
                    # base_act_BLD[batch_arange, base_entity_pos] = source_decoded_act_BD # for testing doing the full activation patch

                    residual_stream_module.output = (base_act_BLD,)

                    # Save the generated tokens
                    out = nnsight_model.generator.output.save()
                    # out = nnsight_model.generator.output[:-(n_generated_tokens)].save()

                out = out[:, -n_generated_tokens:]
                generated_output_tokens.append(out)
            generated_output_tokens = torch.cat(generated_output_tokens, dim=0)
            generated_output_strings = nnsight_model.tokenizer.batch_decode(generated_output_tokens)
            torch.cuda.empty_cache()

            return generated_output_strings


        def evaluate_intervention(prompts, generations):
            n_prompts = len(prompts)
            texts = [p.text for p in prompts]
            labels = [p.attribute_label for p in prompts]
            correct_cnt = 0
            for text, label, generation in zip(texts, labels, generations):
                correct_cnt += int(evaluate_completion(text, label, generation))
            return correct_cnt / n_prompts  # accuracy


        def compute_disentanglement_BtoA(
            model,
            sae,
            dataset,
            attribute_A,
            attribute_B,
            n_interventions=10,
            n_generated_tokens=4,
            llm_batch_size=5,
            tracer_kwargs={"scan": False, "validate": False},
        ):
            base_A_prompts, source_A_prompts, source_B_prompts = get_prompt_pairs(
                dataset, attribute_A, attribute_B, n_interventions
            )

            # Cause evaluation: High accuracy if intervening with A_features is successful on base_A_template, ie. source_A_attribute_value is generated.
            sae_latent_dict = attribute_feature_dict[attribute_A]
            feature_thresholds = sae_latent_dict.keys()
            cause_A_accuracies = {}
            for threshold in tqdm(feature_thresholds, desc="Cause evaluation across feature thresholds"):
                sae_latent_idxs = sae_latent_dict[threshold]
                generated_output_strings = generate_with_intervention(
                    model,
                    layer,
                    sae,
                    base_A_prompts,
                    source_A_prompts,
                    sae_latent_idxs,
                    n_generated_tokens=n_generated_tokens,
                    llm_batch_size=llm_batch_size,
                    tracer_kwargs=tracer_kwargs,
                )
                cause_A_accuracies[threshold] = evaluate_intervention(
                    source_A_prompts, generated_output_strings
                )

            # Isolation evaluation: High accuracy if intervening with B_features is unsuccessful on base_A_template, ie. base_A_attribute is generated regardless of intervention.
            sae_latent_dict = attribute_feature_dict[attribute_B]
            feature_thresholds = sae_latent_dict.keys()
            isolation_BtoA_accuracies = {}
            for threshold in tqdm(
                feature_thresholds, desc="Isolation evaluation across feature thresholds"
            ):
                sae_latent_idxs = sae_latent_dict[threshold]
                generated_output_strings = generate_with_intervention(
                    model,
                    layer,
                    sae,
                    base_A_prompts,
                    source_B_prompts,
                    sae_latent_idxs,
                    n_generated_tokens=n_generated_tokens,
                    llm_batch_size=llm_batch_size,
                    tracer_kwargs=tracer_kwargs,
                )
                isolation_BtoA_accuracies[threshold] = evaluate_intervention(
                    base_A_prompts, generated_output_strings
                )

            return cause_A_accuracies, isolation_BtoA_accuracies


        def compute_disentanglement_AB_bidirectional(
            model,
            sae,
            dataset,
            attribute_A,
            attribute_B,
            n_interventions=10,
            n_generated_tokens=4,
            llm_batch_size=5,
            tracer_kwargs={"scan": False, "validate": False},
        ):
            print(f"1/2 Computing disentanglement score for {attribute_A} -> {attribute_B}")
            cause_A, isolation_BtoA = compute_disentanglement_BtoA(
                model,
                sae,
                dataset,
                attribute_A,
                attribute_B,
                n_interventions=n_interventions,
                n_generated_tokens=n_generated_tokens,
                llm_batch_size=llm_batch_size,
                tracer_kwargs=tracer_kwargs,
            )

            print(f"2/2 Computing disentanglement score for {attribute_B} -> {attribute_A}")
            cause_B, isolation_AtoB = compute_disentanglement_BtoA(
                model,
                sae,
                dataset,
                attribute_B,
                attribute_A,
                n_interventions=n_interventions,
                n_generated_tokens=n_generated_tokens,
                llm_batch_size=llm_batch_size,
                tracer_kwargs=tracer_kwargs,
            )

            # Mean = disentanglement score
            disentanglement_score = {
                t: (cause_A[t] + cause_B[t] + isolation_AtoB[t] + isolation_BtoA[t]) / 4
                for t in cause_A.keys()
            }

            # reorganize results dict
            results = {
                t: {
                    "cause_A": cause_A[t],
                    "isolation_BtoA": isolation_BtoA[t],
                    "cause_B": cause_B[t],
                    "isolation_AtoB": isolation_AtoB[t],
                    "disentanglement": disentanglement_score[t],
                }
                for t in disentanglement_score.keys()
            }
            return results


        results = compute_disentanglement_AB_bidirectional(
            model,
            sae,
            dataset,
            attribute_A="Field",
            attribute_B="Country of Birth",
            n_interventions=128,
            n_generated_tokens=8,
            llm_batch_size=16,
            tracer_kwargs={"scan": False, "validate": False},
        )

        


        def plot_disentanglement(results, title):
            thresholds = list(results.keys())
            print(results)
            cause_A = [results[t]["cause_A"] for t in thresholds]
            isolation_BtoA = [results[t]["isolation_BtoA"] for t in thresholds]
            cause_B = [results[t]["cause_B"] for t in thresholds]
            isolation_AtoB = [results[t]["isolation_AtoB"] for t in thresholds]
            disentanglement = [results[t]["disentanglement"] for t in thresholds]

            fig_dir = os.path.join(RESULTS_DIR, "disentanglement_plot.png")
            plt.figure(figsize=(12, 6))
            plt.plot(thresholds, cause_A, label="Cause A")
            plt.plot(thresholds, isolation_BtoA, label="Isolation B->A")
            plt.plot(thresholds, cause_B, label="Cause B")
            plt.plot(thresholds, isolation_AtoB, label="Isolation A->B")
            plt.plot(thresholds, disentanglement, label="Disentanglement")
            plt.xlabel("Threshold")
            plt.ylabel("Accuracy")
            plt.xscale("log")
            plt.title(title)
            plt.legend()
            plt.savefig(fig_dir)
            plt.show()


        plot_disentanglement(results, "Disentanglement score for Field and Country of Birth")



def create_config_and_selected_saes(
    args,
) -> tuple[RAVELEvalConfig, list[tuple[str, str]]]:
    config = RAVELEvalConfig(
        random_seed=args.random_seed,
        model_name=args.model_name,
    )

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run SCR or TPP evaluation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, default="pythia-70m-deduped", help="Model name")
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
    return parser


if __name__ == "__main__":
    """
    Example Gemma-2-2B SAE Bench usage:
    python evals/ravel/main.py \
    --sae_regex_pattern "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824" \
    --sae_block_pattern "blocks.19.hook_resid_post__trainer_2" \
    --model_name gemma-2-2b
    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    # create output folder
    # os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")