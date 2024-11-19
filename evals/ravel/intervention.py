import torch
from transformers import BatchEncoding
from nnsight import LanguageModel
from sae_lens import SAE
import random
from tqdm import tqdm
from typing import List, Dict

from evals.ravel.instance import Prompt, evaluate_completion
from evals.ravel.eval_config import RAVELEvalConfig

eval_config = RAVELEvalConfig()
rng = random.Random(eval_config.random_seed)


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

def format_prompt_batch(prompts: List[Prompt], device) -> (BatchEncoding, torch.Tensor):
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
    model: LanguageModel,
    layer: int,
    sae: SAE,
    base_prompts: List[Prompt],
    source_prompts: List[Prompt],
    sae_latent_idxs: torch.Tensor,
    n_generated_tokens: int = 8,
    inv_batch_size: int = 5,
    tracer_kwargs={"scan": False, "validate": False},
):
    inverted_latent_mask = create_inverted_latent_mask(sae_latent_idxs, sae_dict_size=sae.cfg.d_sae)
    ## Iterate over batches
    generated_output_tokens = []
    for batch_idx in range(0, len(base_prompts), inv_batch_size):
        base_batch = base_prompts[batch_idx : batch_idx + inv_batch_size]
        source_batch = source_prompts[batch_idx : batch_idx + inv_batch_size]

        base_encoding, base_entity_pos = format_prompt_batch(base_batch, device=model.device)
        source_encoding, source_entity_pos = format_prompt_batch(source_batch, device=model.device)

        # Get source activations
        with model.trace(source_encoding, **tracer_kwargs):
            source_llm_act_BLD = model.model.layers[layer].output[0].save()

        batch_arange = torch.arange(source_llm_act_BLD.shape[0])
        source_llm_act_BD = source_llm_act_BLD[batch_arange, source_entity_pos]
        source_sae_act_BS = sae.encode(source_llm_act_BD)
        source_sae_act_BS[:, inverted_latent_mask] = 0
        source_decoded_act_BD = sae.decode(source_sae_act_BS)

        # Generate from base prompts with interventions
        with model.generate(
            base_encoding, max_new_tokens=n_generated_tokens, **tracer_kwargs
        ):
            # Cache original activations at the final token position of the attribute
            residual_stream_module = model.model.layers[layer]
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
            out = model.generator.output.save()
            # out = nnsight_model.generator.output[:-(n_generated_tokens)].save()

        out = out[:, -n_generated_tokens:]
        generated_output_tokens.append(out)
    generated_output_tokens = torch.cat(generated_output_tokens, dim=0)
    generated_output_strings = model.tokenizer.batch_decode(generated_output_tokens)
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
    layer,
    dataset,
    attribute_A,
    attribute_B,
    attribute_feature_dict: Dict[str, Dict[float, torch.Tensor]],
    n_interventions=10,
    n_generated_tokens=4,
    inv_batch_size=5,
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
            inv_batch_size=inv_batch_size,
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
            inv_batch_size=inv_batch_size,
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
    attribute_feature_dict: Dict[str, Dict[float, torch.Tensor]],
    n_interventions=10,
    n_generated_tokens=4,
    inv_batch_size=5,
    tracer_kwargs={"scan": False, "validate": False},
):
    print(f"1/2 Computing disentanglement score for {attribute_A} -> {attribute_B}")
    cause_A, isolation_BtoA = compute_disentanglement_BtoA(
        model,
        sae,
        sae.cfg.hook_layer,
        dataset,
        attribute_A,
        attribute_B,
        attribute_feature_dict,
        n_interventions,
        n_generated_tokens,
        inv_batch_size,
        tracer_kwargs,
    )

    print(f"2/2 Computing disentanglement score for {attribute_B} -> {attribute_A}")
    cause_B, isolation_AtoB = compute_disentanglement_BtoA(
        model,
        sae,
        sae.cfg.hook_layer,
        dataset,
        attribute_B,
        attribute_A,
        attribute_feature_dict,
        n_interventions,
        n_generated_tokens,
        inv_batch_size,
        tracer_kwargs,
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