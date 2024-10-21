from datasets import load_dataset
import json
import einops
from tqdm import tqdm
import torch
from torch import Tensor
from jaxtyping import Float
import gc
import numpy as np
import random
import os

from sae_lens import SAE
from transformer_lens import HookedTransformer

FORGET_FILENAME = "feature_sparsity_forget.txt"
RETAIN_FILENAME = "feature_sparsity_retain.txt"

SPARSITIES_DIR = "results/sparsities"


def get_forget_retain_data(
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    min_len: int = 50,
    max_len: int = 2000,
    batch_size: int = 4,
) -> tuple[list[str], list[str]]:
    retain_dataset = []
    if retain_corpora == "wikitext":
        raw_retain = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for x in raw_retain:
            if len(x["text"]) > min_len:
                retain_dataset.append(str(x["text"]))
    else:
        raise Exception("Unknown retain corpora")

    forget_dataset = []
    for line in open(f"./data/{forget_corpora}.jsonl", "r"):
        if "bio-forget-corpus" in forget_corpora:
            raw_text = json.loads(line)["text"]
        else:
            raw_text = line
        if len(raw_text) > min_len:
            forget_dataset.append(str(raw_text))

    return forget_dataset, retain_dataset


def tokenize_dataset(
    model: HookedTransformer, dataset: list[str], seq_len: int = 1024, max_batch: int = 32
):
    # just for quick testing on smaller tokens
    # dataset = dataset[:max_batch]
    full_text = model.tokenizer.eos_token.join(dataset)

    # divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
    tokens = model.tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()

    # remove pad token
    tokens = tokens[tokens != model.tokenizer.pad_token_id]
    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len

    # drop last batch if not full
    tokens = tokens[: num_batches * seq_len]
    tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)
    # change first token to bos
    tokens[:, 0] = model.tokenizer.bos_token_id
    return tokens.to("cuda")


def get_shuffled_forget_retain_tokens(
    model: HookedTransformer,
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    batch_size: int = 2048,
    seq_len: int = 1024,
):
    """
    get shuffled forget tokens and retain tokens, with given batch size and sequence length
    note: wikitext has less than 2048 batches with seq_len=1024
    """
    forget_dataset, retain_dataset = get_forget_retain_data(forget_corpora, retain_corpora)

    print(len(forget_dataset), len(forget_dataset[0]))
    print(len(retain_dataset), len(retain_dataset[0]))

    shuffled_forget_dataset = random.sample(forget_dataset, min(batch_size, len(forget_dataset)))

    forget_tokens = tokenize_dataset(model, shuffled_forget_dataset, seq_len=seq_len)
    retain_tokens = tokenize_dataset(model, retain_dataset, seq_len=seq_len)

    print(forget_tokens.shape, retain_tokens.shape)
    shuffled_forget_tokens = forget_tokens[torch.randperm(forget_tokens.shape[0])]
    shuffled_retain_tokens = retain_tokens[torch.randperm(retain_tokens.shape[0])]

    return shuffled_forget_tokens[:batch_size], shuffled_retain_tokens[:batch_size]


def gather_residual_activations(model: HookedTransformer, target_layer: int, inputs):
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act


def get_feature_activation_sparsity(
    model: HookedTransformer, sae: SAE, tokens, batch_size: int = 4
):
    mean_acts = []
    layer = int(sae.cfg.hook_layer)

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens[i : i + batch_size], names_filter=sae.cfg.hook_name
            )
            resid: Float[Tensor, "batch pos d_model"] = cache[sae.cfg.hook_name]
            # resid: Float[Tensor, 'batch pos d_model'] = gather_residual_activations(model, layer, tokens[i:i + batch_size])
            resid = resid.to(torch.float)

            act: Float[Tensor, "batch pos d_sae"] = sae.encode(resid)
            # make act to zero or one
            act = (act > 0).to(torch.float)
            current_mean_act = einops.reduce(act, "batch pos d_sae -> d_sae", "mean")

        mean_acts.append(current_mean_act)

        # Free up memory
        del resid, act
        torch.cuda.empty_cache()
        gc.collect()

    mean_acts = torch.stack(mean_acts)
    return mean_acts.to(torch.float16).mean(dim=0).detach().cpu().numpy()


def get_top_features(forget_score, retain_score, retain_threshold=0.01):
    # criteria for selecting features: retain score < 0.01 and then sort by forget score
    high_retain_score_features = np.where(retain_score >= retain_threshold)[0]
    modified_forget_score = forget_score.copy()
    modified_forget_score[high_retain_score_features] = 0
    top_features = modified_forget_score.argsort()[::-1]
    # print(top_features[:20])

    n_non_zero_features = np.count_nonzero(modified_forget_score)
    top_features_non_zero = top_features[:n_non_zero_features]

    return top_features_non_zero


def check_existing_results(sae_folder):
    forget_path = os.path.join(SPARSITIES_DIR, sae_folder, FORGET_FILENAME)
    retain_path = os.path.join(SPARSITIES_DIR, sae_folder, RETAIN_FILENAME)
    return os.path.exists(forget_path) and os.path.exists(retain_path)


def calculate_sparsity(
    model: HookedTransformer, sae: SAE, forget_tokens, retain_tokens, batch_size: int
):
    feature_sparsity_forget = get_feature_activation_sparsity(
        model, sae, forget_tokens, batch_size=batch_size
    )
    feature_sparsity_retain = get_feature_activation_sparsity(
        model, sae, retain_tokens, batch_size=batch_size
    )
    return feature_sparsity_forget, feature_sparsity_retain


def save_results(sae_folder, feature_sparsity_forget, feature_sparsity_retain):
    output_dir = os.path.join(SPARSITIES_DIR, sae_folder)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, FORGET_FILENAME), feature_sparsity_forget, fmt="%f")
    np.savetxt(os.path.join(output_dir, RETAIN_FILENAME), feature_sparsity_retain, fmt="%f")


def load_sparsity_data(sae_folder: str) -> tuple[np.ndarray, np.ndarray]:
    forget_sparsity = np.loadtxt(
        os.path.join(SPARSITIES_DIR, sae_folder, FORGET_FILENAME), dtype=float
    )
    retain_sparsity = np.loadtxt(
        os.path.join(SPARSITIES_DIR, sae_folder, RETAIN_FILENAME), dtype=float
    )
    return forget_sparsity, retain_sparsity


def save_feature_sparsity(
    model: HookedTransformer,
    sae: SAE,
    sae_name: str,
    dataset_size: int,
    seq_len: int,
    batch_size: int,
):
    if check_existing_results(sae_name):
        print(f"Sparsity calculation for {sae_name} is already done")
        return

    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
        model, batch_size=dataset_size, seq_len=seq_len
    )

    feature_sparsity_forget, feature_sparsity_retain = calculate_sparsity(
        model, sae, forget_tokens, retain_tokens, batch_size
    )

    save_results(sae_name, feature_sparsity_forget, feature_sparsity_retain)
