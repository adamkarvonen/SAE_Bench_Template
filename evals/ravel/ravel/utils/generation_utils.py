# utils.py
from typing import Union, List, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from nnsight import NNsight
from transformer_lens import HookedTransformer

@dataclass
class Prompt:
    """Represents a single prompt with its associated data."""

    text: str
    template: str
    attribute: str
    entity: str
    context_split: str
    entity_split: str
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    completion: Optional[str] = None
    is_correct: Optional[bool] = None


def generate_batched(
    pretrained_model,
    tokenizer,
    all_prompts: List[Prompt],
    max_length=None,
    prompt_max_length=32,
    max_new_tokens=None,
    sample_n=None,
    batch_size=32,
    **kwargs,
):
    print(f"Total #prompts={len(all_prompts)}")
    pretrained_model = pretrained_model.eval()
    if prompt_max_length is None:
        max_length_prompt = max(all_prompts, key=lambda x: len(x.text))
        prompt_max_length = 8 * (
            len(tokenizer(max_length_prompt.text).input_ids) // 8 + 1
        )
        print(f"Set prompt_max_length={prompt_max_length}")

    completions = []
    for batch_begin in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[batch_begin : batch_begin + batch_size]
        batch_completions = _generate_single_batch(
            pretrained_model,
            tokenizer,
            batch_prompts,
            prompt_max_length=prompt_max_length,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            sample_n=sample_n,
            **kwargs,
        )
        completions.extend(batch_completions)
    return completions


def _generate_single_batch(
    pretrained_model: Union[NNsight, HookedTransformer],
    tokenizer,
    prompt_batch: List[Prompt],
    max_length=None,
    prompt_max_length=32,
    max_new_tokens=None,
    sample_n=None,
    **kwargs,
):
    if isinstance(pretrained_model, NNsight):
        device = pretrained_model.device
    elif isinstance(pretrained_model, HookedTransformer):
        device = next(pretrained_model.parameters()).device
    if not sample_n:
        sample_n = 1
    if not max_new_tokens:
        assert max_length and prompt_max_length
        max_new_tokens = max_length - prompt_max_length

    input_ids = torch.stack([p.input_ids for p in prompt_batch]).to(
        device
    )
    attention_mask = torch.stack([p.attention_mask for p in prompt_batch]).to(
        device
    )

    with torch.no_grad():
        if isinstance(pretrained_model, NNsight):
            outputs = pretrained_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True if sample_n > 1 else False,
                num_return_sequences=sample_n,
                return_dict_in_generate=False,
                pad_token_id=tokenizer.pad_token_id,
                **kwargs,
            )
        elif isinstance(pretrained_model, HookedTransformer):
            if sample_n != 1:
                raise NotImplementedError(f"sample_n not supported for HookedTransformer; got {sample_n}")
            # we need to re-create the prompts and pad on the left, because in
            # transformerlens we can't pass the attention mask directly
            input_ids = pretrained_model.to_tokens([p.text for p in prompt_batch], padding_side='left', prepend_bos=True, )
            outputs = pretrained_model.generate(
                input=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0,
                # padding_side='left',
                verbose=False,
            )
    preds = [
        (prompt_batch[i // sample_n].text, p)
        for i, p in enumerate(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ]
    return preds
