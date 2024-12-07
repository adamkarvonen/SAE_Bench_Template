from tqdm import tqdm
from jaxtyping import Int

import torch
from transformers import AutoTokenizer, BatchEncoding
from nnsight import LanguageModel


def generate_batched(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    input_ids_BL: Int[torch.Tensor, "batch_size seq_len"],
    attention_mask_BL: Int[torch.Tensor, "batch_size seq_len"],
    max_new_tokens,
    llm_batch_size=32,
):
    num_total_prompts = len(input_ids_BL)

    generations = []
    for batch_begin in tqdm(range(0, num_total_prompts, llm_batch_size), desc="Generate completions to test model knowledge"):
        input_ids = input_ids_BL[batch_begin : batch_begin + llm_batch_size].to(model.device)
        attention_mask = attention_mask_BL[batch_begin : batch_begin + llm_batch_size].to(model.device)
        # Generate using huggingface model
        output_ids = model._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False, # greedy decoding for reproducibility
        )
        generated_ids = output_ids[:, -max_new_tokens:]
        generations.append(generated_ids)
        
    generations = torch.cat(generations, dim=0)
    generated_strings = tokenizer.batch_decode(generations)
    return generated_strings


if __name__ == "__main__":
    # Test the generation
    from nnsight import LanguageModel
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    model = LanguageModel("eleutherAI/pythia-70m-deduped", device_map=device, dispatch=True)
    tokenizer = AutoTokenizer.from_pretrained("eleutherAI/pythia-70m-deduped")
    tokenizer.pad_token = tokenizer.eos_token

    encoded = model.tokenizer.batch_encode_plus(["Hello, world!", "Moin "], return_tensors="pt", padding='max_length', max_length=20).to(device)
    input_ids_BL = encoded['input_ids']
    attention_mask_BL = encoded['attention_mask']
    
    generated_strings = generate_batched(model, tokenizer, input_ids_BL, attention_mask_BL, max_new_tokens=10)
    print(generated_strings)