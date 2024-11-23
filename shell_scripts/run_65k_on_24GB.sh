# User configuration
sae_regex_pattern="gemma-scope-2b-pt-res-canonical"
model_name="gemma-2-2b"
model_name_it="gemma-2-2b-it"

# Create array of patterns
declare -a sae_block_patterns=(
    ".*layer_5.*(65k).*"
    ".*layer_12.*(65k).*"
    ".*layer_19.*(65k).*"
)

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting pattern ${sae_block_pattern}..."
#     python evals/absorption/main.py \
#         --sae_regex_pattern "${sae_regex_pattern}" \
#         --sae_block_pattern "${sae_block_pattern}" \
#         --model_name ${model_name} --llm_batch_size 4 || {
#             echo "Pattern ${sae_block_pattern} failed, continuing to next pattern..."
#             continue
#         }
#     echo "Completed pattern ${sae_block_pattern}"
# done

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting pattern ${sae_block_pattern}..."
    python evals/autointerp/main.py \
        --sae_regex_pattern "${sae_regex_pattern}" \
        --sae_block_pattern "${sae_block_pattern}" \
        --model_name ${model_name} || {
            echo "Pattern ${sae_block_pattern} failed, continuing to next pattern..."
            continue
        }
    echo "Completed pattern ${sae_block_pattern}"
done

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting core eval for pattern ${sae_block_pattern}..."
    python evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
    --batch_size_prompts 16 \
    --n_eval_sparsity_variance_batches 2000 \
    --n_eval_reconstruction_batches 200 \
    --output_folder "eval_results/core" \
    --exclude_special_tokens_from_reconstruction --verbose --llm_dtype bfloat16 || {
        echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed core eval for pattern ${sae_block_pattern}"
done

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting SCR eval for pattern ${sae_block_pattern}..."
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "${sae_regex_pattern}" \
    --sae_block_pattern "${sae_block_pattern}" \
    --model_name ${model_name} \
    --perform_scr true \
    --clean_up_activations \
    --lower_vram_usage || {
        echo "SCR eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed SCR eval for pattern ${sae_block_pattern}"
done

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting TPP eval for pattern ${sae_block_pattern}..."
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "${sae_regex_pattern}" \
    --sae_block_pattern "${sae_block_pattern}" \
    --model_name ${model_name} \
    --perform_scr false \
    --clean_up_activations \
    --lower_vram_usage || {
        echo "TPP eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed TPP eval for pattern ${sae_block_pattern}"
done

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting sparse probing for pattern ${sae_block_pattern}..."
    python evals/sparse_probing/main.py \
    --sae_regex_pattern "${sae_regex_pattern}" \
    --sae_block_pattern "${sae_block_pattern}" \
    --model_name ${model_name} \
    --clean_up_activations \
    --lower_vram_usage || {
        echo "Sparse probing for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed sparse probing for pattern ${sae_block_pattern}"
done

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting unlearning for pattern ${sae_block_pattern}..."
    python evals/unlearning/main.py \
    --sae_regex_pattern "${sae_regex_pattern}" \
    --sae_block_pattern "${sae_block_pattern}" \
    --model_name ${model_name_it} || {
        echo "Unlearning for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed unlearning for pattern ${sae_block_pattern}"
done
