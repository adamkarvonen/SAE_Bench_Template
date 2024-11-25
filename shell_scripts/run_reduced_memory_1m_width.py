import subprocess
from sae_bench_utils.sae_selection_utils import get_saes_from_regex

# User configuration
sae_regex_pattern = "gemma-scope-2b-pt-res"
model_name = "gemma-2-2b"
model_name_it = "gemma-2-2b-it"

layers = [5, 12, 19]

sae_block_patterns = []
clean_up_patterns = []

for layer in layers:
    # Also configure this to select the SAE width
    single_sae_block_pattern = rf".*layer_({layer}).*(1m).*"

    selected_saes = get_saes_from_regex(sae_regex_pattern, single_sae_block_pattern)

    print(f"Selected {len(selected_saes)} SAEs:")

    count = 0
    for sae_release, sae_id in selected_saes:
        count += 1
        sae_block_patterns.append(sae_id)

        if count == len(selected_saes):
            clean_up_patterns.append(sae_id)

print("SAE block patterns:")
print(sae_block_patterns)
print()
print("Clean up patterns:")
print(clean_up_patterns)

# Get total number of patterns
total_patterns = len(sae_block_patterns)
print(f"Total patterns: {total_patterns}")


# Function to run a command with error handling
def run_command(cmd, fail_message):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"{fail_message}, continuing to next pattern...")
        return False
    return True


# SCR evaluation
current_pattern = 0
for sae_block_pattern in sae_block_patterns:
    print(
        f"Starting SCR eval for pattern {sae_block_pattern} ({current_pattern}/{total_patterns})..."
    )

    cmd = [
        "python",
        "evals/scr_and_tpp/main.py",
        "--sae_regex_pattern",
        sae_regex_pattern,
        "--sae_block_pattern",
        sae_block_pattern,
        "--model_name",
        model_name,
        "--perform_scr",
        "true",
        "--sae_batch_size=5",
        "--lower_vram_usage",
    ]

    if sae_block_pattern in clean_up_patterns:
        cmd.append("--clean_up_activations")
        print("Final iteration - cleanup enabled")

    if run_command(cmd, f"SCR eval for pattern {sae_block_pattern} failed"):
        print(f"Completed SCR eval for pattern {sae_block_pattern}")

# TPP evaluation
current_pattern = 0
for sae_block_pattern in sae_block_patterns:
    current_pattern += 1
    print(
        f"Starting TPP eval for pattern {sae_block_pattern} ({current_pattern}/{total_patterns})..."
    )

    cmd = [
        "python",
        "evals/scr_and_tpp/main.py",
        "--sae_regex_pattern",
        sae_regex_pattern,
        "--sae_block_pattern",
        sae_block_pattern,
        "--model_name",
        model_name,
        "--perform_scr",
        "false",
        "--sae_batch_size=5",
        "--lower_vram_usage",
    ]

    if sae_block_pattern in clean_up_patterns:
        cmd.append("--clean_up_activations")
        print("Final iteration - cleanup enabled")

    if run_command(cmd, f"TPP eval for pattern {sae_block_pattern} failed"):
        print(f"Completed TPP eval for pattern {sae_block_pattern}")

# Sparse probing evaluation
current_pattern = 0
for sae_block_pattern in sae_block_patterns:
    current_pattern += 1
    print(
        f"Starting sparse probing eval for pattern {sae_block_pattern} ({current_pattern}/{total_patterns})..."
    )

    cmd = [
        "python",
        "evals/sparse_probing/main.py",
        "--sae_regex_pattern",
        sae_regex_pattern,
        "--sae_block_pattern",
        sae_block_pattern,
        "--model_name",
        model_name,
        "--sae_batch_size=5",
        "--lower_vram_usage",
    ]

    if sae_block_pattern in clean_up_patterns:
        cmd.append("--clean_up_activations")
        print("Final iteration - cleanup enabled")

    if run_command(cmd, f"Sparse probing eval for pattern {sae_block_pattern} failed"):
        print(f"Completed sparse probing eval for pattern {sae_block_pattern}")

# Absorption evaluation
for sae_block_pattern in sae_block_patterns:
    print(f"Starting absorption eval for pattern {sae_block_pattern}...")
    cmd = [
        "python",
        "evals/absorption/main.py",
        "--sae_regex_pattern",
        sae_regex_pattern,
        "--sae_block_pattern",
        sae_block_pattern,
        "--model_name",
        model_name,
        "--llm_batch_size",
        "4",
        "--k_sparse_probe_batch_size",
        "512",
    ]
    if run_command(cmd, f"Absorption eval for pattern {sae_block_pattern} failed"):
        print(f"Completed absorption eval for pattern {sae_block_pattern}")

# Autointerp evaluation
for sae_block_pattern in sae_block_patterns:
    print(f"Starting autointerp eval for pattern {sae_block_pattern}...")
    cmd = [
        "python",
        "evals/autointerp/main.py",
        "--sae_regex_pattern",
        sae_regex_pattern,
        "--sae_block_pattern",
        sae_block_pattern,
        "--model_name",
        model_name,
        "--llm_batch_size",
        "4",
    ]
    if run_command(cmd, f"Autointerp eval for pattern {sae_block_pattern} failed"):
        print(f"Completed autointerp eval for pattern {sae_block_pattern}")

# Core evaluation
for sae_block_pattern in sae_block_patterns:
    print(f"Starting core eval for pattern {sae_block_pattern}...")
    cmd = [
        "python",
        "evals/core/main.py",
        sae_regex_pattern,
        sae_block_pattern,
        "--batch_size_prompts",
        "2",
        "--n_eval_sparsity_variance_batches",
        "16000",
        "--n_eval_reconstruction_batches",
        "1600",
        "--output_folder",
        "eval_results/core",
        "--exclude_special_tokens_from_reconstruction",
        "--verbose",
        "--llm_dtype",
        "bfloat16",
    ]
    if run_command(cmd, f"Core eval for pattern {sae_block_pattern} failed"):
        print(f"Completed core eval for pattern {sae_block_pattern}")

# Unlearning evaluation
for sae_block_pattern in sae_block_patterns:
    print(f"Starting unlearning eval for pattern {sae_block_pattern}...")
    cmd = [
        "python",
        "evals/unlearning/main.py",
        "--sae_regex_pattern",
        sae_regex_pattern,
        "--sae_block_pattern",
        sae_block_pattern,
        "--model_name",
        model_name_it,
        "--llm_batch_size",
        "1",
    ]
    if run_command(cmd, f"Unlearning eval for pattern {sae_block_pattern} failed"):
        print(f"Completed unlearning eval for pattern {sae_block_pattern}")
