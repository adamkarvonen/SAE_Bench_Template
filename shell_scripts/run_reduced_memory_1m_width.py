import subprocess

# User configuration
sae_regex_pattern = "gemma-scope-9b-pt-res"
model_name = "gemma-2-9b"
model_name_it = "gemma-2-9b-it"

# Create list of patterns
sae_block_patterns = [
    "layer_20/width_1m/average_l0_101",
    "layer_20/width_1m/average_l0_11",
    "layer_20/width_1m/average_l0_19",
    "layer_20/width_1m/average_l0_193",
    "layer_20/width_1m/average_l0_34",
    "layer_20/width_1m/average_l0_57",
    "layer_31/width_1m/average_l0_11",
    "layer_31/width_1m/average_l0_132",
    "layer_31/width_1m/average_l0_25",
    "layer_31/width_1m/average_l0_27",
    "layer_31/width_1m/average_l0_45",
    "layer_31/width_1m/average_l0_77",
    "layer_9/width_1m/average_l0_122",
    "layer_9/width_1m/average_l0_14",
    "layer_9/width_1m/average_l0_24",
    "layer_9/width_1m/average_l0_41",
    "layer_9/width_1m/average_l0_70",
    "layer_9/width_1m/average_l0_9",
]

# These are the final pattern for each SAE layer that signal time to cleanup activations
clean_up_patterns = [
    "layer_20/width_1m/average_l0_57",
    "layer_31/width_1m/average_l0_77",
    "layer_9/width_1m/average_l0_9",
]

# gemma-2-2b patterns
# sae_block_patterns = [
#     "layer_5/width_1m/average_l0_114",
#     "layer_5/width_1m/average_l0_13",
#     "layer_5/width_1m/average_l0_21",
#     "layer_5/width_1m/average_l0_36",
#     "layer_5/width_1m/average_l0_63",
#     "layer_5/width_1m/average_l0_9",
#     "layer_12/width_1m/average_l0_107",
#     "layer_12/width_1m/average_l0_19",
#     "layer_12/width_1m/average_l0_207",
#     "layer_12/width_1m/average_l0_26",
#     "layer_12/width_1m/average_l0_58",
#     "layer_12/width_1m/average_l0_73",
#     "layer_19/width_1m/average_l0_157",
#     "layer_19/width_1m/average_l0_16",
#     "layer_19/width_1m/average_l0_18",
#     "layer_19/width_1m/average_l0_29",
#     "layer_19/width_1m/average_l0_50",
#     "layer_19/width_1m/average_l0_88",
# ]

# clean_up_patterns = [
#     "layer_5/width_1m/average_l0_9",
#     "layer_12/width_1m/average_l0_73",
#     "layer_19/width_1m/average_l0_88",
# ]

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
