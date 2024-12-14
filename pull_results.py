import matplotlib.pyplot as plt
# import json
# import os
# from typing import Dict, Optional, Tuple
# import glob
# from pathlib import Path


# def get_custom_metric_key_and_name(eval_path: str, k: Optional[int] = None) -> tuple[str, str]:
#     """Get the metric key and name based on evaluation type."""
#     if "tpp" in eval_path:
#         return f"tpp_threshold_{k}_total_metric", f"TPP Top {k} Metric"
#     elif "scr" in eval_path:
#         return f"scr_metric_threshold_{k}", f"SCR Top {k} Metric"
#     elif "sparse_probing" in eval_path:
#         return f"sae_top_{k}_test_accuracy", f"Sparse Probing Top {k} Test Accuracy"
#     elif "absorption" in eval_path:
#         return "mean_absorption_score", "Mean Absorption Score"
#     elif "autointerp" in eval_path:
#         return "autointerp_score", "Autointerp Score"
#     elif "unlearning" in eval_path:
#         return "unlearning_score", "Unlearning Score"
#     elif "core" in eval_path:
#         return "frac_recovered", "Loss Recovered"
#     else:
#         raise ValueError("Please add the correct key for the custom metric")


# def get_metric_from_json(json_path: str, eval_type: str) -> float:
#     """Extract the relevant metric from a JSON file based on eval type."""
#     print(f"Reading metric from {json_path} for eval_type {eval_type}")
#     try:
#         with open(json_path, "r") as f:
#             data = json.load(f)

#         if "core" in eval_type:
#             sparsity = data["eval_result_metrics"]["sparsity"]["l0"]
#             print(f"Found L0 sparsity value: {sparsity}")
#             return sparsity

#         # Use specific thresholds for certain eval types
#         k = None
#         if "tpp" in eval_type:
#             k = 10
#         elif "scr" in eval_type:
#             k = 10
#         elif "sparse_probing" in eval_type:
#             k = 1

#         metric_key, _ = get_custom_metric_key_and_name(eval_type, k)
#         print(f"Using metric key: {metric_key}")

#         if "tpp" in eval_type:
#             return data["eval_result_metrics"]["tpp_metrics"][metric_key]
#         elif "scr" in eval_type:
#             return data["eval_result_metrics"]["scr_metrics"][metric_key]
#         elif "absorption" in eval_type:
#             return data["eval_result_metrics"]["mean"][metric_key]
#         elif "autointerp" in eval_type:
#             return data["eval_result_metrics"]["autointerp"][metric_key]
#         elif "sparse_probing" in eval_type:
#             return data["eval_result_metrics"]["sae"][metric_key]
#         elif "unlearning" in eval_type:
#             return data["eval_result_metrics"]["unlearning"][metric_key]
#         else:
#             raise ValueError(f"Unknown eval type: {eval_type}")
#     except Exception as e:
#         print(f"Error processing {json_path}: {str(e)}")
#         raise


# def get_filename_for_arch(base_path: str, arch_pattern: str, eval_type: str) -> str:
#     """Get the filename for a specific architecture and evaluation type."""
#     if arch_pattern == "scope-2b-pt-res":
#         return f"{base_path}/{eval_type}/gemma-scope-2b-pt-res/gemma-scope-2b-pt-res_layer_12_width_16k_average_l0_41_eval_results.json"
#     elif arch_pattern == "vanilla":
#         return f"{base_path}/{eval_type}/sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109/sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_3_eval_results.json"
#     elif arch_pattern == "topk":
#         return f"{base_path}/{eval_type}/sae_bench_gemma-2-2b_topk_width-2pow16_date-1109/sae_bench_gemma-2-2b_topk_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_1_eval_results.json"
#     else:
#         raise ValueError(f"Unknown architecture pattern: {arch_pattern}")


# def get_scores_for_models(base_path: str) -> Dict[str, Dict[str, float]]:
#     """Get evaluation scores for the specified models."""
#     eval_types = ["absorption", "autointerp", "core", "scr", "sparse_probing", "tpp", "unlearning"]
#     arch_patterns = ["vanilla", "topk", "scope-2b-pt-res"]
#     scores = {eval_type: {} for eval_type in eval_types}

#     print("\nCollecting scores for specified models:")
#     for arch_pattern in arch_patterns:
#         print(f"\nProcessing architecture: {arch_pattern}")

#         for eval_type in eval_types:
#             json_path = get_filename_for_arch(base_path, arch_pattern, eval_type)
#             print(f"Looking for {eval_type} evaluation at: {json_path}")

#             if os.path.exists(json_path):
#                 try:
#                     score = get_metric_from_json(json_path, eval_type)
#                     scores[eval_type][arch_pattern] = score
#                     print(f"Found score: {score}")
#                 except Exception as e:
#                     print(f"Error getting score: {str(e)}")
#             else:
#                 print(f"Evaluation file not found: {json_path}")

#     return scores


# def plot_evaluation_results(scores: Dict[str, Dict[str, float]]):
#     """Create a grouped bar plot of evaluation results."""
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Add missing unlearning values
#     if "unlearning" in scores:
#         scores["unlearning"]["vanilla"] = 0.0125
#         scores["unlearning"]["topk"] = 0.125

#     # Set up the plot
#     fig, ax = plt.subplots(figsize=(12, 6))

#     # Define the metrics to plot and architectures
#     metrics = ["absorption", "autointerp", "sparse_probing", "scr", "tpp", "unlearning"]
#     architectures = ["vanilla", "topk", "scope-2b-pt-res"]

#     # Define colors for each architecture
#     colors = ["#2ecc71", "#3498db", "#e74c3c"]

#     # Set the width of each bar and positions of the bars
#     width = 0.25
#     x = np.arange(len(metrics))

#     # Plot bars for each architecture
#     for i, (arch, color) in enumerate(zip(architectures, colors)):
#         values = [scores[metric][arch] if arch in scores[metric] else 0 for metric in metrics]
#         ax.bar(x + i * width - width, values, width, label=arch, color=color, alpha=0.8)

#     # Customize the plot
#     ax.set_ylabel("Score", fontsize=12)
#     ax.set_title("SAE Evaluation Results by Architecture\nGemma-2-2B, Layer 12, 16k width\n, Sparsities: vanilla:57, topk:40, jumprelu:41", fontsize=14, pad=20)
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics, rotation=45, ha="right")
#     ax.legend(frameon=True)
#     ax.set_ylim(0, 1)

#     # Add grid
#     ax.yaxis.grid(True, linestyle="--", alpha=0.7)
#     ax.set_axisbelow(True)

#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()

#     return fig


# def main(base_path: str):
#     print(f"Starting analysis with base path: {base_path}")

#     # Get scores for specified models
#     scores = get_scores_for_models(base_path)

#     # Print scores
#     print("\nScores for selected models:")
#     for eval_type, arch_scores in scores.items():
#         print(f"\n{eval_type.upper()}:")
#         for arch, score in arch_scores.items():
#             print(f"{arch}: {score:.4f}")

#     # Print final summary of selected models
#     print("\n" + "=" * 50)
#     print("FINAL SUMMARY OF SELECTED MODELS:")
#     print("=" * 50)
#     model_names = {
#         "vanilla": "sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_3",
#         "topk": "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_1",
#         "scope-2b-pt-res": "gemma-scope-2b-pt-res_layer_12_width_16k_average_l0_41",
#     }

#     for arch, model_name in model_names.items():
#         print(f"\n{arch}:")
#         print(f"Selected model: {model_name}")
#         if "core" in scores and arch in scores["core"]:
#             print(f"L0 sparsity: {scores['core'][arch]:.2f}%")

#     # Create and save the plot
#     fig = plot_evaluation_results(scores)
#     plt.savefig("sae_evaluation_results.png", dpi=300, bbox_inches="tight")
#     plt.close()


# if __name__ == "__main__":
#     base_path = "graphing_eval_results"  # Adjust this path as needed
#     main(base_path)


import json
import os
from typing import Dict, Optional, Tuple
import glob
from pathlib import Path

def get_custom_metric_key_and_name(eval_path: str, k: Optional[int] = None) -> tuple[str, str]:
    """Get the metric key and name based on evaluation type."""
    if "tpp" in eval_path:
        return f"tpp_threshold_{k}_total_metric", f"TPP Top {k} Metric"
    elif "scr" in eval_path:
        return f"scr_metric_threshold_{k}", f"SCR Top {k} Metric"
    elif "sparse_probing" in eval_path:
        return f"sae_top_{k}_test_accuracy", f"Sparse Probing Top {k} Test Accuracy"
    elif "absorption" in eval_path:
        return "mean_absorption_score", "Mean Absorption Score"
    elif "autointerp" in eval_path:
        return "autointerp_score", "Autointerp Score"
    elif "unlearning" in eval_path:
        return "unlearning_score", "Unlearning Score"
    elif "core" in eval_path:
        return "frac_recovered", "Loss Recovered"
    else:
        raise ValueError("Please add the correct key for the custom metric")

def get_metric_from_json(json_path: str, eval_type: str) -> float:
    """Extract the relevant metric from a JSON file based on eval type."""
    print(f"Reading metric from {json_path} for eval_type {eval_type}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if "core" in eval_type:
            sparsity = data["eval_result_metrics"]["sparsity"]["l0"]
            print(f"Found L0 sparsity value: {sparsity}")
            return sparsity
            
        # Use specific thresholds for certain eval types
        k = None
        if "tpp" in eval_type:
            k = 10
        elif "scr" in eval_type:
            k = 10
        elif "sparse_probing" in eval_type:
            k = 1
            
        metric_key, _ = get_custom_metric_key_and_name(eval_type, k)
        print(f"Using metric key: {metric_key}")
        
        if "tpp" in eval_type:
            return data["eval_result_metrics"]["tpp_metrics"][metric_key]
        elif "scr" in eval_type:
            return data["eval_result_metrics"]["scr_metrics"][metric_key]
        elif "absorption" in eval_type:
            return data["eval_result_metrics"]["mean"][metric_key]
        elif "autointerp" in eval_type:
            return data["eval_result_metrics"]["autointerp"][metric_key]
        elif "sparse_probing" in eval_type:
            return data["eval_result_metrics"]["sae"][metric_key]
        elif "unlearning" in eval_type:
            return data["eval_result_metrics"]["unlearning"][metric_key]
        else:
            raise ValueError(f"Unknown eval type: {eval_type}")
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        raise

def get_filename_for_arch(base_path: str, arch_pattern: str, eval_type: str) -> str:
    """Get the filename for a specific architecture and evaluation type."""
    if arch_pattern == 'scope-2b-pt-res':
        return f"{base_path}/{eval_type}/gemma-scope-2b-pt-res/gemma-scope-2b-pt-res_layer_12_width_65k_average_l0_72_eval_results.json"
    elif arch_pattern == 'vanilla':
        return f"{base_path}/{eval_type}/sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109/sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_3_eval_results.json"
    elif arch_pattern == 'topk':
        return f"{base_path}/{eval_type}/sae_bench_gemma-2-2b_topk_width-2pow16_date-1109/sae_bench_gemma-2-2b_topk_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_2_eval_results.json"
    else:
        raise ValueError(f"Unknown architecture pattern: {arch_pattern}")


def get_scores_for_models(base_path: str) -> Dict[str, Dict[str, float]]:
    """Get evaluation scores for the specified models."""
    eval_types = ['absorption', 'autointerp', 'core', 'scr', 'sparse_probing', 'tpp', 'unlearning']
    arch_patterns = ['vanilla', 'topk', 'scope-2b-pt-res']
    scores = {eval_type: {} for eval_type in eval_types}
    
    print("\nCollecting scores for specified models:")
    for arch_pattern in arch_patterns:
        print(f"\nProcessing architecture: {arch_pattern}")
        
        for eval_type in eval_types:
            json_path = get_filename_for_arch(base_path, arch_pattern, eval_type)
            print(f"Looking for {eval_type} evaluation at: {json_path}")
            
            if os.path.exists(json_path):
                try:
                    score = get_metric_from_json(json_path, eval_type)
                    scores[eval_type][arch_pattern] = score
                    print(f"Found score: {score}")
                except Exception as e:
                    print(f"Error getting score: {str(e)}")
            else:
                print(f"Evaluation file not found: {json_path}")
                
    return scores

def plot_evaluation_results(scores: Dict[str, Dict[str, float]]):
    """Create a grouped bar plot of evaluation results."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Add missing unlearning values
    if 'unlearning' in scores:
        scores['unlearning']['vanilla'] = 0.0125
        scores['unlearning']['topk'] = 0.125
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the metrics to plot and architectures
    metrics = ['absorption', 'autointerp', 'sparse_probing', 'scr', 'tpp', 'unlearning']
    architectures = ['vanilla', 'topk', 'scope-2b-pt-res']
    
    # Define colors for each architecture
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Set the width of each bar and positions of the bars
    width = 0.25
    x = np.arange(len(metrics))
    
    # Plot bars for each architecture
    for i, (arch, color) in enumerate(zip(architectures, colors)):
        values = [scores[metric][arch] if arch in scores[metric] else 0 for metric in metrics]
        ax.bar(x + i*width - width, values, width, label=arch, color=color, alpha=0.8)
    
    # Customize the plot
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('SAE Evaluation Results by Architecture\nGemma-2-2B, Layer 12, 65k width\n, Sparsities: vanilla:72, topk:80, jumprelu:72"', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(frameon=True)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def main(base_path: str):
    print(f"Starting analysis with base path: {base_path}")
    
    # Get scores for specified models
    scores = get_scores_for_models(base_path)
    
    # Print scores
    print("\nScores for selected models:")
    for eval_type, arch_scores in scores.items():
        print(f"\n{eval_type.upper()}:")
        for arch, score in arch_scores.items():
            print(f"{arch}: {score:.4f}")
            
    # Print final summary of selected models
    print("\n" + "="*50)
    print("FINAL SUMMARY OF SELECTED MODELS:")
    print("="*50)
    model_names = {
        'vanilla': "sae_bench_gemma-2-2b_vanilla_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_3",
        'topk': "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109_blocks.12.hook_resid_post__trainer_2",
        'scope-2b-pt-res': "gemma-scope-2b-pt-res_layer_12_width_65k_average_l0_72"
    }
    
    for arch, model_name in model_names.items():
        print(f"\n{arch}:")
        print(f"Selected model: {model_name}")
        if 'core' in scores and arch in scores['core']:
            print(f"L0 sparsity: {scores['core'][arch]:.2f}%")
    
    # Create and save the plot
    fig = plot_evaluation_results(scores)
    plt.savefig('sae_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_path = "graphing_eval_results"  # Adjust this path as needed
    main(base_path)