import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def load_core_data(core_path: str, trainer_ids: List[int]) -> Dict[str, Dict]:
    """Load and process core evaluation data for specific trainer IDs."""
    core_data = {}
    for trainer_id in trainer_ids:
        filename = (
            f"gemma-2-2b_layer_4_additivity_trainer_{trainer_id}_custom_sae_eval_results.json"
        )
        filepath = os.path.join(core_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                core_data[trainer_id] = {
                    "mse": data["eval_result_metrics"]["reconstruction_quality"]["mse"],
                    "l0": data["eval_result_metrics"]["sparsity"]["l0"],
                }
    print("Loaded core data:", core_data)
    return core_data


def load_scr_data(scr_path: str, trainer_ids: List[int]) -> Dict[str, float]:
    """Load and process SCR evaluation data for specific trainer IDs."""
    scr_data = {}
    for trainer_id in trainer_ids:
        filename = (
            f"gemma-2-2b_layer_4_additivity_trainer_{trainer_id}_custom_sae_eval_results.json"
        )
        filepath = os.path.join(scr_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                scr_data[trainer_id] = data["eval_result_metrics"]["scr_metrics"][
                    "scr_metric_threshold_10"
                ]
    print("Loaded SCR data:", scr_data)
    return scr_data


def load_tpp_data(tpp_path: str, trainer_ids: List[int]) -> Dict[str, float]:
    """Load and process TPP evaluation data for specific trainer IDs."""
    tpp_data = {}
    for trainer_id in trainer_ids:
        filename = (
            f"gemma-2-2b_layer_4_additivity_trainer_{trainer_id}_custom_sae_eval_results.json"
        )
        filepath = os.path.join(tpp_path, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                tpp_data[trainer_id] = data["eval_result_metrics"]["tpp_metrics"][
                    "tpp_threshold_10_total_metric"
                ]
    print("Loaded TPP data:", tpp_data)
    return tpp_data


def get_trainer_class(trainer_id: int) -> Tuple[int, str]:
    """Get trainer class and number from trainer ID."""
    class_num = trainer_id % 4
    class_names = ["default", "intersect", "add", "intersect + add"]
    return class_num, class_names[class_num]


def create_plots(
    core_data: Dict, 
    absorption_data: Optional[Dict] = None, 
    scr_data: Optional[Dict] = None,
    tpp_data: Optional[Dict] = None
):
    """Create and save the required plots."""
    # Set up colors for each class
    colors = ["blue", "red", "green", "purple"]
    class_names = ["default", "intersect", "add", "intersect + add"]

    # Calculate number of plots needed
    num_plots = 1 + (absorption_data is not None) + (scr_data is not None) + (tpp_data is not None)

    # Disable scientific notation
    # plt.rcParams["axes.formatter.use_scientific"] = False

    # Create figure with appropriate number of subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    # Organize data by class
    class_data = {
        i: {"l0": [], "mse": [], "absorption": [], "scr": [], "tpp": [], "trainer_ids": []} 
        for i in range(4)
    }

    for trainer_id in core_data:
        class_num, _ = get_trainer_class(trainer_id)
        class_data[class_num]["l0"].append(core_data[trainer_id]["l0"])
        class_data[class_num]["mse"].append(core_data[trainer_id]["mse"])
        class_data[class_num]["trainer_ids"].append(trainer_id)
        if absorption_data and trainer_id in absorption_data:
            class_data[class_num]["absorption"].append(absorption_data[trainer_id])
        if scr_data and trainer_id in scr_data:
            class_data[class_num]["scr"].append(scr_data[trainer_id])
        if tpp_data and trainer_id in tpp_data:
            class_data[class_num]["tpp"].append(tpp_data[trainer_id])

    print("Organized data:", class_data)

    # Plot data points for each class
    for class_num in range(4):
        if class_data[class_num]["l0"]:  # Only plot if there's data for this class
            print(f"Plotting class {class_num}")
            print(f"L0 values: {class_data[class_num]['l0']}")
            print(f"MSE values: {class_data[class_num]['mse']}")
            print(f"SCR values: {class_data[class_num]['scr']}")
            print(f"TPP values: {class_data[class_num]['tpp']}")

            # MSE vs L0 plot
            axes[0].scatter(
                class_data[class_num]["l0"],
                class_data[class_num]["mse"],
                color=colors[class_num],
                alpha=0.7,
                label=class_names[class_num],
                s=100,
                zorder=5,
            )

            current_plot = 1

            # SCR vs L0 plot (if enabled)
            if scr_data is not None:
                axes[current_plot].scatter(
                    class_data[class_num]["l0"],
                    class_data[class_num]["scr"],
                    color=colors[class_num],
                    alpha=0.7,
                    label=class_names[class_num],
                    s=100,
                    zorder=5,
                )
                current_plot += 1

            # TPP vs L0 plot (if enabled)
            if tpp_data is not None:
                axes[current_plot].scatter(
                    class_data[class_num]["l0"],
                    class_data[class_num]["tpp"],
                    color=colors[class_num],
                    alpha=0.7,
                    label=class_names[class_num],
                    s=100,
                    zorder=5,
                )
                current_plot += 1

    # Configure plots
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3)

    # Set axis labels and titles
    axes[0].set_xlabel("L0 Sparsity")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE vs L0 Sparsity")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys())

    # Fine-tune x-axis limits with some padding
    min_l0 = min([min(data["l0"]) for data in class_data.values() if data["l0"]])
    max_l0 = max([max(data["l0"]) for data in class_data.values() if data["l0"]])
    padding = (max_l0 - min_l0) * 0.05  # 5% padding
    for ax in axes:
        ax.set_xlim(min_l0 - padding, max_l0 + padding)

    current_plot = 1

    if scr_data is not None:
        axes[current_plot].set_xlabel("L0 Sparsity")
        axes[current_plot].set_ylabel("SCR Score (threshold 10)")
        axes[current_plot].set_title("SCR Score vs L0 Sparsity")
        handles, labels = axes[current_plot].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[current_plot].legend(by_label.values(), by_label.keys())
        current_plot += 1

    if tpp_data is not None:
        axes[current_plot].set_xlabel("L0 Sparsity")
        axes[current_plot].set_ylabel("TPP Score (threshold 10)")
        axes[current_plot].set_title("TPP Score vs L0 Sparsity")
        handles, labels = axes[current_plot].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[current_plot].legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig("sae_analysis_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def main(
    core_path: str,
    eval_path: str,
    trainer_ids: List[int],
    plot_absorption: bool = True,
    plot_scr: bool = False,
    plot_tpp: bool = False,
    scr_path: Optional[str] = None,
    tpp_path: Optional[str] = None,
):
    """Main function to run the analysis and create plots."""
    # Load data
    core_data = load_core_data(core_path, trainer_ids)
    absorption_data = load_absorption_data(eval_path, trainer_ids) if plot_absorption else None
    scr_data = load_scr_data(scr_path, trainer_ids) if plot_scr and scr_path else None
    tpp_data = load_tpp_data(tpp_path, trainer_ids) if plot_tpp and tpp_path else None

    # Create plots
    create_plots(core_data, absorption_data, scr_data, tpp_data)
    print("Analysis complete. Plots saved as 'sae_analysis_plots.png'")


if __name__ == "__main__":
    # Define the specific trainer IDs you want to analyze
    trainer_ids = list(range(4))  # Modify this list as needed

    main(
        core_path="./eval_results/core",
        eval_path="./eval_results/absorption",
        trainer_ids=trainer_ids,
        plot_absorption=False,  # Set to False to only show mse plot
        plot_scr=True,  # Set to True to show SCR plot
        plot_tpp=True,  # Set to True to show TPP plot
        scr_path="./eval_results/scr/scr",  # Path to SCR evaluation results
        tpp_path="./eval_results/tpp/tpp",  # Path to TPP evaluation results
    )