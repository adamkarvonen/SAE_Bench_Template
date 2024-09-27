import matplotlib.pyplot as plt
import json
import torch
import pickle
from matplotlib.colors import Normalize
import numpy as np
import os
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from collections import defaultdict


# “Gated SAE”, “Gated SAE w/ p-annealing”, “Standard”, “Standard w/ p-annealing”
label_lookup = {
    "StandardTrainer": "Standard",
    # "PAnnealTrainer": "Standard w/ p-annealing",
    # "GatedSAETrainer": "Gated SAE",
    "TrainerJumpRelu": "JumpReLU",
    # "GatedAnnealTrainer": "Gated SAE w/ p-annealing",
    "TrainerTopK": "Top K",
    # "Identity": "Identity",
}

unique_trainers = list(label_lookup.keys())

# create a dictionary mapping trainer types to marker shapes

trainer_markers = {
    "StandardTrainer": "o",
    "TrainerJumpRelu": "X",
    "TrainerTopK": "^",
    "GatedSAETrainer": "d",
}


# default text size
plt.rcParams.update({"font.size": 20})


def plot_3var_graph(
    results: dict[str, dict[str, float]],
    title: str,
    custom_metric: str,
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    colorbar_label: str = "Average Diff",
    output_filename: Optional[str] = None,
    legend_location: str = "lower right",
    x_axis_key: str = "l0",
    y_axis_key: str = "frac_recovered",
):
    # Extract data from results
    l0_values = [data[x_axis_key] for data in results.values()]
    frac_recovered_values = [data[y_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a normalize object for color scaling
    norm = Normalize(vmin=min(custom_metric_values), vmax=max(custom_metric_values))

    handles, labels = [], []

    for trainer, marker in trainer_markers.items():
        # Filter data for this trainer
        trainer_data = {k: v for k, v in results.items() if v["trainer_class"] == trainer}

        if not trainer_data:
            continue  # Skip this trainer if no data points

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        frac_recovered_values = [data[y_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        # Plot data points
        scatter = ax.scatter(
            l0_values,
            frac_recovered_values,
            c=custom_metric_values,
            cmap="viridis",
            marker=marker,
            s=100,
            label=label_lookup[trainer],
            norm=norm,
            edgecolor="black",
        )

        # custom legend stuff
        _handle, _ = scatter.legend_elements(prop="sizes")
        _handle[0].set_markeredgecolor("black")
        _handle[0].set_markerfacecolor("white")
        _handle[0].set_markersize(10)
        if marker == "d":
            _handle[0].set_markersize(13)
        handles += _handle
        labels.append(label_lookup[trainer])

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, label=colorbar_label)

    # Set labels and title
    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel("Loss Recovered (Fidelity)")
    ax.set_title(title)

    ax.legend(handles, labels, loc=legend_location)

    # Set axis limits
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    # Save and show the plot
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")
    plt.show()


def plot_interactive_3var_graph(
    results: dict[str, dict[str, float]],
    custom_color_metric: str,
    xlims: Optional[tuple[float, float]] = None,
    y_lims: Optional[tuple[float, float]] = None,
    output_filename: Optional[str] = None,
    x_axis_key: str = "l0",
    y_axis_key: str = "frac_recovered",
):
    # Extract data from results
    ae_paths = list(results.keys())
    l0_values = [data[x_axis_key] for data in results.values()]
    frac_recovered_values = [data[y_axis_key] for data in results.values()]

    custom_metric_value = [data[custom_color_metric] for data in results.values()]

    dict_size = [data["dict_size"] for data in results.values()]
    lr = [data["lr"] for data in results.values()]
    l1_penalty = [data["l1_penalty"] for data in results.values()]

    # Create the scatter plot
    fig = go.Figure()

    # Add trace
    fig.add_trace(
        go.Scatter(
            x=l0_values,
            y=frac_recovered_values,
            mode="markers",
            marker=dict(
                size=10,
                color=custom_metric_value,  # Color points based on frac_recovered
                colorscale="Viridis",  # You can change this colorscale
                showscale=True,
            ),
            text=[
                f"AE Path: {ae}<br>L0: {l0:.4f}<br>Frac Recovered: {fr:.4f}<br>Custom Metric: {ad:.4f}<br>Dict Size: {d:.4f}<br>LR: {l:.4f}<br>L1 Penalty: {l1:.4f}"
                for ae, l0, fr, ad, d, l, l1 in zip(
                    ae_paths,
                    l0_values,
                    frac_recovered_values,
                    custom_metric_value,
                    dict_size,
                    lr,
                    l1_penalty,
                )
            ],
            hoverinfo="text",
        )
    )

    # Update layout
    fig.update_layout(
        title="L0 vs Loss Recovered",
        xaxis_title="L0 (Sparsity)",
        yaxis_title="Loss Recovered (Fidelity)",
        hovermode="closest",
    )

    # Set axis limits
    if xlims:
        fig.update_xaxes(range=xlims)
    if y_lims:
        fig.update_yaxes(range=y_lims)

    # Save and show the plot
    if output_filename:
        fig.write_html(output_filename)

    fig.show()


def plot_2var_graph(
    results: dict[str, dict[str, float]],
    custom_metric: str,
    title: str = "L0 vs Custom Metric",
    y_label: str = "Custom Metric",
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    output_filename: Optional[str] = None,
    legend_location: str = "lower right",
    original_acc: Optional[float] = None,
    x_axis_key: str = "l0",
):
    # Extract data from results
    l0_values = [data[x_axis_key] for data in results.values()]
    custom_metric_values = [data[custom_metric] for data in results.values()]

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    handles, labels = [], []

    for trainer, marker in trainer_markers.items():
        # Filter data for this trainer
        trainer_data = {k: v for k, v in results.items() if v["trainer_class"] == trainer}

        if not trainer_data:
            continue  # Skip this trainer if no data points

        l0_values = [data[x_axis_key] for data in trainer_data.values()]
        custom_metric_values = [data[custom_metric] for data in trainer_data.values()]

        # Plot data points
        scatter = ax.scatter(
            l0_values,
            custom_metric_values,
            marker=marker,
            s=100,
            label=label_lookup[trainer],
            edgecolor="black",
        )

        # custom legend stuff
        _handle, _ = scatter.legend_elements(prop="sizes")
        _handle[0].set_markeredgecolor("black")
        _handle[0].set_markerfacecolor("white")
        _handle[0].set_markersize(10)
        if marker == "d":
            _handle[0].set_markersize(13)
        handles += _handle
        labels.append(label_lookup[trainer])

    # Set labels and title
    ax.set_xlabel("L0 (Sparsity)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if original_acc:
        ax.axhline(original_acc, color="red", linestyle="--", label="Original Probe Accuracy")

    ax.legend(handles, labels, loc=legend_location)

    # Set axis limits
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)

    plt.tight_layout()

    # Save and show the plot
    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")
    plt.show()


def plot_steps_vs_average_diff(
    results_dict: dict,
    steps_key: str = "steps",
    avg_diff_key: str = "average_diff",
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    output_filename: Optional[str] = None,
):
    # Initialize a defaultdict to store data for each trainer
    trainer_data = defaultdict(lambda: {"steps": [], "average_diffs": []})

    all_steps = set()

    # Extract data from the dictionary
    for key, value in results_dict.items():
        # Extract trainer number from the key
        trainer = key.split("/")[-1].split("_")[1]  # Assuming format like "trainer_1_..."
        layer = key.split("/")[-2].split("_")[-2]

        if "topk_ctx128" in key:
            trainer_type = "TopK SAE"
        elif "standard_ctx128" in key:
            trainer_type = "Standard SAE"
        else:
            raise ValueError(f"Unknown trainer type in key: {key}")

        step = int(value[steps_key])
        avg_diff = value[avg_diff_key]

        trainer_key = f"{trainer_type} Layer {layer} Trainer {trainer}"

        trainer_data[trainer_key]["steps"].append(step)
        trainer_data[trainer_key]["average_diffs"].append(avg_diff)
        all_steps.add(step)

    # Calculate average across all trainers
    average_trainer_data = {"steps": [], "average_diffs": []}
    for step in sorted(all_steps):
        step_diffs = []
        for data in trainer_data.values():
            if step in data["steps"]:
                idx = data["steps"].index(step)
                step_diffs.append(data["average_diffs"][idx])
        if step_diffs:
            average_trainer_data["steps"].append(step)
            average_trainer_data["average_diffs"].append(np.mean(step_diffs))

    # Add average_trainer_data to trainer_data
    trainer_data["Average"] = average_trainer_data

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot data for each trainer
    for trainer_key, data in trainer_data.items():
        steps = data["steps"]
        average_diffs = data["average_diffs"]

        # Sort the data by steps to ensure proper ordering
        sorted_data = sorted(zip(steps, average_diffs))
        steps, average_diffs = zip(*sorted_data)

        # Find the maximum step value for this trainer
        max_step = max(steps)

        # Convert steps to percentages of max_step
        step_percentages = [step / max_step * 100 for step in steps]

        # Plot the line for this trainer
        if trainer_key == "Average":
            plt.plot(
                step_percentages,
                average_diffs,
                marker="o",
                label=trainer_key,
                linewidth=3,
                color="red",
                zorder=10,
            )  # Emphasized average line
        else:
            plt.plot(
                step_percentages,
                average_diffs,
                marker="o",
                label=trainer_key,
                alpha=0.3,
                linewidth=1,
            )  # More transparent individual lines

    # log scale
    # plt.xscale("log")

    # if not title:
    #     title = f'{steps_key.capitalize()} vs {avg_diff_key.replace("_", " ").capitalize()}'

    if not y_label:
        y_label = avg_diff_key.replace("_", " ").capitalize()

    plt.xlabel("Training Progess (%)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)  # More transparent grid

    if len(trainer_data) < 50 and False:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, bbox_inches="tight")

    # Show the plot
    plt.show()
