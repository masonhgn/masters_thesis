#!/usr/bin/env python3
"""
Plot tool to aggregate and visualize cumulative regret from all experiments.

This script scans all experiment folders in the output directory,
extracts cumulative regret data from experiment_data.json files,
and creates a combined plot showing all experiments together.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_experiment_data(output_dir: Path):
    """
    Load cumulative regret data from all experiment folders.

    Args:
        output_dir: Path to the output directory containing experiment folders

    Returns:
        Dictionary mapping experiment names to their data
    """
    experiments = {}

    # Iterate through all subdirectories in output/
    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue

        # Look for experiment_data.json in each folder
        json_path = folder / "experiment_data.json"
        if not json_path.exists():
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract the relevant data
            iterations = data['iteration_history']['iterations']
            cumulative_regret = data['iteration_history']['cumulative_regret']

            # Get algorithm metadata from new structure
            algo_name = data['algo_metadata']['algo_name']
            algo_longname = data['algo_metadata']['algo_longname']

            # Create a readable experiment name
            experiment_name = folder.name

            # Store the data
            experiments[experiment_name] = {
                'iterations': iterations,
                'cumulative_regret': cumulative_regret,
                'algo_name': algo_name,
                'algo_longname': algo_longname,
                'final_regret': cumulative_regret[-1] if cumulative_regret else None
            }

            print(f"Loaded: {experiment_name} ({algo_longname})")

        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}")
            continue

    return experiments


def create_combined_plot(experiments: dict, output_path: Path):
    """
    Create a combined plot of cumulative regret for all experiments.

    Args:
        experiments: Dictionary of experiment data
        output_path: Where to save the output plot
    """
    if not experiments:
        print("No experiments found to plot!")
        return

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Color map for different algorithms
    algorithm_colors = {
        'cfr': 'blue',
        'efr_tips': 'red',
        'efr_csps': 'orange',
        'mccfr': 'green',
        'unknown': 'purple'
    }

    # Plot each experiment
    for exp_name, data in experiments.items():
        iterations = data['iterations']
        cumulative_regret = data['cumulative_regret']
        algo_name = data['algo_name']
        algo_longname = data['algo_longname']

        # Get color based on algorithm short name
        color = algorithm_colors.get(algo_name.lower(), 'gray')

        # Create label with long algorithm name and final regret value
        label = f"{algo_longname} (final={data['final_regret']:.6f})"

        # Plot the data with marker style matching test_cfr_metrics
        plt.plot(iterations, cumulative_regret,
                'o-',
                color=color,
                label=label,
                linewidth=1,
                markersize=2,
                alpha=0.8)

    # Customize the plot
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title('Cumulative Regret Convergence Across All Experiments', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Collect all regret values to determine range
    all_regrets = []
    for data in experiments.values():
        all_regrets.extend(data['cumulative_regret'])

    # Find min and max across all experiments
    min_regret = min(all_regrets)
    max_regret = max(all_regrets)

    # Use log scale for better visibility of small differences
    if max_regret > 0:
        plt.yscale('log')

    # Set y-limits to stretch the plot slightly more to show gaps between lines
    # Use smaller margins than test_cfr_metrics to make it more compact but still visible
    if min_regret > 0:
        y_min = min_regret * 0.8  # 20% margin below
        y_max = max_regret * 1.2  # 20% margin above
        plt.ylim(y_min, y_max)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.close()


def main():
    """Main entry point for the plot tool."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "output"

    if not output_dir.exists():
        print(f"Error: Output directory not found at {output_dir}")
        return

    print(f"Scanning for experiments in: {output_dir}\n")

    # Load all experiment data
    experiments = load_experiment_data(output_dir)

    if not experiments:
        print("No valid experiments found!")
        return

    print(f"\nFound {len(experiments)} experiments")

    # Create the combined plot
    output_path = output_dir / "combined_cumulative_regret.png"
    create_combined_plot(experiments, output_path)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for exp_name, data in sorted(experiments.items(), key=lambda x: x[1]['final_regret'] or float('inf')):
        print(f"{exp_name:50s} | Final Regret: {data['final_regret']:.8f}")


if __name__ == "__main__":
    main()