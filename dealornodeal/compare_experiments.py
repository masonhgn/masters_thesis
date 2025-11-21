"""
Script to compare statistics across different algorithm runs.
Generates plots comparing wall clock times, convergence metrics, and other statistics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define the experiment directories to compare
EXPERIMENTS = {
    "CFR": "output/cfr_experiment_20251117_153225",
    "EFR (TIPS)": "output/efr_tips_experiment_20251117_235453",
    "EFR (CSPS)": "output/efr_csps_experiment_20251118_001551",
    "MCCFR (1k)": "output/mccfr_experiment_20251118_025429",
}

def load_experiment_data(experiment_path):
    """Load experiment data from JSON file."""
    json_path = Path(experiment_path) / "experiment_data.json"
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_wall_clock_comparison():
    """Generate bar chart comparing wall clock times across algorithms."""
    algorithms = []
    wall_times = []

    for algo_name, exp_path in EXPERIMENTS.items():
        data = load_experiment_data(exp_path)
        algorithms.append(algo_name)
        wall_times.append(data['timing']['wall_clock_seconds'])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algorithms, wall_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Wall Clock Time (seconds)', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('Wall Clock Time Comparison (1000 iterations)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('output/wall_clock_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: output/wall_clock_comparison.png")
    plt.close()

def plot_iterations_per_second_comparison():
    """Generate bar chart comparing iterations per second across algorithms."""
    algorithms = []
    iter_per_sec = []

    for algo_name, exp_path in EXPERIMENTS.items():
        data = load_experiment_data(exp_path)
        algorithms.append(algo_name)
        iter_per_sec.append(data['timing']['iterations_per_second'])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algorithms, iter_per_sec, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Iterations per Second', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('Iterations per Second Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')  # Use log scale due to large differences

    plt.tight_layout()
    plt.savefig('output/iterations_per_second_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: output/iterations_per_second_comparison.png")
    plt.close()

def plot_final_metrics_comparison():
    """Generate grouped bar chart comparing final convergence metrics."""
    algorithms = []
    cumulative_regret = []
    policy_delta = []
    total_variance = []

    for algo_name, exp_path in EXPERIMENTS.items():
        data = load_experiment_data(exp_path)
        algorithms.append(algo_name)
        cumulative_regret.append(data['final_metrics']['cumulative_regret'])
        policy_delta.append(data['final_metrics']['policy_delta'])
        total_variance.append(data['final_metrics']['total_variance'])

    # Create grouped bar chart
    x = np.arange(len(algorithms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, cumulative_regret, width, label='Cumulative Regret', color='#1f77b4')
    bars2 = ax.bar(x, policy_delta, width, label='Policy Delta', color='#ff7f0e')
    bars3 = ax.bar(x + width, total_variance, width, label='Total Variance', color='#2ca02c')

    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('Final Convergence Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('output/final_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: output/final_metrics_comparison.png")
    plt.close()

def plot_social_welfare_comparison():
    """Generate bar chart comparing final social welfare."""
    algorithms = []
    social_welfare = []

    for algo_name, exp_path in EXPERIMENTS.items():
        data = load_experiment_data(exp_path)
        algorithms.append(algo_name)
        social_welfare.append(data['final_metrics']['social_welfare'])

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(algorithms, social_welfare, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Social Welfare', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('Final Social Welfare Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('output/social_welfare_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: output/social_welfare_comparison.png")
    plt.close()

def print_summary_statistics():
    """Print summary statistics for all experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("="*80)

    for algo_name, exp_path in EXPERIMENTS.items():
        data = load_experiment_data(exp_path)
        print(f"\n{algo_name}:")
        print(f"  Wall Clock Time: {data['timing']['wall_clock_seconds']:.2f} seconds")
        print(f"  Iterations/Second: {data['timing']['iterations_per_second']:.2f}")
        print(f"  Final Cumulative Regret: {data['final_metrics']['cumulative_regret']:.6f}")
        print(f"  Final Policy Delta: {data['final_metrics']['policy_delta']:.4f}")
        print(f"  Final Social Welfare: {data['final_metrics']['social_welfare']:.4f}")
        print(f"  Final Total Variance: {data['final_metrics']['total_variance']:.4f}")
        print(f"  Strategy Convergence: {data['convergence_assessment']['strategy_convergence']}")
        print(f"  Policy Stability: {data['convergence_assessment']['policy_stability']}")

if __name__ == "__main__":
    print("Generating experiment comparison plots...")

    # Generate all plots
    plot_wall_clock_comparison()
    plot_iterations_per_second_comparison()
    plot_final_metrics_comparison()
    plot_social_welfare_comparison()

    # Print summary statistics
    print_summary_statistics()

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)
