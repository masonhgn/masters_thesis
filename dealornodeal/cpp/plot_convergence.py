import matplotlib.pyplot as plt
import os
from pathlib import Path

# directory containing the text files
results_dir = Path("experiment_results")

# groupings by equilibrium type
equilibrium_groups = {
    "AFCCE": ["afcce_cfr.txt", "afcce_efr_act.txt"],
    "AFCE": ["afce_cfr.txt", "afce_efr_act_in.txt"],
    "EFCCE": ["efcce_cfr.txt", "efcce_efr_bhv.txt", "efcce_efr_csps.txt", "efcce_efr_tips.txt"],
    "EFCE": ["efce_cfr.txt", "efce_efr_cfps_in.txt", "efce_efr_tips_in.txt"],
    "Zero-Sum": ["zerosum_cfr_cce.txt", "zerosum_cfr_ce.txt"]
}

def read_convergence_file(filepath):
    """
    read convergence data from a text file.

    returns tuple of (algorithm_name, iterations, distances)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # first line is the algorithm name
    algorithm_name = lines[0].strip()

    # parse data lines
    iterations = []
    distances = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 2:
            try:
                iterations.append(int(parts[0]))
                distances.append(float(parts[1]))
            except ValueError:
                # skip lines that don't parse correctly
                continue

    return algorithm_name, iterations, distances

def plot_equilibrium_group(equilibrium_name, file_list, output_dir):
    """
    create a plot for a single equilibrium type with all its algorithms.
    """
    plt.figure(figsize=(10, 8))

    for filename in file_list:
        filepath = results_dir / filename

        if not filepath.exists():
            print(f"warning: {filepath} not found, skipping")
            continue

        algorithm_name, iterations, distances = read_convergence_file(filepath)

        # plot this algorithm's convergence
        plt.plot(iterations, distances, marker='o', markersize=2,
                 label=algorithm_name, linewidth=1)

    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Equilibrium Distance (log scale)', fontsize=12)
    plt.title(f'{equilibrium_name} Convergence', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # save the plot
    output_path = output_dir / f"{equilibrium_name.lower().replace('-', '_')}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved: {output_path}")

    plt.close()

def main():
    # create output directory for plots
    output_dir = Path("convergence_plots")
    output_dir.mkdir(exist_ok=True)

    # create a plot for each equilibrium group
    for equilibrium_name, file_list in equilibrium_groups.items():
        print(f"\nprocessing {equilibrium_name}...")
        plot_equilibrium_group(equilibrium_name, file_list, output_dir)

    print(f"\nall plots saved to {output_dir}/")

if __name__ == "__main__":
    main()
