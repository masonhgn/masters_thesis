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
    "EFCE": ["efce_cfr.txt", "efce_cfr_in.txt", "efce_efr_cfps_in.txt", "efce_efr_tips_in.txt", "efce_efr_csps_in.txt", "efce_efr_bhv_in.txt"],
    "Zero-Sum": ["zerosum_cfr_cce.txt", "zerosum_cfr_ce.txt"]
}

# consistent color mapping for algorithms across all plots
# maps base algorithm name (before external/internal labels) to color
# using bright, high-contrast colors for better differentiation
algorithm_colors = {
    "CFR": "#0066FF",           # bright blue
    "CFR (CCE)": "#0066FF",     # bright blue (for zero-sum CCE)
    "CFR (CE)": "#00CC00",      # bright green (for zero-sum CE)
    "EFR_ACT": "#FF6600",       # bright orange
    "EFR_BHV": "#FF0000",       # bright red
    "EFR_CSPS": "#9933FF",      # bright purple
    "EFR_TIPS": "#00CCCC",      # bright cyan/teal
    "EFR_CFPS": "#FF00FF",      # bright magenta
}

# line styles for external vs internal sampling
line_styles = {
    "external": "-",      # solid line
    "internal": "--",     # dashed line
    "default": "-"        # solid line for CFR
}

def format_algorithm_name(algorithm_name):
    """
    format algorithm name to explicitly label external vs internal sampling.

    if the algorithm name doesn't contain '_in', add '(external)' suffix.
    if it contains '_in', replace '_in' with '(internal)'.
    """
    # check if this is an internal sampling algorithm
    if '_in' in algorithm_name:
        # replace _in with (internal)
        return algorithm_name.replace('_in', ' (internal)')
    else:
        # for external sampling, add (external) label
        # but skip CFR since it's the baseline
        if algorithm_name.startswith('CFR '):
            return algorithm_name
        else:
            # split on first space to insert (external) after algorithm name
            parts = algorithm_name.split(' ', 1)
            if len(parts) == 2:
                return f"{parts[0]} (external) {parts[1]}"
            else:
                return f"{algorithm_name} (external)"
    return algorithm_name

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

def get_algorithm_style(algorithm_name):
    """
    determine color and line style for an algorithm based on its name.

    returns tuple of (color, linestyle)
    """
    # extract base algorithm name (before external/internal label)
    base_name = algorithm_name.split(' (')[0]

    # determine if external or internal
    if '(internal)' in algorithm_name:
        style = line_styles["internal"]
    elif '(external)' in algorithm_name:
        style = line_styles["external"]
    else:
        style = line_styles["default"]

    # get color from mapping, default to black if not found
    color = algorithm_colors.get(base_name, "#000000")

    return color, style

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

        # format the algorithm name to show external/internal explicitly
        formatted_name = format_algorithm_name(algorithm_name)

        # get consistent color and line style for this algorithm
        color, linestyle = get_algorithm_style(formatted_name)

        # plot this algorithm's convergence with consistent styling
        plt.plot(iterations, distances, marker='o', markersize=2,
                 label=formatted_name, linewidth=1.5,
                 color=color, linestyle=linestyle)

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
