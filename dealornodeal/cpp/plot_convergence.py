import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# directory containing the text files
results_dir = Path("experiment_results")

# groupings by equilibrium type
# each entry is the base filename (without _seedN.txt)
equilibrium_groups = {
    "EFCCE": ["efcce_cfr", "efcce_efr_csps", "efcce_efr_tips"],
    "EFCE": ["efce_cfr", "efce_cfr_in", "efce_efr_tips_in", "efce_efr_csps"],
}

# distinct color for every algorithm variant that appears in experiments
algorithm_colors = {
    "CFR": "#000000",                # black
    "CFR (internal)": "#666666",     # gray
    "EFR_ACT": "#FF6600",            # bright orange
    "EFR_ACT (internal)": "#CC4400", # dark orange
    "EFR_BHV": "#FF0000",            # bright red
    "EFR_BHV (internal)": "#AA0000", # dark red
    "EFR_CSPS": "#9933FF",           # bright purple
    "EFR_TIPS (external)": "#00CCCC",# bright cyan/teal
    "EFR_TIPS (internal)": "#007777",# dark teal
    "EFR_TIPS": "#00CCCC",           # bright cyan/teal
    "EFR_CFPS": "#FF00FF",           # bright magenta
    "EFR_CFPS (internal)": "#AA00AA",# dark magenta
}


def format_algorithm_name(algorithm_name):
    """
    format algorithm name to explicitly label external vs internal sampling.

    handles _in (internal) and _ex (external) suffixes in algorithm names.
    """
    if '_in ' in algorithm_name or algorithm_name.endswith('_in'):
        return algorithm_name.replace('_in', ' (internal)')
    elif '_ex ' in algorithm_name or algorithm_name.endswith('_ex'):
        return algorithm_name.replace('_ex', ' (external)')
    else:
        if algorithm_name.startswith('CFR '):
            return algorithm_name
        else:
            parts = algorithm_name.split(' ', 1)
            if len(parts) == 2:
                return f"{parts[0]} {parts[1]}"
            else:
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
                continue

    return algorithm_name, iterations, distances


def find_seed_files(base_name):
    """
    find all seed files for a given base experiment name.

    looks for files matching {base_name}_seed{N}.txt, falls back to
    {base_name}.txt if no seed files exist (backwards compatibility).
    """
    seed_files = sorted(results_dir.glob(f"{base_name}_seed*.txt"))
    if seed_files:
        return seed_files
    # fall back to single file
    single = results_dir / f"{base_name}.txt"
    if single.exists():
        return [single]
    return []


def get_algorithm_style(algorithm_name):
    """
    determine color and line style for an algorithm based on its name.

    looks up the formatted name (e.g. "CFR (internal)") directly in the
    color map. strips the equilibrium suffix first.
    """
    # remove equilibrium suffix like " EFCE" or " EFCCE"
    name = algorithm_name
    for suffix in [" EFCE", " EFCCE", " AFCE", " AFCCE", " CCE", " CE"]:
        name = name.replace(suffix, "")
    name = name.strip()

    if "(internal)" in name:
        style = "--"
    elif "(external)" in name:
        style = "-."
    else:
        style = "-"

    color = algorithm_colors.get(name, "#000000")

    return color, style


def plot_equilibrium_group(equilibrium_name, base_names, output_dir):
    """
    create a plot for a single equilibrium type with confidence intervals.
    """
    plt.figure(figsize=(10, 8))

    for base_name in base_names:
        seed_files = find_seed_files(base_name)

        if not seed_files:
            print(f"  warning: no files found for {base_name}, skipping")
            continue

        # read all seeds
        all_distances = []
        algo_name = None
        common_iterations = None

        for filepath in seed_files:
            name, iterations, distances = read_convergence_file(filepath)
            if algo_name is None:
                algo_name = name
                common_iterations = np.array(iterations)
            all_distances.append(distances)

        if not all_distances:
            continue

        # align to shortest run
        min_len = min(len(d) for d in all_distances)
        all_distances = np.array([d[:min_len] for d in all_distances])
        iterations = common_iterations[:min_len]

        mean = np.mean(all_distances, axis=0)
        formatted_name = format_algorithm_name(algo_name)
        color, linestyle = get_algorithm_style(formatted_name)

        num_seeds = len(seed_files)
        if num_seeds > 1:
            std = np.std(all_distances, axis=0, ddof=1)
            # 95% confidence interval using t-distribution
            t_crit = stats.t.ppf(0.975, df=num_seeds - 1)
            ci = t_crit * std / np.sqrt(num_seeds)

            plt.plot(iterations, mean, label=formatted_name,
                     linewidth=1.5, color=color, linestyle=linestyle)
            plt.fill_between(iterations, mean - ci, mean + ci,
                             alpha=0.2, color=color)
        else:
            plt.plot(iterations, mean, label=formatted_name,
                     linewidth=1.5, color=color, linestyle=linestyle)

    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Equilibrium Distance (log scale)', fontsize=12)
    plt.title(f'{equilibrium_name} Convergence', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{equilibrium_name.lower().replace('-', '_')}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  saved: {output_path}")

    plt.close()


def main():
    output_dir = Path("convergence_plots")
    output_dir.mkdir(exist_ok=True)

    for equilibrium_name, base_names in equilibrium_groups.items():
        print(f"\nprocessing {equilibrium_name}...")
        plot_equilibrium_group(equilibrium_name, base_names, output_dir)

    print(f"\nall plots saved to {output_dir}/")

if __name__ == "__main__":
    main()
