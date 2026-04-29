"""
convergence plot generator for efcce/efce experiments.

usage:
  python plot_convergence.py <results_dir>

reads experiment result .txt files from <results_dir>, generates convergence
plots into <results_dir>/plots/.
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats


# distinct color for every algorithm variant
ALGORITHM_COLORS = {
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
    """format algorithm name to explicitly label external vs internal."""
    if '_in ' in algorithm_name or algorithm_name.endswith('_in'):
        return algorithm_name.replace('_in', ' (internal)')
    elif '_ex ' in algorithm_name or algorithm_name.endswith('_ex'):
        return algorithm_name.replace('_ex', ' (external)')
    else:
        return algorithm_name


def read_convergence_file(filepath):
    """read convergence data from a text file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

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


def find_seed_files(results_dir, base_name):
    """find all seed files, falling back to single file."""
    seed_files = sorted(results_dir.glob(f"{base_name}_seed*.txt"))
    if seed_files:
        return seed_files
    single = results_dir / f"{base_name}.txt"
    if single.exists():
        return [single]
    return []


def get_algorithm_style(algorithm_name):
    """determine color and line style based on formatted algorithm name."""
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

    color = ALGORITHM_COLORS.get(name, "#000000")
    return color, style


def discover_experiments(results_dir):
    """auto-discover experiment groups from result files."""
    groups = {}
    for f in sorted(results_dir.glob("*.txt")):
        name = f.stem
        # strip seed suffix
        base = name
        if "_seed" in name:
            base = name[:name.rfind("_seed")]

        # extract equilibrium type from prefix
        for eq in ["efcce", "efce", "afcce", "afce"]:
            if base.startswith(eq + "_"):
                eq_upper = eq.upper()
                if eq_upper not in groups:
                    groups[eq_upper] = set()
                groups[eq_upper].add(base)
                break

    return {k: sorted(v) for k, v in groups.items()}


def plot_equilibrium_group(equilibrium_name, base_names, results_dir, output_dir):
    """create a convergence plot for a single equilibrium type."""
    plt.figure(figsize=(10, 8))

    for base_name in base_names:
        seed_files = find_seed_files(results_dir, base_name)

        if not seed_files:
            print(f"  warning: no files found for {base_name}, skipping")
            continue

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

        min_len = min(len(d) for d in all_distances)
        all_distances = np.array([d[:min_len] for d in all_distances])
        iterations = common_iterations[:min_len]

        mean = np.mean(all_distances, axis=0)
        formatted_name = format_algorithm_name(algo_name)
        color, linestyle = get_algorithm_style(formatted_name)

        num_seeds = len(seed_files)
        if num_seeds > 1:
            std = np.std(all_distances, axis=0, ddof=1)
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

    output_path = output_dir / f"{equilibrium_name.lower()}_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_convergence.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"error: {results_dir} does not exist")
        sys.exit(1)

    output_dir = results_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    groups = discover_experiments(results_dir)
    if not groups:
        print(f"no experiment files found in {results_dir}")
        sys.exit(1)

    for equilibrium_name, base_names in groups.items():
        print(f"\nprocessing {equilibrium_name} ({len(base_names)} algorithms)...")
        plot_equilibrium_group(equilibrium_name, base_names, results_dir, output_dir)

    print(f"\nall plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
