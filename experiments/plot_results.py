"""
plot convergence curves from an experiment run directory.

reads all result files from a run, groups by equilibrium type,
averages across seeds, and plots with confidence intervals.

usage:
  python3 experiments/plot_results.py experiments/runs/20260401_130634
  python3 experiments/plot_results.py  # auto-picks latest run
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import defaultdict

# consistent colors per algorithm across all plots
ALGO_COLORS = {
    "CFR":                "#000000",
    "CFR (internal)":     "#666666",
    "EFR_ACT":            "#FF6600",
    "EFR_ACT (internal)": "#CC4400",
    "EFR_CSPS":           "#9933FF",
    "EFR_CFPS (internal)":"#0066AA",
    "EFR_TIPS":           "#00CCCC",
    "EFR_TIPS (internal)":"#007777",
    "EFR_BHV":            "#FF0000",
    "EFR_BHV (internal)": "#AA0000",
}

# map filename prefixes to display names
DISPLAY_NAMES = {
    "cfr":          "CFR",
    "cfr_in":       "CFR (internal)",
    "efr_act":      "EFR_ACT",
    "efr_act_in":   "EFR_ACT (internal)",
    "efr_csps":     "EFR_CSPS",
    "efr_cfps_in":  "EFR_CFPS (internal)",
    "efr_tips":     "EFR_TIPS",
    "efr_tips_in":  "EFR_TIPS (internal)",
    "efr_bhv":      "EFR_BHV",
    "efr_bhv_in":   "EFR_BHV (internal)",
}

EQUILIBRIA = ["afcce", "afce", "efcce", "efce", "cce", "ce"]


def read_result_file(filepath):
    """read a result file, return (algo_label, iterations, distances)."""
    lines = filepath.read_text().strip().split("\n")
    algo_label = lines[0].strip()

    iterations = []
    distances = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                iterations.append(int(parts[0]))
                distances.append(float(parts[1]))
            except ValueError:
                continue

    return algo_label, np.array(iterations), np.array(distances)


def parse_filename(filename):
    """
    extract equilibrium, algorithm, max_turns, and seed from filename.

    new format: '{eq}_{algo}_t{turns}_seed{seed}'
      e.g. 'efce_efr_tips_in_t3_seed4' -> ('efce', 'efr_tips_in', 3, 4)
    """
    stem = filename.stem
    # split off seed
    parts = stem.rsplit("_seed", 1)
    if len(parts) != 2:
        return None, None, None, None
    base, seed_str = parts
    seed = int(seed_str)

    # split off turns suffix '_t{N}' if present (new format only)
    turns = None
    if "_t" in base:
        head, _, tail = base.rpartition("_t")
        try:
            turns = int(tail)
            base = head
        except ValueError:
            pass
    if turns is None:
        return None, None, None, None

    # split equilibrium prefix from algorithm name
    for eq in EQUILIBRIA:
        if base.startswith(eq + "_"):
            algo = base[len(eq) + 1:]
            return eq, algo, turns, seed

    return None, None, None, None


def find_latest_run():
    """find the most recent run directory."""
    runs_dir = Path("experiments/runs")
    if not runs_dir.exists():
        return None
    dirs = sorted(runs_dir.iterdir(), reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "progress.json").exists():
            return d
    return None


def plot_equilibrium(eq_name, turns, algo_data, output_dir):
    """
    plot convergence for one (equilibrium, max_turns) pair.

    algo_data: dict of algo_name -> list of (iterations, distances) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for algo_key in sorted(algo_data.keys()):
        runs = algo_data[algo_key]
        display = DISPLAY_NAMES.get(algo_key, algo_key)
        color = ALGO_COLORS.get(display, "#000000")

        # align to shortest run
        min_len = min(len(d) for _, d in runs)
        iterations = runs[0][0][:min_len]
        all_dists = np.array([d[:min_len] for _, d in runs])

        mean = np.mean(all_dists, axis=0)

        if len(runs) > 1:
            std = np.std(all_dists, axis=0, ddof=1)
            t_crit = stats.t.ppf(0.975, df=len(runs) - 1)
            ci = t_crit * std / np.sqrt(len(runs))
            ax.fill_between(iterations, mean - ci, mean + ci,
                            alpha=0.2, color=color)

        ax.plot(iterations, mean, label=display, linewidth=1.5, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Equilibrium Distance (log scale)", fontsize=12)
    ax.set_title(
        f"{eq_name.upper()} Convergence (max_turns={turns})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / f"t{turns}_{eq_name}_convergence.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        run_dir = Path(sys.argv[1])
    else:
        run_dir = find_latest_run()
        if not run_dir:
            print("no run directory found")
            sys.exit(1)

    print(f"plotting results from: {run_dir}")

    # group results by (turns, equilibrium, algorithm)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in sorted(run_dir.glob("*.txt")):
        eq, algo, turns, seed = parse_filename(f)
        if eq is None:
            continue
        _, iterations, distances = read_result_file(f)
        if len(iterations) > 0:
            grouped[turns][eq][algo].append((iterations, distances))

    if not grouped:
        print("no result files found")
        sys.exit(1)

    # create plots directory inside the run
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    for turns in sorted(grouped.keys()):
        for eq_name in EQUILIBRIA:
            if eq_name in grouped[turns]:
                plot_equilibrium(eq_name, turns, grouped[turns][eq_name], plot_dir)

    print(f"all plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
