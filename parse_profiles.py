#!/usr/bin/env python3
import os
import re

profile_dir = "experiment_results"
profile_files = [f for f in os.listdir(profile_dir) if f.endswith("_profile.txt")]

data = []

for filename in sorted(profile_files):
    filepath = os.path.join(profile_dir, filename)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # parse the file
    algo = None
    equil = None
    total_time = None
    percentages = {}

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("algorithm,"):
            algo = line.split(",")[1]
        elif line.startswith("equilibrium,"):
            equil = line.split(",")[1]
        elif line.startswith("total_experiment,"):
            total_time = float(line.split(",")[1])
        elif line.startswith("percentages"):
            # read next lines for percentages
            for j in range(i+1, len(lines)):
                pct_line = lines[j].strip()
                if not pct_line or pct_line == "":
                    break
                parts = pct_line.split(",")
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1].rstrip("%")
                    percentages[key] = float(value)
            break

    data.append({
        "file": filename,
        "algorithm": algo,
        "equilibrium": equil,
        "total_time_ms": total_time,
        "total_time_s": total_time / 1000.0,
        "percentages": percentages
    })

# group by equilibrium
by_equilibrium = {}
for d in data:
    eq = d["equilibrium"]
    if eq not in by_equilibrium:
        by_equilibrium[eq] = []
    by_equilibrium[eq].append(d)

# generate latex tables
print("=" * 80)
print("TABLE 1: Total Experiment Time")
print("=" * 80)
print()
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Total experiment time (seconds) for 10 iterations}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Algorithm & AFCCE & AFCE & EFCCE & EFCE \\\\")
print("\\midrule")

# organize by algorithm
algos_seen = {}
for d in data:
    algo = d["algorithm"]
    if algo not in algos_seen:
        algos_seen[algo] = {"AFCCE": None, "AFCE": None, "EFCCE": None, "EFCE": None}
    algos_seen[algo][d["equilibrium"]] = d["total_time_s"]

for algo in sorted(algos_seen.keys()):
    row_data = algos_seen[algo]
    row = f"{algo}"
    for eq in ["AFCCE", "AFCE", "EFCCE", "EFCE"]:
        if row_data[eq] is not None:
            row += f" & {row_data[eq]:.2f}"
        else:
            row += " & --"
    row += " \\\\"
    print(row)

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
print()
print()

# table 2: percentage breakdown
print("=" * 80)
print("TABLE 2: Time Breakdown (Percentage)")
print("=" * 80)
print()

function_order = ["cfr_update", "policy_extraction", "sampling", "get_corr_dev", "reset", "distance_calc"]
function_names = {
    "cfr_update": "CFR Update",
    "policy_extraction": "Policy Extract",
    "sampling": "Sampling",
    "get_corr_dev": "Get Corr Dev",
    "reset": "Reset",
    "distance_calc": "Distance Calc"
}

for eq in ["AFCCE", "AFCE", "EFCCE", "EFCE"]:
    if eq not in by_equilibrium:
        continue

    print(f"\\begin{{table}}[h]")
    print("\\centering")
    print(f"\\caption{{Time breakdown (\\%) for {eq}}}")
    print("\\begin{tabular}{l" + "c" * len(by_equilibrium[eq]) + "}")
    print("\\toprule")

    # header
    header = "Function"
    for d in by_equilibrium[eq]:
        header += f" & {d['algorithm']}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    # rows
    for func in function_order:
        row = function_names[func]
        for d in by_equilibrium[eq]:
            pct = d["percentages"].get(func, 0.0)
            row += f" & {pct:.2f}"
        row += " \\\\"
        print(row)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
