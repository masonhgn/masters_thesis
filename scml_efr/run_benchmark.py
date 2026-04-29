"""
phase 6: benchmark EFROneShotAgent against the stock SCML OneShot
baselines and produce a single bar chart of mean score per agent.

usage:
    python run_benchmark.py [--n-configs 10] [--n-steps 50] [--out plot.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scml.oneshot.agents import (
    GreedyOneShotAgent,
    RandDistOneShotAgent,
    EqualDistOneShotAgent,
    SyncRandomOneShotAgent,
)
from scml.utils import anac2024_oneshot

from efr_oneshot_agent import (
    EFR3PHybrid_NoAccept,
    EFR3PHybrid_NoFirst,
    EFR3PHybridAgent,
    EFRHybridAgentNoEFR,
)


SHORT_NAMES = {
    "EFRHybridAgentNoEFR": "hybrid (no EFR)",
    "EFR3PHybridAgent": "3p-hybrid full",
    "EFR3PHybrid_NoFirst": "3p (accept only)",
    "EFR3PHybrid_NoAccept": "3p (firstprop only)",
    "GreedyOneShotAgent": "Greedy",
    "RandDistOneShotAgent": "RandDist",
    "EqualDistOneShotAgent": "EqualDist",
    "SyncRandomOneShotAgent": "SyncRandom",
}


def short(agent_type: str) -> str:
    leaf = agent_type.split(".")[-1]
    return SHORT_NAMES.get(leaf, leaf)


def main() -> int:
    from _runlog import start as _start_log
    _start_log("benchmark")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=30)
    ap.add_argument("--n-runs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "benchmark.png")
    args = ap.parse_args()

    competitors = [
        # 3-player EFR ablation: does a real multi-player abstraction help?
        EFRHybridAgentNoEFR,     # control: pure winner pipeline, no EFR at all
        EFR3PHybridAgent,        # full: 3p seller joint offers + 3p buyer accept probs
        EFR3PHybrid_NoFirst,     # accept-probs only (first proposals = distribute_evenly)
        EFR3PHybrid_NoAccept,    # seller joint only (subset scoring = pure diff)
        # reference baselines
        RandDistOneShotAgent,    # strongest stock baseline (~1.06)
    ]

    import time as _time
    t0 = _time.time()
    print(
        f"[bench] running tournament: n_configs={args.n_configs} "
        f"n_steps={args.n_steps} n_runs={args.n_runs} "
        f"competitors={[c.__name__ for c in competitors]}",
        flush=True,
    )
    print("[bench] (no per-iter callback from anac2024_oneshot — sit tight, "
          "tournament prints when done)", flush=True)
    results = anac2024_oneshot(
        competitors=competitors,
        n_configs=args.n_configs,
        n_runs_per_world=args.n_runs,
        n_steps=args.n_steps,
        parallelism="serial",
    )

    # results.scores is a long-form DataFrame: world, agent_type, score, ...
    df = results.scores.copy()
    df["short"] = df["agent_type"].map(short)
    grouped = df.groupby("short")["score"].agg(["mean", "std", "count"])
    print()
    print(grouped.sort_values("mean", ascending=False).to_string())

    # bar chart, sorted by mean for readability
    grouped = grouped.sort_values("mean", ascending=False)
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values
    labels = grouped.index.tolist()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["tab:red" if lbl.startswith("EFR") else "tab:gray" for lbl in labels]
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean score")
    ax.set_title(
        f"SCML OneShot 2024 — EFR v1 vs builtins\n"
        f"({args.n_configs} configs × {args.n_steps} steps)"
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[bench] wrote {args.out}  (total {_time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
