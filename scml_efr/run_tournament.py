"""
phase 0: smoke-test the scml install by running a small builtin
anac2024_oneshot tournament with the stock baseline agents.
"""

from scml.oneshot.agents import (
    GreedyOneShotAgent,
    RandDistOneShotAgent,
    EqualDistOneShotAgent,
)
from scml.utils import anac2024_oneshot

from efr_oneshot_agent import EFROneShotAgent


def main():
    # small config so the smoke test runs in seconds
    results = anac2024_oneshot(
        competitors=[
            EFROneShotAgent,
            GreedyOneShotAgent,
            RandDistOneShotAgent,
            EqualDistOneShotAgent,
        ],
        n_configs=2,
        n_steps=10,
        n_runs_per_world=1,
        parallelism="serial",
    )
    print(results.score_stats)


if __name__ == "__main__":
    main()
