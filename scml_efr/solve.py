"""
phase 4: solve scml_oneshot_v1.efg with EFR and dump the average
behavioral policy as a flat lookup table keyed by InfosetKey labels.

uses pyspiel's EFRSolver (`open_spiel.python.algorithms.efr`), which
implements the same family of deviation-set-based extensive-form regret
minimizers as the project's C++ `run_corr_dist` (see run_corr_dist.cc:349+).

the canonical EFR variant for v1 assumption 6 (CCE target) is the
BlindActionSequencePredecessors / "blind action" deviation set — external
deviations → average policy converges to a CCE.

available deviation sets (pass via --deviations):
    blind action        external, CCE-converging        (default)
    informed action     action deviations, AFCCE/AFCE target
    bps                 blind partial sequence, CCE
    cfps                counterfactual partial sequence, AFCE/EFCE
    csps                causal partial sequence, AFCE/EFCE
    tips                twice informed partial sequence, EFCE
    bhv                 behavioral, EFCE (strongest)

output format (one infoset per line):
    <iset_label>\t<action_str>:<prob>,<action_str>:<prob>,...

where <iset_label> is exactly InfosetKey.serialize(), and <action_str>
matches the action labels emitted by build_game.py (acc / end / oXY).
"""

from __future__ import annotations

# silence pyspiel EFR's intermediate numpy matmul warnings on multi-player
# games — they fire on edge cases but the final policy is still valid
# (all finite, rows sum to 1). verified empirically before suppressing.
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)

import argparse
import sys
from pathlib import Path

import pyspiel
from open_spiel.python.algorithms import efr, exploitability


def extract_label(info_state_string: str) -> str:
    """efg_game prefixes our infoset label with `<player>-<chance>-<iset_id>-`.
    our labels start with the role char (S or B) and contain `|` but no `-`,
    so the label is the substring after the last `-`."""
    return info_state_string.rsplit("-", 1)[-1]


def collect_legal_actions(game) -> dict[str, list[tuple[int, str]]]:
    """walk the tree once, return iset_label → list of (action_id, action_str)
    for the first node encountered in each infoset (legal actions are constant
    across all nodes in an infoset by definition)."""
    seen: dict[str, list[tuple[int, str]]] = {}

    def recurse(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                recurse(state.child(a))
            return
        label = extract_label(state.information_state_string())
        if label not in seen:
            player = state.current_player()
            seen[label] = [
                (a, state.action_to_string(player, a)) for a in state.legal_actions()
            ]
        for a in state.legal_actions():
            recurse(state.child(a))

    recurse(game.new_initial_state())
    return seen


def dump_policy(avg_policy, legal_by_label, out_path: Path) -> int:
    """write the policy table. returns number of rows written."""
    table = avg_policy.action_probability_array
    state_lookup = avg_policy.state_lookup

    # group by our label (different efg-prefixes can map to same label only if
    # the same iset is reached via multiple chance paths — efg_game gives them
    # one row per (player, chance, iset) so we just take the first occurrence)
    rows: dict[str, int] = {}
    for info_state_string, row_idx in state_lookup.items():
        label = extract_label(info_state_string)
        if label not in rows:
            rows[label] = row_idx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for label, row_idx in sorted(rows.items()):
            probs = table[row_idx]
            legal = legal_by_label.get(label)
            if legal is None:
                # shouldn't happen but skip cleanly
                continue
            parts = []
            for action_id, action_str in legal:
                p = float(probs[action_id])
                parts.append(f"{action_str}:{p:.6f}")
            f.write(f"{label}\t{','.join(parts)}\n")
    return len(rows)


def main() -> int:
    from _runlog import start as _start_log
    _start_log("solve")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--efg",
        type=Path,
        default=Path(__file__).parent / "games" / "scml_oneshot_v1.efg",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "policies" / "scml_oneshot_v1.policy",
    )
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="print nash_conv every N iterations",
    )
    ap.add_argument(
        "--deviations",
        type=str,
        default="blind action",
        help="EFR deviation set (blind action, informed action, bps, cfps, "
             "csps, tips, bhv). 'blind action' → CCE target (v1 assumption 6).",
    )
    args = ap.parse_args()

    import time as _time
    t0 = _time.time()

    print(f"[solve] loading {args.efg} ...", flush=True)
    game = pyspiel.load_game(f"efg_game(filename={args.efg})")
    print(
        f"[solve] loaded in {_time.time()-t0:.1f}s: "
        f"{game.num_distinct_actions()} actions, {game.num_players()} players",
        flush=True,
    )

    t1 = _time.time()
    print("[solve] walking tree to collect legal actions per infoset ...", flush=True)
    legal_by_label = collect_legal_actions(game)
    print(
        f"[solve] collected {len(legal_by_label)} infosets in {_time.time()-t1:.1f}s",
        flush=True,
    )

    print(f"[solve] starting EFR ({args.deviations!r}) for {args.iterations} iterations "
          f"(report every {args.report_every})", flush=True)
    print("[solve] note: EFR is ~5x slower per iter than CFR+. 1000 iters ≈ 30 min.", flush=True)
    solver = efr.EFRSolver(game, args.deviations)
    t_iter = _time.time()
    for t in range(1, args.iterations + 1):
        solver.evaluate_and_update_policy()
        # cheap heartbeat every iteration so the user always sees progress
        if t % max(1, args.report_every // 10) == 0 and t % args.report_every != 0:
            elapsed = _time.time() - t_iter
            rate = t / elapsed if elapsed else 0
            eta = (args.iterations - t) / rate if rate else 0
            print(
                f"[solve]   .. iter {t:>5d}/{args.iterations}  "
                f"({rate:.1f} it/s, eta {eta:.0f}s)",
                flush=True,
            )
        if t % args.report_every == 0 or t == args.iterations:
            gap = exploitability.nash_conv(game, solver.average_policy())
            elapsed = _time.time() - t_iter
            print(
                f"[solve] iter {t:>5d}/{args.iterations}  "
                f"nash_conv={gap:.6f}  ({elapsed:.0f}s elapsed)",
                flush=True,
            )

    print("[solve] dumping policy ...", flush=True)
    n_rows = dump_policy(solver.average_policy(), legal_by_label, args.out)
    print(
        f"[solve] DONE: wrote {args.out} ({n_rows} infoset rows) "
        f"in total {_time.time()-t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
