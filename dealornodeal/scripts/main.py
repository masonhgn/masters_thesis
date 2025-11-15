# helpers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.algorithms import cfr, efr
from open_spiel.python.algorithms import outcome_sampling_mccfr as mccfr
import sys




import games.deal_or_no_deal as deal_or_no_deal, games.deal_or_no_deal_zerosum as deal_or_no_deal_zerosum
import games.deal_or_no_deal_mini as deal_or_no_deal_mini, games.deal_or_no_deal_mini_zerosum as deal_or_no_deal_mini_zerosum

# helpers.py (minimal single-function version)








def _nodes_map(solver):
    return getattr(solver, "_info_state_nodes", None) or getattr(solver, "_info_states", None)


def _cumreg_vals(node):
    cr = getattr(node, "cumulative_regret", None)
    if cr is None:
        cr = getattr(node, "cumulative_regrets", None)
    if cr is None:
        return []
    return list(cr.values()) if isinstance(cr, dict) else list(cr)


def _agg_avg_ext_regret(solver, t):
    nodes = _nodes_map(solver)
    raw_sum = 0.0
    for node in nodes.values():
        vals = _cumreg_vals(node)
        if vals:
            raw_sum += max(0.0, float(max(vals)))
    return raw_sum / max(t, 1)


def _plot(x, y, title, fname):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker=".")
    plt.yscale("log")
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("aggregate avg external regret")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def run_cfr_agg(game_name="python_deal_or_no_deal_mini", iters=1000, eval_every=10, seed=7):
    np.random.seed(seed)
    game = pyspiel.load_game(game_name)
    solver = cfr.CFRSolver(game)
    xs, agg = [], []
    for t in range(1, iters + 1):
        solver.evaluate_and_update_policy()
        if t % eval_every == 0:
            val = _agg_avg_ext_regret(solver, t)
            xs.append(t)
            agg.append(val)
            print(f"[CFR] Iter {t}/{iters}: avg_ext_regret={val:.6f}") 
    _plot(xs, agg, "CFR Convergence", f"{game_name}_cfr.png")
    return {"x": xs, "agg": agg}


def run_efr_agg(game_name="python_deal_or_no_deal_mini", iters=1000, eval_every=10, seed=7, deviation="csps"):
    np.random.seed(seed)
    game = pyspiel.load_game(game_name)
    solver = efr.EFRSolver(game, deviations_name=deviation)
    xs, agg = [], []
    for t in range(1, iters + 1):
        solver.evaluate_and_update_policy()
        if t % eval_every == 0:
            val = _agg_avg_ext_regret(solver, t)
            xs.append(t)
            agg.append(val)
            print(f"[EFR-{deviation}] Iter {t}/{iters}: avg_ext_regret={val:.6f}") 
    _plot(xs, agg, f"EFR Convergence ({deviation})", f"{game_name}_efr_{deviation}.png")
    return {"x": xs, "agg": agg}





def run_mccfr_agg(game_name="python_deal_or_no_deal_mini", iters=2000, eval_every=100, seed=7):
    """
    Tracks MCCFR convergence by computing aggregate average external regret
    directly from internal _infostates (no NashConv).
    """
    np.random.seed(seed)
    game = pyspiel.load_game(game_name)
    solver = mccfr.OutcomeSamplingSolver(game)
    xs, agg = [], []

    for t in range(1, iters + 1):
        solver.iteration()
        if t % eval_every == 0:
            nodes = solver._infostates  # {info_state_str: [regrets, avg_strat]}
            raw_sum = 0.0
            for key, (regrets, avg_strat) in nodes.items():
                max_reg = np.maximum(regrets, 0.0)
                raw_sum += float(np.max(max_reg))
            avg_regret = raw_sum / max(t, 1)
            xs.append(t)
            agg.append(avg_regret)
            print(f"[MCCFR] Iter {t}/{iters}: avg_ext_regret={avg_regret:.6f}  (raw_sum={raw_sum:.3f})")

    _plot(xs, agg, "MCCFR Convergence (Avg External Regret)", f"{game_name}_mccfr.png")
    return {"x": xs, "agg": agg}

















#run_cfr_agg("python_deal_or_no_deal_mini", iters=500, eval_every=10)
run_efr_agg("python_deal_or_no_deal_mini", iters=500, eval_every=10, deviation="tips")
#run_mccfr_agg("python_deal_or_no_deal", iters=20000, eval_every=50)
#run_mcefr_agg("python_deal_or_no_deal", iters=20000, eval_every=50, deviation="blind cf")