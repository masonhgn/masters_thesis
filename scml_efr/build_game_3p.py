"""
v2: 3-player single-round EFG for SCML OneShot.

players: Seller, Buyer1, Buyer2

chance: draws (s_type, s_exog, b1_type, b1_exog, b2_type, b2_exog)
  uniform over 3^6 = 729 outcomes.

decisions:
  seller (P1): picks joint offer (q_A, q_B, p) — 4 * 4 * 2 = 32 actions.
               seller infoset depends only on own type+exog (hidden buyer types).
  buyer1 (P2): accept / reject the offer to themselves.
               infoset: (b1_type, b1_exog, q_A, p). hides q_B, hides b2 info.
  buyer2 (P3): symmetric to buyer1.

payoffs (marginal attribution so disagreement is 0):
  seller: (p - cost(s_type)) * closed - penalty(closed, s_target)
          where closed = q_A*acc_A + q_B*acc_B.
  buyer:  accept → (v-p)*q - marginal penalties vs zero-trade baseline.
          reject → 0.

this is the smallest abstraction that captures the multi-partner coupling:
each buyer's equilibrium accept threshold depends on what they believe the
other buyer is doing, because the seller's supply is shared between them.
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path


# discretization (matched to 2-player v5 for comparability)
N_TYPES = 3
N_EXOG = 3

PRICE_LOW = 9
PRICE_HIGH = 10
PRICES = (PRICE_LOW, PRICE_HIGH)
QUANTITIES = (1, 2, 3, 4)
EXOG_TARGETS = (1, 2, 4)

SHORTFALL = 3.0
STORAGE = 2.0
DISPOSAL = 2.0


def seller_cost(t: int) -> float:
    return 7.0 + float(t)


def buyer_value(t: int) -> float:
    return 10.0 + float(t)


def seller_payoff(s_type: int, s_exog: int, q_closed: int, p: int) -> float:
    target = EXOG_TARGETS[s_exog]
    cost = seller_cost(s_type)
    revenue = p * q_closed
    total_cost = cost * q_closed
    penalty = (SHORTFALL * max(0, target - q_closed)
               + DISPOSAL * max(0, q_closed - target))
    return revenue - total_cost - penalty


def buyer_marginal_payoff(b_type: int, b_exog: int, q: int, p: int) -> float:
    """payoff of accepting (q,p) minus the payoff of rejecting (no trade).
    this marginal form makes disagreement clean at 0 and keeps the leaf
    numbers comparable across decisions."""
    target = EXOG_TARGETS[b_exog]
    v = buyer_value(b_type)
    accept = (v - p) * q - STORAGE * max(0, q - target) - SHORTFALL * max(0, target - q)
    reject = -SHORTFALL * target
    return accept - reject


# enumerate action sets once
SELLER_ACTIONS: list[tuple[int, int, int]] = [
    (q_a, q_b, p)
    for q_a in range(len(QUANTITIES))
    for q_b in range(len(QUANTITIES))
    for p in range(len(PRICES))
]
BUYER_ACTIONS = ("acc", "rej")


def seller_action_label(q_a_idx: int, q_b_idx: int, p_idx: int) -> str:
    return f"s{q_a_idx}{q_b_idx}{p_idx}"


def seller_iset_label(s_type: int, s_exog: int) -> str:
    return f"S|{s_type}|{s_exog}"


def buyer_iset_label(role: str, b_type: int, b_exog: int, q_idx: int, p_idx: int) -> str:
    # role = "B1" or "B2". hides the OTHER buyer's everything.
    return f"{role}|{b_type}|{b_exog}|{q_idx}{p_idx}"


def _emit_chance_branches(n: int) -> list[str]:
    """uniform over n outcomes, enumerated by name."""
    return [f'"c{i}" 1/{n}' for i in range(n)]


def build() -> tuple[str, dict]:
    t0 = _time.time()
    lines: list[str] = []

    # EFG header for a 3-player game
    lines.append(
        'EFG 2 R "SCML OneShot v2 3-player single-round" '
        '{ "Seller" "Buyer1" "Buyer2" } '
        '"single-round 3-player abstraction; see scml_efr/build_game_3p.py"'
    )

    # root chance: enumerate (s_type, s_exog, b1_type, b1_exog, b2_type, b2_exog)
    chance_outcomes: list[tuple[int, int, int, int, int, int, str]] = []
    for st in range(N_TYPES):
        for se in range(N_EXOG):
            for b1t in range(N_TYPES):
                for b1e in range(N_EXOG):
                    for b2t in range(N_TYPES):
                        for b2e in range(N_EXOG):
                            name = f"s{st}{se}b{b1t}{b1e}b{b2t}{b2e}"
                            chance_outcomes.append(
                                (st, se, b1t, b1e, b2t, b2e, name)
                            )

    total_chance = len(chance_outcomes)
    branches = [f'"{name}" 1/{total_chance}'
                for (*_, name) in chance_outcomes]
    lines.append(f'c "ROOT" 1 "root_chance" {{ {" ".join(branches)} }} 0')

    # per-player infoset numbering, 1-indexed (EFG convention)
    seller_iset_ids: dict[str, int] = {}
    b1_iset_ids: dict[str, int] = {}
    b2_iset_ids: dict[str, int] = {}

    def iset_id_for(d: dict[str, int], label: str) -> int:
        if label not in d:
            d[label] = len(d) + 1
        return d[label]

    next_outcome_id = 1

    print(f"[build_3p] starting tree walk: {total_chance} chance outcomes, "
          f"{len(SELLER_ACTIONS)} seller actions each", flush=True)

    # iterate each chance outcome, emit seller decision + buyer decisions + terminal
    for ci, (st, se, b1t, b1e, b2t, b2e, cname) in enumerate(chance_outcomes):
        # seller decision node
        s_label = seller_iset_label(st, se)
        s_id = iset_id_for(seller_iset_ids, s_label)
        action_names = [seller_action_label(a, b, p) for (a, b, p) in SELLER_ACTIONS]
        actions_str = " ".join(f'"{a}"' for a in action_names)
        lines.append(f'p "" 1 {s_id} "{s_label}" {{ {actions_str} }} 0')

        # for each seller action, emit b1 decision, then b2 decision, then terminal
        for (qa_idx, qb_idx, p_idx) in SELLER_ACTIONS:
            # b1 decision node
            b1_label = buyer_iset_label("B1", b1t, b1e, qa_idx, p_idx)
            b1_id = iset_id_for(b1_iset_ids, b1_label)
            lines.append(
                f'p "" 2 {b1_id} "{b1_label}" '
                f'{{ "{BUYER_ACTIONS[0]}" "{BUYER_ACTIONS[1]}" }} 0'
            )

            for b1_choice, acc_a in enumerate([True, False]):
                # b2 decision node
                b2_label = buyer_iset_label("B2", b2t, b2e, qb_idx, p_idx)
                b2_id = iset_id_for(b2_iset_ids, b2_label)
                lines.append(
                    f'p "" 3 {b2_id} "{b2_label}" '
                    f'{{ "{BUYER_ACTIONS[0]}" "{BUYER_ACTIONS[1]}" }} 0'
                )

                for b2_choice, acc_b in enumerate([True, False]):
                    # terminal
                    qa = QUANTITIES[qa_idx]
                    qb = QUANTITIES[qb_idx]
                    pv = PRICES[p_idx]
                    q_closed = (qa if acc_a else 0) + (qb if acc_b else 0)

                    s_pay = seller_payoff(st, se, q_closed, pv)
                    b1_pay = buyer_marginal_payoff(b1t, b1e, qa, pv) if acc_a else 0.0
                    b2_pay = buyer_marginal_payoff(b2t, b2e, qb, pv) if acc_b else 0.0

                    oid = next_outcome_id
                    next_outcome_id += 1
                    tlabel = f"t_{cname}_{seller_action_label(qa_idx, qb_idx, p_idx)}_{int(acc_a)}{int(acc_b)}"
                    lines.append(
                        f't "" {oid} "{tlabel}" '
                        f'{{ {s_pay} {b1_pay} {b2_pay} }}'
                    )

        if (ci + 1) % 50 == 0 or ci + 1 == total_chance:
            elapsed = _time.time() - t0
            print(
                f"[build_3p]   chance {ci+1:>4d}/{total_chance}  "
                f"lines={len(lines):>9,d}  "
                f"isets S={len(seller_iset_ids):>4d} B1={len(b1_iset_ids):>4d} B2={len(b2_iset_ids):>4d}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    stats = {
        "n_lines": len(lines),
        "n_seller_infosets": len(seller_iset_ids),
        "n_b1_infosets": len(b1_iset_ids),
        "n_b2_infosets": len(b2_iset_ids),
        "n_terminals": next_outcome_id - 1,
        "n_chance_outcomes": total_chance,
        "elapsed_s": _time.time() - t0,
    }
    return "\n".join(lines) + "\n", stats


def main() -> int:
    from _runlog import start as _start_log
    _start_log("build_game_3p")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "games" / "scml_oneshot_v2_3p.efg",
    )
    args = ap.parse_args()

    text, stats = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text):,} bytes)", flush=True)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
