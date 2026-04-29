"""
generate the SCML EFR v1 bilateral game as a Gambit .efg file.

structure:
  ROOT chance node — uniform over 81 = 3^4 combinations of
    (p1_type_bucket, p1_exog_bucket, p2_type_bucket, p2_exog_bucket).
  then K_ROUNDS alternating decisions, P1 (seller) first:
    round 0          : P1 must propose an offer (no accept option)
    rounds 1..K-2    : current player chooses accept / end / counter-offer
    round K-1 (last) : current player chooses accept / end only
                       (a counter at the last round is identical to disagreement,
                        so we prune it)

infoset labels are produced by infoset.InfosetKey.serialize() — same encoder
the runtime agent uses, so no drift between offline solve and runtime lookup.
perfect recall is preserved by including each player's full offer history.

payoff model (v5 — assumption 3: marginal penalty with multi-partner prior):
  - prior versions treated this bilateral as if it were the agent's only
    negotiation, so the seller's CCE always collapsed to "demand the full
    target from this one partner." v5 implements v1 assumption 3: the leaf
    payoff is the MARGINAL contribution of this bilateral to the agent's
    end-of-day penalty, assuming N_OTHER_PARTNERS=2 other concurrent
    negotiations whose outcomes are drawn from a uniform q ∈ {0..4} prior.
  - seller per-unit cost  = 7 + p1_type     (∈ {7,8,9})
  - buyer  per-unit value = 10 + p2_type    (∈ {10,11,12})
  - price low = 9, price high = 10
  - target_q from exog bucket: {0:4, 1:6, 2:8}   (matched to prior so
      q_me has real marginal pressure: target=4 means others already
      over-fill, target=8 means undershoot)
  - shortfall=3, storage=2, disposal=2 per unit
  - disagreement leaf: (0, 0) — marginal attribution makes q_me=0
    contribute nothing to the total day-end penalty by construction.

run:
  python build_game.py [--out games/scml_oneshot_v1.efg] [--rounds K]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from infoset import (
    InfosetKey,
    K_ROUNDS,
    N_EXOG_BUCKETS,
    N_OTHER_BUCKETS,
    N_OTHER_VALUES,
    N_PRICE,
    N_QTY,
    N_TYPE_BUCKETS,
)


# ----------------------------------------------------------------------
# payoff model (v1 stub)
# ----------------------------------------------------------------------

PRICE_LOW = 9
PRICE_HIGH = 10
PRICE_VALUE = (PRICE_LOW, PRICE_HIGH)        # by p_bucket
QTY_VALUE = (1, 2, 3, 4)                     # by q_bucket

# v6: the number of OTHER concurrent partners is drawn at the chance node
# (see N_OTHER_VALUES in infoset.py). the CCE then conditions its strategy
# on it, so one trained policy handles all runtime partner counts instead
# of being stuck with a single fixed prior. runtime reads the drawn value
# from `len(self.negotiators) - 1` and looks up the matching infoset.
#
# each other partner independently closes q ∈ {0..4} uniformly, so with
# n_other others E[Q_other] = 2 * n_other.
OTHER_QTY_PMF: list[tuple[int, float]] = [(k, 1.0 / 5) for k in range(5)]

# targets sized relative to expected-Q-other so q_me has a real tradeoff:
# target=4 means others already over-fill → low q_me optimal.
# target=8 means others undershoot → high q_me optimal.
EXOG_TARGET = (4, 6, 8)

SHORTFALL_PER_UNIT = 3.0
STORAGE_PER_UNIT = 2.0
DISPOSAL_PER_UNIT = 2.0


def seller_cost(p1_type: int) -> float:
    # type 0: cheap producer (margin 3 at high), type 2: expensive (margin 1)
    return 7.0 + float(p1_type)


def buyer_value(p2_type: int) -> float:
    # type 0: zero margin at high price, type 2: margin 2 at high
    return 10.0 + float(p2_type)


# --- marginal expected penalty --------------------------------------------

def _convolve_pmf(pmf: list[tuple[int, float]], n: int) -> dict[int, float]:
    """pmf of sum of n iid draws from `pmf`."""
    result: dict[int, float] = {0: 1.0}
    for _ in range(n):
        new: dict[int, float] = {}
        for k, p in result.items():
            for kk, pp in pmf:
                new[k + kk] = new.get(k + kk, 0.0) + p * pp
        result = new
    return result


# precompute Q_OTHER pmf for each supported n_other value. leaves built
# later will look up the matching pmf by the chance-drawn n_other.
Q_OTHER_PMFS: dict[int, dict[int, float]] = {
    n: _convolve_pmf(OTHER_QTY_PMF, n) for n in N_OTHER_VALUES
}


def _penalty_seller(q_total: int, target: int) -> float:
    return (
        SHORTFALL_PER_UNIT * max(0, target - q_total)
        + DISPOSAL_PER_UNIT * max(0, q_total - target)
    )


def _penalty_buyer(q_total: int, target: int) -> float:
    return (
        SHORTFALL_PER_UNIT * max(0, target - q_total)
        + STORAGE_PER_UNIT * max(0, q_total - target)
    )


def _marginal_penalty(q_me: int, target: int, penalty_fn, n_other: int) -> float:
    """E[penalty(Q_other + q_me, target) − penalty(Q_other, target)] under
    Q_OTHER_PMFS[n_other]. marginal attribution → q_me=0 contributes 0."""
    pmf = Q_OTHER_PMFS[n_other]
    total = 0.0
    for q_other, p in pmf.items():
        total += p * (
            penalty_fn(q_other + q_me, target) - penalty_fn(q_other, target)
        )
    return total


def agreement_payoffs(
    q_bucket: int, p_bucket: int,
    p1_type: int, p1_exog: int,
    p2_type: int, p2_exog: int,
    n_other: int,
) -> tuple[float, float]:
    q = QTY_VALUE[q_bucket]
    p = PRICE_VALUE[p_bucket]
    cost = seller_cost(p1_type)
    val = buyer_value(p2_type)
    p1_target = EXOG_TARGET[p1_exog]
    p2_target = EXOG_TARGET[p2_exog]
    seller = (p - cost) * q - _marginal_penalty(q, p1_target, _penalty_seller, n_other)
    buyer = (val - p) * q - _marginal_penalty(q, p2_target, _penalty_buyer, n_other)
    return seller, buyer


def disagreement_payoffs(
    p1_exog: int, p2_exog: int, n_other: int,
) -> tuple[float, float]:
    # under marginal attribution the q_me=0 outcome contributes 0 to either
    # side's total day-end penalty regardless of n_other.
    return (0.0, 0.0)


# ----------------------------------------------------------------------
# EFG writer
# ----------------------------------------------------------------------

@dataclass
class GameContext:
    out: list[str]
    rounds: int
    # per-player infoset numbering (1-indexed for EFG)
    iset_ids: tuple[dict[str, int], dict[str, int]]
    next_outcome_id: int = 1

    def get_iset_id(self, player_idx: int, label: str) -> int:
        # player_idx is 0 or 1 (we add 1 when emitting per EFG convention)
        d = self.iset_ids[player_idx]
        if label not in d:
            d[label] = len(d) + 1
        return d[label]

    def new_outcome_id(self) -> int:
        oid = self.next_outcome_id
        self.next_outcome_id += 1
        return oid

    def emit(self, line: str) -> None:
        self.out.append(line)


def offer_action_label(q_bucket: int, p_bucket: int) -> str:
    return f"o{q_bucket}{p_bucket}"


OFFER_ACTIONS: list[tuple[int, int]] = [
    (q, p) for q in range(N_QTY) for p in range(N_PRICE)
]


def emit_decision(
    ctx: GameContext,
    *,
    player: int,                    # 1 = seller, 2 = buyer
    round_idx: int,
    p1_type: int, p1_exog: int,
    p2_type: int, p2_exog: int,
    n_other_idx: int,
    last_offer: tuple[int, int] | None,
    p1_history: tuple[tuple[int, int], ...],
    p2_history: tuple[tuple[int, int], ...],
) -> None:
    """recursively emit a player decision node and all of its children."""
    is_seller = player == 1
    role = "S" if is_seller else "B"
    my_type = p1_type if is_seller else p2_type
    my_exog = p1_exog if is_seller else p2_exog
    my_history = p1_history if is_seller else p2_history

    key = InfosetKey(
        role=role,
        my_type=my_type,
        my_exog=my_exog,
        n_other_idx=n_other_idx,
        round=round_idx,
        last_offer=last_offer,
        my_history=my_history,
    )
    label = key.serialize()
    iset_id = ctx.get_iset_id(player - 1, label)
    n_other = N_OTHER_VALUES[n_other_idx]

    last_round = round_idx == ctx.rounds - 1
    can_accept = round_idx > 0          # nothing to accept on round 0
    can_end = round_idx > 0             # symmetry: end only meaningful after some exchange
    can_offer = not last_round          # last-round offer pruned (= disagreement)

    actions: list[str] = []
    if can_accept:
        actions.append("acc")
    if can_end:
        actions.append("end")
    if can_offer:
        for q, p in OFFER_ACTIONS:
            actions.append(offer_action_label(q, p))

    if not actions:
        raise RuntimeError(f"no actions at {label}")

    actions_str = " ".join(f'"{a}"' for a in actions)
    ctx.emit(f'p "" {player} {iset_id} "{label}" {{ {actions_str} }} 0')

    # children, in the same order as actions
    if can_accept:
        assert last_offer is not None
        s_pay, b_pay = agreement_payoffs(
            last_offer[0], last_offer[1],
            p1_type, p1_exog, p2_type, p2_exog,
            n_other,
        )
        oid = ctx.new_outcome_id()
        ctx.emit(f't "" {oid} "acc_{label}" {{ {s_pay} {b_pay} }}')

    if can_end:
        s_pay, b_pay = disagreement_payoffs(p1_exog, p2_exog, n_other)
        oid = ctx.new_outcome_id()
        ctx.emit(f't "" {oid} "end_{label}" {{ {s_pay} {b_pay} }}')

    if can_offer:
        next_player = 2 if player == 1 else 1
        next_round = round_idx + 1
        for q, p in OFFER_ACTIONS:
            new_offer = (q, p)
            if is_seller:
                new_p1_history = p1_history + (new_offer,)
                new_p2_history = p2_history
            else:
                new_p1_history = p1_history
                new_p2_history = p2_history + (new_offer,)
            emit_decision(
                ctx,
                player=next_player,
                round_idx=next_round,
                p1_type=p1_type, p1_exog=p1_exog,
                p2_type=p2_type, p2_exog=p2_exog,
                n_other_idx=n_other_idx,
                last_offer=new_offer,
                p1_history=new_p1_history,
                p2_history=new_p2_history,
            )


def build(rounds: int) -> tuple[str, dict]:
    import time as _time
    print(f"[build_game] starting tree walk (K={rounds})", flush=True)
    t0 = _time.time()
    ctx = GameContext(
        out=[],
        rounds=rounds,
        iset_ids=({}, {}),
    )

    ctx.emit(
        'EFG 2 R "SCML OneShot v1 bilateral (N-indexed)" '
        '{ "Seller" "Buyer" } '
        '"single-day bilateral abstraction with n_other drawn at chance; '
        'see scml_efr/build_game.py"'
    )

    # root chance: 81 (types × exogs) × N_OTHER_BUCKETS (partner count) outcomes
    branches = []
    # tuple = (p1t, p1e, p2t, p2e, n_other_idx, name)
    chance_outcomes: list[tuple[int, int, int, int, int, str]] = []
    total = (N_TYPE_BUCKETS ** 2) * (N_EXOG_BUCKETS ** 2) * N_OTHER_BUCKETS
    for p1t in range(N_TYPE_BUCKETS):
        for p1e in range(N_EXOG_BUCKETS):
            for p2t in range(N_TYPE_BUCKETS):
                for p2e in range(N_EXOG_BUCKETS):
                    for noi in range(N_OTHER_BUCKETS):
                        name = f"t{p1t}e{p1e}t{p2t}e{p2e}n{noi}"
                        chance_outcomes.append((p1t, p1e, p2t, p2e, noi, name))
                        branches.append(f'"{name}" 1/{total}')

    ctx.emit(f'c "ROOT" 1 "root_chance" {{ {" ".join(branches)} }} 0')

    for i, (p1t, p1e, p2t, p2e, noi, _name) in enumerate(chance_outcomes):
        emit_decision(
            ctx,
            player=1,
            round_idx=0,
            p1_type=p1t, p1_exog=p1e,
            p2_type=p2t, p2_exog=p2e,
            n_other_idx=noi,
            last_offer=None,
            p1_history=(),
            p2_history=(),
        )
        # heartbeat per chance branch. with ~324 of them, ~every few s
        if (i + 1) % 10 == 0 or i + 1 == len(chance_outcomes):
            elapsed = _time.time() - t0
            print(
                f"[build_game]   chance {i+1:>4d}/{len(chance_outcomes)}  "
                f"lines={len(ctx.out):>9,d}  "
                f"isets P1={len(ctx.iset_ids[0]):>5d} P2={len(ctx.iset_ids[1]):>5d}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    stats = {
        "n_lines": len(ctx.out),
        "n_p1_infosets": len(ctx.iset_ids[0]),
        "n_p2_infosets": len(ctx.iset_ids[1]),
        "n_terminals": ctx.next_outcome_id - 1,
        "n_chance_outcomes": len(chance_outcomes),
    }
    return "\n".join(ctx.out) + "\n", stats


def main() -> int:
    from _runlog import start as _start_log
    _start_log("build_game")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "games" / "scml_oneshot_v1.efg")
    ap.add_argument("--rounds", type=int, default=K_ROUNDS,
                    help=f"K, negotiation rounds (default {K_ROUNDS} from infoset.py)")
    args = ap.parse_args()

    text, stats = build(args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text):,} bytes)")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
