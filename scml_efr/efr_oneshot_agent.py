"""
SCML OneShot agent that plays the offline-solved EFR (CCE) policy.

pipeline (see scml_efr/build_game.py and solve.py for the offline side):
  init():
    load policies/scml_oneshot_v1.policy into memory.
  before_step():
    reset per-day per-partner trackers.
  first_proposals() / counter_all():
    for each active partner, build the abstract InfosetKey, look it up
    in the policy table, sample an action, convert to a SCML offer or
    SAOResponse. on miss (or on rounds beyond K_ROUNDS-1), fall back to
    a hardcoded greedy strategy.

v1 mappings from real SCML state to the abstract game (intentionally crude
— see scml_efr/.claude references/v1_assumptions.md):
  - private cost type bucket : constant middle (1). proper mapping is v2 work.
  - own exog quantity bucket : derived from needed_sales / needed_supplies.
  - last offer price bucket  : split at midpoint of nmi.issues[UNIT_PRICE].
  - last offer qty bucket    : clipped quantize of offer[QUANTITY].
  - abstract round           : seller decisions land at rounds 0,2;
                               buyer decisions land at rounds 1,3.
                               beyond K_ROUNDS-1 → fallback.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

from negmas import Outcome, ResponseType, SAOResponse, SAOState
from scml.oneshot import QUANTITY, TIME, UNIT_PRICE, OneShotSyncAgent

from infoset import (
    InfosetKey,
    K_ROUNDS,
    N_EXOG_BUCKETS,
    N_OTHER_VALUES,
    N_PRICE,
    N_QTY,
    bucket_exog,
    bucket_price,
    bucket_qty,
    bucket_type,
)


# ----------------------------------------------------------------------
# policy file io (one row per infoset, action_str:prob,...)
# ----------------------------------------------------------------------

def load_policy(path: Path) -> dict[str, list[tuple[str, float]]]:
    table: dict[str, list[tuple[str, float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        label, body = line.split("\t", 1)
        entries: list[tuple[str, float]] = []
        for tok in body.split(","):
            name, prob = tok.split(":")
            entries.append((name, float(prob)))
        table[label] = entries
    return table


# defaults match the layout the rest of phase 4 writes to
DEFAULT_POLICY_PATH = Path(__file__).parent / "policies" / "scml_oneshot_v1.policy"

# structured diagnostic log. one jsonl per tournament run; set the path via
# env var EFR_LOG_PATH so different runs can land in different files.
DEFAULT_LOG_PATH = Path(__file__).parent / "logs" / "efr_agent.jsonl"


def _open_log() -> Optional[Any]:
    path_str = os.environ.get("EFR_LOG_PATH", str(DEFAULT_LOG_PATH))
    if not path_str:
        return None
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    # line-buffered append; multiple agent instances in the same process
    # share the file (serial tournaments are single-process so this is safe)
    return open(path, "a", buffering=1)


class EFROneShotAgent(OneShotSyncAgent):
    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        self._policy: dict[str, list[tuple[str, float]]] = {}
        if DEFAULT_POLICY_PATH.exists():
            self._policy = load_policy(DEFAULT_POLICY_PATH)
        # use a per-agent rng so multiple agents in one world don't share state
        self._rng = random.Random()
        # cumulative metrics across the world's lifetime
        self._policy_hits = 0
        self._policy_misses = 0
        # missed-key counter for offline analysis
        self._missed_keys: dict[str, int] = {}
        # diagnostic log
        self._log_fh = _open_log()
        self._log(
            type="init",
            policy_loaded=bool(self._policy),
            policy_size=len(self._policy),
            level=getattr(self.awi, "level", None),
            n_lines=getattr(self.awi, "n_lines", None),
            is_first=getattr(self.awi, "is_first_level", None),
            is_last=getattr(self.awi, "is_last_level", None),
        )

    def before_step(self) -> None:
        # prime ufun bounds for today
        self.ufun.find_limit(True)
        self.ufun.find_limit(False)
        # per-day per-partner trackers
        self._decision_count: dict[str, int] = {}
        self._my_history: dict[str, list[tuple[int, int]]] = {}
        # capture day-level snapshot for the log
        self._log(
            type="day",
            day=self.awi.current_step,
            needed_sales=self.awi.needed_sales,
            needed_supplies=self.awi.needed_supplies,
            exog_in_q=self.awi.current_exogenous_input_quantity,
            exog_in_p=self.awi.current_exogenous_input_price,
            exog_out_q=self.awi.current_exogenous_output_quantity,
            exog_out_p=self.awi.current_exogenous_output_price,
            ufun_max=float(self.ufun.max_utility),
            ufun_min=float(self.ufun.min_utility),
            n_partners=len(self.negotiators),
        )

    def step(self) -> None:
        total = self._policy_hits + self._policy_misses
        self._log(
            type="step_end",
            day=self.awi.current_step,
            score=float(self.awi.current_score),
            balance=float(self.awi.current_balance),
            cum_hits=self._policy_hits,
            cum_misses=self._policy_misses,
            hit_rate=(self._policy_hits / total) if total else None,
        )

    # ------------------------------------------------------------------
    # logging helper
    # ------------------------------------------------------------------

    def _log(self, **fields: Any) -> None:
        if self._log_fh is None:
            return
        try:
            base = {
                "ts": time.time(),
                "agent_id": getattr(self, "id", None),
                "world": getattr(getattr(self, "awi", None), "current_step", None),
            }
            base.update(fields)
            self._log_fh.write(json.dumps(base, default=str) + "\n")
        except Exception:
            # never let a logging failure crash the agent
            pass

    # ------------------------------------------------------------------
    # sync agent interface
    # ------------------------------------------------------------------

    def first_proposals(self) -> dict[str, Outcome | None]:
        out: dict[str, Outcome | None] = {}
        for nid in self.negotiators.keys():
            if self._is_seller(nid):
                # seller leads in the abstract game — consult the policy
                out[nid] = self._propose(nid, last_offer=None)
            else:
                # buyer's first_proposals is out of domain for our EFG
                # (the buyer only acts in response to the seller's round-0
                # offer there). open with the fallback offer; don't consume
                # an abstract decision slot so the buyer's first counter_all
                # lands at abstract round 1, exactly as the EFG expects.
                # importantly: do NOT record this offer to my_history — it
                # is a placeholder, not part of the abstract game.
                offer = self._fallback_offer(nid)
                self._log(
                    type="propose",
                    day=self.awi.current_step,
                    partner=nid,
                    role="B",
                    decision_idx=0,
                    abstract_round=None,
                    key=None,
                    key_in_range=False,
                    hit=False,
                    sampled_action=None,
                    fallback=True,
                    offer=list(offer) if offer is not None else None,
                    opp_last_offer=None,
                    note="buyer_first_proposals_oodomain",
                )
                out[nid] = offer
        return out

    def counter_all(
        self,
        offers: dict[str, Outcome | None],
        states: dict[str, SAOState],
    ) -> dict[str, SAOResponse]:
        out: dict[str, SAOResponse] = {}
        for nid, offer in offers.items():
            if offer is None:
                # opponent has no offer this round, just keep proposing
                out[nid] = SAOResponse(
                    ResponseType.REJECT_OFFER, self._propose(nid, last_offer=None)
                )
                continue
            decision, sampled_offer = self._react(nid, offer)
            if decision == "accept":
                out[nid] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            elif decision == "end":
                out[nid] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            else:
                # counter — if the policy already gave us an offer in _react,
                # use it directly; otherwise call the fallback. either way we
                # do NOT consult the policy a second time for the same SAO round.
                if sampled_offer is None:
                    sampled_offer = self._fallback_offer(nid)
                if sampled_offer is not None:
                    self._record_my_offer(nid, sampled_offer)
                self._decision_count[nid] = self._decision_count.get(nid, 0) + 1
                out[nid] = SAOResponse(ResponseType.REJECT_OFFER, sampled_offer)
        return out

    # ------------------------------------------------------------------
    # policy lookup core
    # ------------------------------------------------------------------

    def _is_seller(self, nid: str) -> bool:
        nmi = self.get_nmi(nid)
        if nmi is not None and nmi.annotation:
            return nmi.annotation.get("product") == self.awi.my_output_product
        return nid in self.awi.current_negotiation_details.get("sell", {})

    def _need(self, nid: str) -> int:
        return self.awi.needed_sales if self._is_seller(nid) else self.awi.needed_supplies

    def _abstract_round(self, nid: str) -> int:
        """seller plays at rounds 0,2; buyer at rounds 1,3 (matches build_game.py).
        decision_count is the number of times this agent has acted on this partner.

        late SCML SAO rounds saturate at the last in-range abstract round for
        this player so we keep using the policy instead of falling back to greedy
        once the abstract horizon is exhausted."""
        k = self._decision_count.get(nid, 0)
        is_seller = self._is_seller(nid)
        # max in-range round per role:
        #   seller (offset=0) plays at 0, 2, ..., last even < K
        #   buyer  (offset=1) plays at 1, 3, ..., last odd  < K
        if is_seller:
            max_round = K_ROUNDS - 1 if (K_ROUNDS - 1) % 2 == 0 else K_ROUNDS - 2
            offset = 0
        else:
            max_round = K_ROUNDS - 1 if (K_ROUNDS - 1) % 2 == 1 else K_ROUNDS - 2
            offset = 1
        rnd = offset + 2 * k
        return min(rnd, max_round)

    def _build_key(
        self, nid: str, last_offer: Optional[Outcome]
    ) -> Optional[InfosetKey]:
        """returns None if the abstract round is past K_ROUNDS-1 (forces fallback)."""
        rnd = self._abstract_round(nid)
        if rnd >= K_ROUNDS:
            return None
        nmi = self.get_nmi(nid)
        if nmi is None:
            return None

        # exog bucket from full-day exogenous quantity. remaining-need
        # bucketing was tried and hurt the score: late-day small-need
        # states would land in exog=0 whose policy row says q=4 (because
        # margin > disposal in that abstract game), causing over-commits.
        # the partner-count divider in _action_to_offer handles the actual
        # scale issue more robustly.
        if self._is_seller(nid):
            exog_q = int(self.awi.current_exogenous_input_quantity or 0)
        else:
            exog_q = int(self.awi.current_exogenous_output_quantity or 0)
        max_q = int(getattr(self.awi, "n_lines", 10) or 10)
        my_exog = bucket_exog(exog_q, 0, max_q)

        # private cost type: SCML costs are small ints in ~{0..5}, so
        # bucketing across [0, n_lines=10] collapses everything to bucket 0.
        # bucket over the realistic cost range instead.
        cost = float(getattr(self.awi.profile, "cost", 0) or 0)
        my_type = bucket_type(cost, 0.0, 5.0)

        last_bucket: Optional[tuple[int, int]] = None
        if last_offer is not None:
            p_issue = nmi.issues[UNIT_PRICE]
            last_bucket = (
                bucket_qty(int(last_offer[QUANTITY])),
                bucket_price(
                    float(last_offer[UNIT_PRICE]),
                    float(p_issue.min_value),
                    float(p_issue.max_value),
                ),
            )

        # cap my_history to the length the EFG expects at this abstract round.
        # at round R for a player with offset O (0=seller, 1=buyer), the player
        # has already taken (R - O) // 2 prior actions in the abstract game,
        # so the history at *decision time* has exactly that length.
        offset = 0 if self._is_seller(nid) else 1
        expected_hist_len = max(0, (rnd - offset) // 2)
        full_history = self._my_history.get(nid, ())
        history = tuple(full_history[-expected_hist_len:]) if expected_hist_len else ()

        # number of OTHER concurrent partners today (beyond this bilateral),
        # clipped into the EFG's N_OTHER_VALUES grid.
        n_other_raw = max(0, len(self.negotiators) - 1)
        n_other = max(N_OTHER_VALUES[0], min(N_OTHER_VALUES[-1], n_other_raw))
        n_other_idx = N_OTHER_VALUES.index(n_other)

        try:
            return InfosetKey(
                role="S" if self._is_seller(nid) else "B",
                my_type=my_type,
                my_exog=my_exog,
                n_other_idx=n_other_idx,
                round=rnd,
                last_offer=last_bucket,
                my_history=history,
            )
        except ValueError:
            return None

    def _lookup(self, key: InfosetKey) -> Optional[tuple[str, str]]:
        """returns (action_str, kind) sampled from the policy, or None on miss.
        kind ∈ {'accept','end','offer'}."""
        label = key.serialize()
        entries = self._policy.get(label)
        if not entries:
            self._policy_misses += 1
            self._missed_keys[label] = self._missed_keys.get(label, 0) + 1
            return None
        self._policy_hits += 1
        names = [e[0] for e in entries]
        weights = [e[1] for e in entries]
        choice = self._rng.choices(names, weights=weights, k=1)[0]
        if choice == "acc":
            return (choice, "accept")
        if choice == "end":
            return (choice, "end")
        return (choice, "offer")

    # ------------------------------------------------------------------
    # propose / react glue (handles policy + fallback)
    # ------------------------------------------------------------------

    def _propose(self, nid: str, last_offer: Optional[Outcome]) -> Outcome | None:
        """build my next offer for nid. consumes one decision slot for that partner."""
        key = self._build_key(nid, last_offer)
        sampled = None
        offer = None
        if key is not None:
            sampled = self._lookup(key)
            if sampled is not None:
                action_str, kind = sampled
                if kind == "offer":
                    offer = self._action_to_offer(nid, action_str)
                # accept/end are not meaningful as a proposal; fall through to fallback
        fallback_used = offer is None
        if offer is None:
            offer = self._fallback_offer(nid)
        if offer is not None:
            self._record_my_offer(nid, offer)
        self._log(
            type="propose",
            day=self.awi.current_step,
            partner=nid,
            role="S" if self._is_seller(nid) else "B",
            decision_idx=self._decision_count.get(nid, 0),
            abstract_round=self._abstract_round(nid),
            key=key.serialize() if key is not None else None,
            key_in_range=key is not None,
            hit=sampled is not None,
            sampled_action=(sampled[0] if sampled is not None else None),
            fallback=fallback_used,
            offer=list(offer) if offer is not None else None,
            opp_last_offer=list(last_offer) if last_offer is not None else None,
            need=self._need(nid),
        )
        self._decision_count[nid] = self._decision_count.get(nid, 0) + 1
        return offer

    def _react(
        self, nid: str, offer: Outcome
    ) -> tuple[str, Optional[Outcome]]:
        """returns (decision, counter_offer).
        decision ∈ {'accept','end','counter'}.
        counter_offer is the SCML offer to send IF decision == 'counter' AND
        the policy sampled an offer; otherwise None (caller falls back).

        consumes one decision slot only when decision != 'counter' (the caller
        consumes it for counters since it then logically continues the round)."""
        key = self._build_key(nid, offer)
        sampled = self._lookup(key) if key is not None else None

        decision: str
        counter_offer: Optional[Outcome] = None
        if sampled is not None:
            action_str, kind = sampled
            if kind == "accept":
                decision = "accept"
            elif kind == "end":
                decision = "end"
            else:
                decision = "counter"
                counter_offer = self._action_to_offer(nid, action_str)
        else:
            decision = "accept" if self._fallback_should_accept(nid, offer) else "counter"

        self._log(
            type="react",
            day=self.awi.current_step,
            partner=nid,
            role="S" if self._is_seller(nid) else "B",
            decision_idx=self._decision_count.get(nid, 0),
            abstract_round=self._abstract_round(nid),
            key=key.serialize() if key is not None else None,
            key_in_range=key is not None,
            hit=sampled is not None,
            sampled_action=(sampled[0] if sampled is not None else None),
            fallback=sampled is None,
            decision=decision,
            opp_offer=list(offer),
            need=self._need(nid),
        )

        if decision != "counter":
            self._decision_count[nid] = self._decision_count.get(nid, 0) + 1
        return decision, counter_offer

    def _action_to_offer(self, nid: str, action_str: str) -> Outcome | None:
        """map a policy action label like 'o21' back to a real SCML offer tuple.
        the abstract grid (q∈{1..N_QTY}, p∈{low,high}) is projected onto the
        nmi's actual issue ranges."""
        if not (len(action_str) == 3 and action_str[0] == "o"):
            return None
        q_bucket = int(action_str[1])
        p_bucket = int(action_str[2])
        nmi = self.get_nmi(nid)
        if nmi is None:
            return None
        q_issue = nmi.issues[QUANTITY]
        p_issue = nmi.issues[UNIT_PRICE]
        raw_q = q_bucket + 1
        q = max(int(q_issue.min_value), min(int(q_issue.max_value), raw_q))
        p = int(p_issue.min_value) if p_bucket == 0 else int(p_issue.max_value)
        offer = [-1, -1, -1]
        offer[QUANTITY] = q
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = p
        return tuple(offer)

    def _record_my_offer(self, nid: str, offer: Outcome) -> None:
        nmi = self.get_nmi(nid)
        if nmi is None:
            return
        p_issue = nmi.issues[UNIT_PRICE]
        q_b = bucket_qty(int(offer[QUANTITY]))
        p_b = bucket_price(
            float(offer[UNIT_PRICE]),
            float(p_issue.min_value),
            float(p_issue.max_value),
        )
        self._my_history.setdefault(nid, []).append((q_b, p_b))

    # ------------------------------------------------------------------
    # fallback strategy (phase 1; used on miss / past K rounds)
    # ------------------------------------------------------------------

    def _fallback_offer(self, nid: str) -> Outcome | None:
        nmi = self.get_nmi(nid)
        if nmi is None:
            return None
        need = max(1, self._need(nid))
        q_issue = nmi.issues[QUANTITY]
        p_issue = nmi.issues[UNIT_PRICE]
        q = max(int(q_issue.min_value), min(int(q_issue.max_value), need))
        p = int(p_issue.max_value) if self._is_seller(nid) else int(p_issue.min_value)
        offer = [-1, -1, -1]
        offer[QUANTITY] = q
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = p
        return tuple(offer)

    def _fallback_should_accept(self, nid: str, offer: Outcome) -> bool:
        if self._need(nid) <= 0:
            return False
        if offer[QUANTITY] > self._need(nid):
            return False
        price = offer[UNIT_PRICE]
        if self._is_seller(nid):
            return self.ufun.ok_to_sell_at(price)
        return self.ufun.ok_to_buy_at(price)


# ======================================================================
# v7: EFRHybridAgent — winner-style pipeline with EFR-scored subset selection.
#
# architecture (learned from SCML 2024 winners — CautiousOneShotAgent,
# MatchingPennies, DistRedistAgent, EpsilonGreedyAgent):
#   - OneShotSyncAgent base
#   - first proposals: distribute needs evenly across partners at best price
#   - accept phase: enumerate powerset of current offers, score each subset
#   - counter phase: redistribute any residual need to rejected partners
#   - price: always hold best-for-me (no concession)
#
# EFR's role: subset scoring is (primary) diff from need, (secondary) the
# sum of EFR-policy accept-probabilities on the subset's offers. winners
# use hand-tuned loss functions here; EFR replaces that with an
# equilibrium-backed signal. set `use_efr=False` to get the pure-heuristic
# variant (ablation control for the research comparison).
# ======================================================================

from itertools import chain, combinations

from scml.common import distribute as _scml_distribute
from scml.oneshot.common import is_system_agent


def _powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class EFRHybridAgent(EFROneShotAgent):
    """winner-style powerset-subset agent with EFR-informed scoring.

    subset score is a lambda-weighted combination:
        score = -(1 - EFR_LAMBDA) * diff_norm + EFR_LAMBDA * efr_norm
      where
        diff_norm = |Σ_q(subset) - needs| / max(1, needs)   (lower better, clamp 1)
        efr_norm  = mean over subset of policy's 'acc' prob  (higher better, in [0,1])

    EFR_LAMBDA sweep interpretation:
        0.0  → pure winner pipeline, diff minimization only.
        0.1  → EFR as a light nudge on top of diff.
        0.5  → balanced.
        0.9  → EFR dominates, diff barely matters.
        1.0  → pure EFR acceptance signal (any q is fine, trust equilibrium).

    earlier tiebreak-only scoring made EFR invisible because true ties are rare
    in powerset-over-current-offers; this version always gives EFR a real weight.
    """

    # class-level flag; subclass and override to disable EFR for ablation
    USE_EFR_SCORING: bool = True
    EFR_LAMBDA: float = 0.1

    # cap on powerset size — 2^8 = 256 subsets. if more partners, greedy fallback.
    POWERSET_MAX_PARTNERS: int = 8

    # if >= 0, use EFR's "end" probability to prune partners from counter-offer
    # redistribution: partners whose policy-end prob is >= threshold get END.
    # set to -1 to disable (behavior then = always counter leftover partners).
    # this is the "EFR as a different decision" experiment per v7.5 — maybe
    # EFR can identify partners not worth engaging, even if it can't improve
    # the accept-subset search.
    END_PROB_THRESHOLD: float = -1.0

    def init(self) -> None:
        super().init()
        # cumulative metrics specific to the hybrid pipeline
        self._hybrid_accepts = 0
        self._hybrid_subsets_scored = 0

    # ------------------------------------------------------------------
    # sync agent interface — winner-style
    # ------------------------------------------------------------------

    def first_proposals(self) -> dict[str, Outcome | None]:
        """best price, distribute needs evenly across active partners."""
        out: dict[str, Outcome | None] = {}
        for needs, all_partners, issues, is_sell in self._sides():
            partners = [p for p in all_partners if p in self.negotiators
                        and not is_system_agent(p)]
            if not partners:
                continue
            if needs <= 0:
                for p in partners:
                    out[p] = None
                continue
            best_price = (
                int(issues[UNIT_PRICE].max_value) if is_sell
                else int(issues[UNIT_PRICE].min_value)
            )
            qs = _scml_distribute(
                int(needs), len(partners),
                equal=True,
                allow_zero=self.awi.allow_zero_quantity,
            )
            for p, q in zip(partners, qs, strict=False):
                out[p] = self._make_offer(q, best_price) if q > 0 else None
        return out

    def counter_all(
        self,
        offers: dict[str, Outcome | None],
        states: dict[str, SAOState],
    ) -> dict[str, SAOResponse]:
        response: dict[str, SAOResponse] = {}

        # drop future-step offers (rare in oneshot but possible)
        offers = {
            p: o for p, o in offers.items()
            if o is None or o[TIME] == self.awi.current_step
        }

        for needs, all_partners, issues, is_sell in self._sides():
            active = [p for p in all_partners if p in offers
                      and not is_system_agent(p)]
            if not active:
                continue

            # real offers (not None) available on this side
            side_offers = {p: offers[p] for p in active if offers[p] is not None}
            no_offer = [p for p in active if offers[p] is None]

            # pick best subset of side_offers to accept
            best_subset, best_score = self._best_subset(
                side_offers, needs, is_sell, issues,
            )
            self._hybrid_subsets_scored += 1

            accepted_q = sum(side_offers[p][QUANTITY] for p in best_subset)
            remaining = max(0, int(needs) - accepted_q)

            for p in best_subset:
                response[p] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
                self._hybrid_accepts += 1

            # partners we didn't accept from — counter-offer the deficit, or end
            leftover = [p for p in side_offers if p not in best_subset] + no_offer

            # EFR end-signal pruning: if enabled, pull partners whose policy
            # strongly suggests ENDing out of the counter-offer pool and end
            # them directly. forces the remaining need to concentrate on the
            # partners where EFR thinks engagement is still productive.
            end_from_efr: list[str] = []
            if self.END_PROB_THRESHOLD >= 0.0 and leftover:
                keep = []
                for p in leftover:
                    offer = side_offers.get(p)
                    if offer is not None and self._efr_end_prob(p, offer) >= self.END_PROB_THRESHOLD:
                        end_from_efr.append(p)
                    else:
                        keep.append(p)
                leftover = keep

            for p in end_from_efr:
                response[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)

            if remaining > 0 and leftover:
                qs = _scml_distribute(
                    remaining, len(leftover),
                    equal=True, allow_zero=self.awi.allow_zero_quantity,
                )
                best_price = (
                    int(issues[UNIT_PRICE].max_value) if is_sell
                    else int(issues[UNIT_PRICE].min_value)
                )
                for p, q in zip(leftover, qs, strict=False):
                    if q > 0:
                        response[p] = SAOResponse(
                            ResponseType.REJECT_OFFER,
                            self._make_offer(q, best_price),
                        )
                    else:
                        response[p] = SAOResponse(
                            ResponseType.END_NEGOTIATION, None
                        )
            else:
                # filled (or EFR ended everyone) — end leftovers
                for p in leftover:
                    response[p] = SAOResponse(
                        ResponseType.END_NEGOTIATION, None
                    )

        # safety net: counter_all must return an entry for every key in offers
        for p in offers:
            if p not in response:
                response[p] = SAOResponse(ResponseType.END_NEGOTIATION, None)

        return response

    # ------------------------------------------------------------------
    # subset scoring
    # ------------------------------------------------------------------

    def _best_subset(
        self,
        side_offers: dict[str, Outcome],
        needs: int,
        is_sell: bool,
        issues,
    ) -> tuple[tuple[str, ...], float]:
        """lambda-weighted score combining diff-minimization with EFR signal."""
        partners = list(side_offers.keys())
        if len(partners) > self.POWERSET_MAX_PARTNERS:
            partners = sorted(
                partners,
                key=lambda p: -self._efr_accept_prob(p, side_offers[p]),
            )[: self.POWERSET_MAX_PARTNERS]

        needs_denom = max(1, int(needs))
        lam = float(self.EFR_LAMBDA) if self.USE_EFR_SCORING else 0.0

        best_subset: tuple[str, ...] = ()
        best_score = float("-inf")

        for subset in _powerset(partners):
            q_total = sum(side_offers[p][QUANTITY] for p in subset)
            diff_norm = min(1.0, abs(q_total - int(needs)) / needs_denom)
            if subset and lam > 0.0:
                efr_norm = sum(
                    self._efr_accept_prob(p, side_offers[p]) for p in subset
                ) / len(subset)
            else:
                efr_norm = 0.0

            score = -(1.0 - lam) * diff_norm + lam * efr_norm
            if score > best_score:
                best_score = score
                best_subset = subset

        return best_subset, best_score

    def _efr_accept_prob(self, nid: str, offer: Outcome) -> float:
        """probability the EFR policy assigns to 'accept' in this state.
        returns 0.5 on policy miss so the hybrid still runs without the policy."""
        key = self._build_key(nid, offer)
        if key is None:
            return 0.5
        entries = self._policy.get(key.serialize())
        if not entries:
            return 0.5
        for name, prob in entries:
            if name == "acc":
                return prob
        return 0.0

    def _efr_end_prob(self, nid: str, offer: Outcome) -> float:
        """probability the EFR policy assigns to 'end' in this state.
        returns 0.0 on policy miss (don't end partners we can't reason about)."""
        key = self._build_key(nid, offer)
        if key is None:
            return 0.0
        entries = self._policy.get(key.serialize())
        if not entries:
            return 0.0
        for name, prob in entries:
            if name == "end":
                return prob
        return 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sides(self):
        """yields (needs, partner_list, issues, is_sell) for each active side.
        for first/last-level oneshot agents only one side is active per day."""
        return (
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
                False,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
                True,
            ),
        )

    def _make_offer(self, q: int, p: int) -> Outcome:
        offer = [-1, -1, -1]
        offer[QUANTITY] = int(q)
        offer[TIME] = self.awi.current_step
        offer[UNIT_PRICE] = int(p)
        return tuple(offer)


class EFRHybridAgentNoEFR(EFRHybridAgent):
    """ablation: same hybrid pipeline but subset scoring ignores EFR policy.
    controls for whether EFR adds value over the pure winner-style architecture."""

    USE_EFR_SCORING = False
    EFR_LAMBDA = 0.0


# lambda-sweep variants for the research ablation. lam ∈ [0, 1]:
#   0 = diff-minimization only (same as NoEFR semantically)
#   1 = EFR accept-prob only (ignore diff)

class EFRHybrid_L01(EFRHybridAgent):
    EFR_LAMBDA = 0.1


class EFRHybrid_L03(EFRHybridAgent):
    EFR_LAMBDA = 0.3


class EFRHybrid_L05(EFRHybridAgent):
    EFR_LAMBDA = 0.5


class EFRHybrid_L07(EFRHybridAgent):
    EFR_LAMBDA = 0.7


class EFRHybrid_L09(EFRHybridAgent):
    EFR_LAMBDA = 0.9


class EFRHybrid_L10(EFRHybridAgent):
    EFR_LAMBDA = 1.0


# end-threshold sweep — keep λ=0 (winner-pipeline scoring), add EFR-driven
# negotiation ending. hypothesis: EFR's "end" probability identifies partners
# not worth engaging; ending them concentrates remaining need on good partners.

class EFRHybrid_End03(EFRHybridAgent):
    EFR_LAMBDA = 0.0
    END_PROB_THRESHOLD = 0.3


class EFRHybrid_End05(EFRHybridAgent):
    EFR_LAMBDA = 0.0
    END_PROB_THRESHOLD = 0.5


class EFRHybrid_End07(EFRHybridAgent):
    EFR_LAMBDA = 0.0
    END_PROB_THRESHOLD = 0.7


class EFRHybrid_End09(EFRHybridAgent):
    EFR_LAMBDA = 0.0
    END_PROB_THRESHOLD = 0.9


# ======================================================================
# v8: EFR3PHybridAgent — winner pipeline augmented with the 3-player policy.
#
# the 3-player EFG solves for a CCE over Seller + Buyer1 + Buyer2, so the
# seller's equilibrium strategy accounts for real multi-partner coupling:
# demanding too much from one partner is punished by the OTHER partner's
# rational response (accept/reject given their belief about seller's supply
# commitment to the first partner).
#
# deployment:
#   - first proposals (seller side): look up S|type|exog → sample joint
#     action (q_A, q_B, price). apply q_A to partner 0 and q_B to partner 1.
#     for partners ≥ 2, fall back to distribute-evenly at the same price.
#   - subset scoring (accept decision): use the 3-player buyer policy's
#     acc probability instead of the 2-player policy's. lambda-weighted
#     same as the 2-player hybrid.
#
# buyer side (when we're the buyer, not the seller in the day): the B1/B2
# policies are symmetric; look up B1|type|exog|qidx|pidx → get accept prob.
# ======================================================================

# 3-player EFG action grid must match build_game_3p.py exactly
QUANTITIES_3P = (1, 2, 3, 4)
PRICES_3P = (9, 10)


DEFAULT_POLICY_3P_PATH = Path(__file__).parent / "policies" / "scml_oneshot_v2_3p.policy"


class EFR3PHybridAgent(EFRHybridAgent):
    """hybrid pipeline using the 3-player EFR policy for seller first offers
    and for buyer-side acceptance probabilities."""

    EFR_LAMBDA = 0.1     # weight of 3p signal in subset scoring
    USE_3P_FIRST_PROPOSALS = True

    def init(self) -> None:
        super().init()
        # load 3-player policy; fall back to empty dict (then behaves as parent)
        self._policy_3p: dict[str, list[tuple[str, float]]] = {}
        if DEFAULT_POLICY_3P_PATH.exists():
            self._policy_3p = load_policy(DEFAULT_POLICY_3P_PATH)
        self._log(
            type="init_3p",
            policy_3p_loaded=bool(self._policy_3p),
            policy_3p_size=len(self._policy_3p),
            use_3p_first_proposals=self.USE_3P_FIRST_PROPOSALS,
        )

    # ------------------------------------------------------------------
    # bucketing helpers, symmetric with build_game_3p.py
    # ------------------------------------------------------------------

    def _3p_seller_bucket(self) -> tuple[int, int]:
        """returns (s_type, s_exog) buckets for the 3p EFG."""
        cost = float(getattr(self.awi.profile, "cost", 0) or 0)
        s_type = bucket_type(cost, 0.0, 5.0)
        exog_q = int(self.awi.current_exogenous_input_quantity or 0)
        max_q = int(getattr(self.awi, "n_lines", 10) or 10)
        s_exog = bucket_exog(exog_q, 0, max_q)
        return s_type, s_exog

    def _3p_buyer_bucket(self) -> tuple[int, int]:
        """returns (b_type, b_exog) buckets for the 3p EFG (buyer side)."""
        cost = float(getattr(self.awi.profile, "cost", 0) or 0)
        b_type = bucket_type(cost, 0.0, 5.0)
        exog_q = int(self.awi.current_exogenous_output_quantity or 0)
        max_q = int(getattr(self.awi, "n_lines", 10) or 10)
        b_exog = bucket_exog(exog_q, 0, max_q)
        return b_type, b_exog

    # ------------------------------------------------------------------
    # 3p policy lookups
    # ------------------------------------------------------------------

    def _sample_3p_seller_action(self) -> Optional[tuple[int, int, int]]:
        """sample (qa_idx, qb_idx, p_idx) from seller's 3p infoset, or None on miss."""
        if not self._policy_3p:
            return None
        s_type, s_exog = self._3p_seller_bucket()
        label = f"S|{s_type}|{s_exog}"
        entries = self._policy_3p.get(label)
        if not entries:
            return None
        names = [e[0] for e in entries]
        weights = [e[1] for e in entries]
        choice = self._rng.choices(names, weights=weights, k=1)[0]
        # action label format: "s{qa}{qb}{p}"
        if not (len(choice) == 4 and choice[0] == "s"):
            return None
        try:
            return int(choice[1]), int(choice[2]), int(choice[3])
        except ValueError:
            return None

    def _3p_buyer_accept_prob(self, q_idx: int, p_idx: int) -> float:
        """look up the symmetric B1 policy's accept probability for this offer.
        returns 0.5 on miss (neutral). we use B1 (== B2 by symmetry in the EFG)."""
        if not self._policy_3p:
            return 0.5
        b_type, b_exog = self._3p_buyer_bucket()
        label = f"B1|{b_type}|{b_exog}|{q_idx}{p_idx}"
        entries = self._policy_3p.get(label)
        if not entries:
            return 0.5
        for name, prob in entries:
            if name == "acc":
                return prob
        return 0.0

    # ------------------------------------------------------------------
    # override: first_proposals uses 3p seller joint action for first 2 partners
    # ------------------------------------------------------------------

    def first_proposals(self) -> dict[str, Outcome | None]:
        out: dict[str, Outcome | None] = {}
        for needs, all_partners, issues, is_sell in self._sides():
            partners = [p for p in all_partners if p in self.negotiators
                        and not is_system_agent(p)]
            if not partners or needs <= 0:
                for p in partners:
                    out[p] = None
                continue

            best_price = (
                int(issues[UNIT_PRICE].max_value) if is_sell
                else int(issues[UNIT_PRICE].min_value)
            )

            used_3p = False
            if (self.USE_3P_FIRST_PROPOSALS and is_sell
                    and len(partners) >= 2 and self._policy_3p):
                sampled = self._sample_3p_seller_action()
                if sampled is not None:
                    qa_idx, qb_idx, p_idx = sampled
                    q_a = QUANTITIES_3P[qa_idx]
                    q_b = QUANTITIES_3P[qb_idx]
                    # map abstract price to the side's real issue range
                    p_val = (
                        int(issues[UNIT_PRICE].max_value) if p_idx == 1
                        else int(issues[UNIT_PRICE].min_value)
                    )
                    # clip quantities to the nmi's issue range
                    q_issue = issues[QUANTITY]
                    q_a = max(int(q_issue.min_value), min(int(q_issue.max_value), q_a))
                    q_b = max(int(q_issue.min_value), min(int(q_issue.max_value), q_b))
                    out[partners[0]] = self._make_offer(q_a, p_val)
                    out[partners[1]] = self._make_offer(q_b, p_val)
                    used_3p = True
                    # distribute the residual need across remaining partners
                    rem_partners = partners[2:]
                    if rem_partners:
                        remaining = max(0, int(needs) - q_a - q_b)
                        if remaining > 0:
                            qs = _scml_distribute(
                                remaining, len(rem_partners),
                                equal=True, allow_zero=self.awi.allow_zero_quantity,
                            )
                            for p, q in zip(rem_partners, qs, strict=False):
                                out[p] = (
                                    self._make_offer(q, best_price)
                                    if q > 0 else None
                                )
                        else:
                            for p in rem_partners:
                                out[p] = None

            if not used_3p:
                # fall back to winner-style even distribution
                qs = _scml_distribute(
                    int(needs), len(partners),
                    equal=True, allow_zero=self.awi.allow_zero_quantity,
                )
                for p, q in zip(partners, qs, strict=False):
                    out[p] = self._make_offer(q, best_price) if q > 0 else None
        return out

    # ------------------------------------------------------------------
    # override: subset scoring uses 3p buyer policy's accept prob
    # ------------------------------------------------------------------

    def _efr_accept_prob(self, nid: str, offer: Outcome) -> float:
        """3p buyer policy's accept probability for this offer,
        measuring 'would a buyer symmetric to me want to accept this q,p?'.
        falls back to parent (2p policy) if 3p policy isn't loaded."""
        if not self._policy_3p:
            return super()._efr_accept_prob(nid, offer)
        # quantize offer into abstract grid
        q_idx = bucket_qty(int(offer[QUANTITY]))
        # price bucket: low or high relative to the negotiation's issue range
        nmi = self.get_nmi(nid)
        if nmi is None:
            return 0.5
        p_issue = nmi.issues[UNIT_PRICE]
        p_idx = bucket_price(
            float(offer[UNIT_PRICE]),
            float(p_issue.min_value), float(p_issue.max_value),
        )
        return self._3p_buyer_accept_prob(q_idx, p_idx)


class EFR3PHybrid_NoFirst(EFR3PHybridAgent):
    """ablation: uses 3p buyer policy for subset scoring but NOT for first
    proposals. isolates the contribution of the seller's joint-action policy."""

    USE_3P_FIRST_PROPOSALS = False


class EFR3PHybrid_NoAccept(EFR3PHybridAgent):
    """ablation: uses 3p seller policy for first proposals but sets λ=0, so
    subset scoring is pure diff-minimization. isolates contribution of the
    seller joint policy alone."""

    EFR_LAMBDA = 0.0
