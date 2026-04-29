"""
parse the EFR agent's jsonl diagnostic log and print a digest:

  - hit/miss rate overall and broken down by role / abstract_round / decision_kind
  - top missed infoset keys
  - distribution of fallback usage
  - per-day score trajectory
  - distribution of `key_in_range` (i.e. how often we ran past K_ROUNDS)
  - sampled vs fallback offer distributions

usage: python summarize_logs.py [path]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "  -  "
    return f"{num/den:6.2%}"


def main() -> int:
    from _runlog import start as _start_log
    _start_log("summarize")

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "logs" / "efr_agent.jsonl"
    )
    if not path.exists():
        print(f"no log at {path}")
        return 1

    rows = load(path)
    print(f"loaded {len(rows):,} log lines from {path}")

    inits = [r for r in rows if r.get("type") == "init"]
    days = [r for r in rows if r.get("type") == "day"]
    steps = [r for r in rows if r.get("type") == "step_end"]
    decisions = [r for r in rows if r.get("type") in ("propose", "react")]

    print()
    print(f"  inits     : {len(inits):,}  (distinct agent instances)")
    print(f"  days      : {len(days):,}")
    print(f"  steps     : {len(steps):,}")
    print(f"  decisions : {len(decisions):,}")

    if not decisions:
        print("\nno decision rows; nothing to summarize")
        return 0

    # ---- overall hit/miss --------------------------------------------------
    total = len(decisions)
    hits = sum(1 for r in decisions if r.get("hit"))
    misses = total - hits
    in_range = sum(1 for r in decisions if r.get("key_in_range"))
    out_of_range = total - in_range

    print()
    print("== hit/miss overall ==")
    print(f"  decisions    : {total:,}")
    print(f"  in K-range   : {in_range:,} ({fmt_pct(in_range, total)})")
    print(f"  out of range : {out_of_range:,} ({fmt_pct(out_of_range, total)})")
    print(f"  hits         : {hits:,} ({fmt_pct(hits, total)})")
    print(f"  misses       : {misses:,} ({fmt_pct(misses, total)})")
    # of in-range only, what's the hit rate? (this is the "is the policy
    # actually covering the abstract states we visit" rate)
    hits_in_range = sum(1 for r in decisions if r.get("hit") and r.get("key_in_range"))
    print(f"  in-range hit : {fmt_pct(hits_in_range, in_range)}")

    # ---- by role -----------------------------------------------------------
    print()
    print("== by role ==")
    print("  role  decisions   in-range   hit-of-in-range   overall-hit")
    for role in ("S", "B"):
        sub = [r for r in decisions if r.get("role") == role]
        in_r = [r for r in sub if r.get("key_in_range")]
        hits_in_r = [r for r in in_r if r.get("hit")]
        hits_all = [r for r in sub if r.get("hit")]
        print(f"  {role}     {len(sub):>9,}   {fmt_pct(len(in_r), len(sub))}      "
              f"{fmt_pct(len(hits_in_r), len(in_r))}            {fmt_pct(len(hits_all), len(sub))}")

    # ---- by abstract round -------------------------------------------------
    print()
    print("== by abstract round ==")
    print("  round  decisions   hit-rate   miss-rate")
    rounds = sorted({r.get("abstract_round") for r in decisions if r.get("abstract_round") is not None})
    for rnd in rounds:
        sub = [r for r in decisions if r.get("abstract_round") == rnd]
        h = sum(1 for r in sub if r.get("hit"))
        print(f"  {rnd:>5}  {len(sub):>9,}   {fmt_pct(h, len(sub))}    {fmt_pct(len(sub) - h, len(sub))}")

    # ---- by kind -----------------------------------------------------------
    print()
    print("== by decision kind ==")
    for kind in ("propose", "react"):
        sub = [r for r in decisions if r.get("type") == kind]
        h = sum(1 for r in sub if r.get("hit"))
        print(f"  {kind:>7}  {len(sub):>9,}   hit={fmt_pct(h, len(sub))}")

    # ---- top missed keys ---------------------------------------------------
    print()
    print("== top 15 missed infoset keys (only in-range misses) ==")
    miss_counter = Counter()
    for r in decisions:
        if r.get("key_in_range") and not r.get("hit") and r.get("key"):
            miss_counter[r["key"]] += 1
    for key, n in miss_counter.most_common(15):
        print(f"  {n:>6,}  {key}")

    # ---- top hit keys -----------------------------------------------------
    print()
    print("== top 10 hit infoset keys ==")
    hit_counter = Counter()
    for r in decisions:
        if r.get("hit") and r.get("key"):
            hit_counter[r["key"]] += 1
    for key, n in hit_counter.most_common(10):
        print(f"  {n:>6,}  {key}")

    # ---- sampled action distribution --------------------------------------
    print()
    print("== sampled actions (when policy hit) ==")
    act_counter = Counter()
    for r in decisions:
        if r.get("hit") and r.get("sampled_action"):
            act_counter[r["sampled_action"]] += 1
    for act, n in act_counter.most_common():
        print(f"  {n:>6,}  {act}")

    # ---- react decisions ----------------------------------------------------
    print()
    print("== react outcomes (accept/end/counter) ==")
    react_counter = Counter()
    react_hit_counter = Counter()
    for r in decisions:
        if r.get("type") != "react":
            continue
        d = r.get("decision")
        react_counter[d] += 1
        if r.get("hit"):
            react_hit_counter[d] += 1
    for d in ("accept", "counter", "end"):
        n = react_counter.get(d, 0)
        nh = react_hit_counter.get(d, 0)
        print(f"  {d:>8}  {n:>7,}  (of which from policy: {nh:,})")

    # ---- score trajectory --------------------------------------------------
    if steps:
        print()
        print("== score per (agent_instance, day) ==")
        per_day_scores = defaultdict(list)
        for r in steps:
            day = r.get("day")
            if day is not None and r.get("score") is not None:
                per_day_scores[day].append(r["score"])
        for day in sorted(per_day_scores.keys())[:8]:
            scores = per_day_scores[day]
            mean = sum(scores) / len(scores)
            print(f"  day {day:>3}  n={len(scores):>3}  mean={mean:+.4f}  "
                  f"min={min(scores):+.4f}  max={max(scores):+.4f}")
        print(f"  (showing first 8 of {len(per_day_scores)} days)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
