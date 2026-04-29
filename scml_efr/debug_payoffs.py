"""
quick sanity check on the v5 payoff model before running the 7-minute
solve. prints the marginal-payoff matrix for seller and buyer across
(type, exog, q_me, p) so we can eyeball whether the CCE will be non-degenerate.
"""

from build_game import (
    EXOG_TARGET,
    N_EXOG_BUCKETS,
    N_PRICE,
    N_QTY,
    N_TYPE_BUCKETS,
    PRICE_VALUE,
    QTY_VALUE,
    Q_OTHER_PMF,
    agreement_payoffs,
    disagreement_payoffs,
)


def main():
    print("Q_OTHER_PMF (prior on what the 2 other partners close):")
    for k in sorted(Q_OTHER_PMF):
        bar = "#" * int(Q_OTHER_PMF[k] * 100)
        print(f"  Q_other={k}: p={Q_OTHER_PMF[k]:.3f}  {bar}")
    exp = sum(k * p for k, p in Q_OTHER_PMF.items())
    print(f"  E[Q_other] = {exp:.2f}")
    print()

    print(f"EXOG_TARGET = {EXOG_TARGET}  (seller/buyer end-of-day target)")
    print(f"disagreement_payoffs = {disagreement_payoffs(0, 0)}  (marginal attribution)")
    print()

    print("marginal seller payoff matrix. rows = seller type, cols = seller exog target.")
    print("each cell shows (q, p) → payoff for the best (q, p) combo AND the full grid.")
    print("(p2_type=0, p2_exog=0 held fixed; buyer side irrelevant to seller payoff)")
    print()
    for p1_type in range(N_TYPE_BUCKETS):
        for p1_exog in range(N_EXOG_BUCKETS):
            target = EXOG_TARGET[p1_exog]
            print(f"  seller type={p1_type} (cost={7+p1_type}), exog_target={target}:")
            best = None
            for q_b in range(N_QTY):
                for p_b in range(N_PRICE):
                    s, _ = agreement_payoffs(q_b, p_b, p1_type, p1_exog, 0, 0)
                    q, p = QTY_VALUE[q_b], PRICE_VALUE[p_b]
                    marker = ""
                    if best is None or s > best[0]:
                        best = (s, q, p)
                        marker = "*"
                    print(f"    q={q} p={p}: {s:+.2f}", end="  ")
                print()
            print(f"    → best: q={best[1]} p={best[2]} payoff={best[0]:+.2f}")
            print()

    print()
    print("marginal buyer payoff matrix. rows = buyer type, cols = buyer exog target.")
    for p2_type in range(N_TYPE_BUCKETS):
        for p2_exog in range(N_EXOG_BUCKETS):
            target = EXOG_TARGET[p2_exog]
            print(f"  buyer type={p2_type} (value={10+p2_type}), exog_target={target}:")
            best = None
            for q_b in range(N_QTY):
                for p_b in range(N_PRICE):
                    _, b = agreement_payoffs(q_b, p_b, 0, 0, p2_type, p2_exog)
                    q, p = QTY_VALUE[q_b], PRICE_VALUE[p_b]
                    if best is None or b > best[0]:
                        best = (b, q, p)
                    print(f"    q={q} p={p}: {b:+.2f}", end="  ")
                print()
            print(f"    → best: q={best[1]} p={best[2]} payoff={best[0]:+.2f}")
            print()


if __name__ == "__main__":
    main()
