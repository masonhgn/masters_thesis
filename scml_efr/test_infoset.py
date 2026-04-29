"""
unit tests for infoset.py. run with: python -m pytest test_infoset.py
or: python test_infoset.py
"""

import pytest

from infoset import (
    ACTION_ACCEPT,
    ACTION_END,
    K_ROUNDS,
    N_ACTIONS,
    N_EXOG_BUCKETS,
    N_OFFER_ACTIONS,
    N_OTHER_BUCKETS,
    N_OTHER_VALUES,
    N_PRICE,
    N_QTY,
    N_TYPE_BUCKETS,
    InfosetKey,
    bucket_exog,
    bucket_price,
    bucket_qty,
    bucket_type,
    decode_action,
    offer_action_id,
)


# --- InfosetKey roundtrip --------------------------------------------------

def test_serialize_roundtrip_with_offer():
    k = InfosetKey(
        role="S", my_type=1, my_exog=2, n_other_idx=1, round=3,
        last_offer=(2, 1), my_history=((3, 0),),
    )
    s = k.serialize()
    assert s == "S|1|2|1|3|21|30"
    assert InfosetKey.parse(s) == k


def test_serialize_roundtrip_no_offer():
    k = InfosetKey(
        role="B", my_type=0, my_exog=0, n_other_idx=0, round=0, last_offer=None,
    )
    assert k.serialize() == "B|0|0|0|0|x|x"
    assert InfosetKey.parse(k.serialize()) == k


def test_serialize_history_multi():
    k = InfosetKey(
        role="S", my_type=2, my_exog=1, n_other_idx=3, round=2,
        last_offer=(1, 0), my_history=((3, 1), (2, 0)),
    )
    assert k.serialize() == "S|2|1|3|2|10|31,20"
    assert InfosetKey.parse(k.serialize()) == k


def test_n_other_property():
    # n_other_idx=2 should map to the 3rd N_OTHER value
    k = InfosetKey(
        role="S", my_type=0, my_exog=0, n_other_idx=2, round=0, last_offer=None,
    )
    assert k.n_other == N_OTHER_VALUES[2]


def test_invalid_role():
    with pytest.raises(ValueError):
        InfosetKey(
            role="Z", my_type=0, my_exog=0, n_other_idx=0, round=0, last_offer=None,
        )


def test_invalid_round():
    with pytest.raises(ValueError):
        InfosetKey(
            role="S", my_type=0, my_exog=0, n_other_idx=0, round=K_ROUNDS,
            last_offer=None,
        )


def test_invalid_n_other_idx():
    with pytest.raises(ValueError):
        InfosetKey(
            role="S", my_type=0, my_exog=0, n_other_idx=N_OTHER_BUCKETS, round=0,
            last_offer=None,
        )


def test_invalid_offer_bucket():
    with pytest.raises(ValueError):
        InfosetKey(
            role="S", my_type=0, my_exog=0, n_other_idx=0, round=0,
            last_offer=(N_QTY, 0),
        )


# --- bucketing -------------------------------------------------------------

def test_bucket_type_edges():
    # 3 buckets over [0, 9]: bins ~ [0,3), [3,6), [6,9]
    assert bucket_type(0.0, 0.0, 9.0) == 0
    assert bucket_type(2.999, 0.0, 9.0) == 0
    assert bucket_type(4.5, 0.0, 9.0) == 1
    assert bucket_type(9.0, 0.0, 9.0) == N_TYPE_BUCKETS - 1
    # clipping
    assert bucket_type(-1.0, 0.0, 9.0) == 0
    assert bucket_type(20.0, 0.0, 9.0) == N_TYPE_BUCKETS - 1


def test_bucket_type_degenerate():
    # min == max: collapse to bucket 0
    assert bucket_type(5.0, 5.0, 5.0) == 0


def test_bucket_exog_basic():
    assert bucket_exog(0, 0, 12) == 0
    assert bucket_exog(6, 0, 12) == 1
    assert bucket_exog(12, 0, 12) == N_EXOG_BUCKETS - 1


def test_bucket_qty_clipping():
    assert bucket_qty(0) == 0
    assert bucket_qty(1) == 0
    assert bucket_qty(N_QTY) == N_QTY - 1
    assert bucket_qty(N_QTY + 5) == N_QTY - 1


def test_bucket_price_two_values():
    # only two distinct legal prices in SCML 2023+
    assert bucket_price(10.0, 10.0, 14.0) == 0
    assert bucket_price(11.9, 10.0, 14.0) == 0
    assert bucket_price(12.0, 10.0, 14.0) == 0  # midpoint goes to low
    assert bucket_price(12.1, 10.0, 14.0) == 1
    assert bucket_price(14.0, 10.0, 14.0) == 1


def test_bucket_price_degenerate():
    assert bucket_price(5.0, 5.0, 5.0) == 0


# --- action encoding -------------------------------------------------------

def test_action_count():
    assert N_OFFER_ACTIONS == N_QTY * N_PRICE
    assert N_ACTIONS == N_OFFER_ACTIONS + 2


def test_offer_action_id_unique_and_dense():
    seen = set()
    for q in range(N_QTY):
        for p in range(N_PRICE):
            aid = offer_action_id(q, p)
            assert 0 <= aid < N_OFFER_ACTIONS
            seen.add(aid)
    assert len(seen) == N_OFFER_ACTIONS


def test_decode_offer_inverse():
    for q in range(N_QTY):
        for p in range(N_PRICE):
            aid = offer_action_id(q, p)
            kind, payload = decode_action(aid)
            assert kind == "offer"
            assert payload == (q, p)


def test_decode_special_actions():
    assert decode_action(ACTION_ACCEPT) == ("accept", None)
    assert decode_action(ACTION_END) == ("end", None)


def test_decode_invalid():
    with pytest.raises(ValueError):
        decode_action(999)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
