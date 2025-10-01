# Deal or No Deal (DoNoD) Implementation Checklist for OpenSpiel

## Phase 1 – Skeleton

- [X] Define `_GAME_TYPE` and `_GAME_INFO` (see `liars_poker.py`).
- [ ] Implement `DealOrNoDealGame(pyspiel.Game)` with `new_initial_state()`.
- [ ] Implement `DealOrNoDealState(pyspiel.State)`:
  - Track players, utilities, pool, offers, agreement status.

## Phase 2 – String State

- [ ] Implement `current_player()`, `is_terminal()`, `returns()`.
- [ ] Implement `_apply_action()` and `_legal_actions()`.
- [ ] Implement `information_state_string()` and `observation_string()`.
- [ ] Debug by playing small games manually.

## Phase 3 – CFR Compatibility

- [ ] Add `information_state_tensor()` and `observation_tensor()`.
- [ ] Decide encoding scheme:
  - Pool: cumulative one-hot over items.
  - Utilities: cumulative encoding (like bargaining).
  - Offers: either just last offer (simpler) or full sequence.
- [ ] Add `max_game_length`, `num_distinct_actions`, etc. to `_GAME_INFO`.

## Phase 4 – Testing

- [ ] Run random playthroughs.
- [ ] Run CFR with strings (sanity check).
- [ ] Run CFR with tensors (scalability check).
- [ ] Compare with paper’s expected negotiation outcomes.
