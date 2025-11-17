#!/usr/bin/env python3
"""simple test to verify deal or no deal implementation"""

import pyspiel
import deal_or_no_deal  # import to register the game

def test_basic_game():
    """test basic game creation and initial state"""
    game = pyspiel.load_game("python_deal_or_no_deal")
    print(f"game created: {game}")
    print(f"num players: {game.num_players()}")
    print(f"num distinct actions: {game.num_distinct_actions()}")
    print(f"max chance outcomes: {game.max_chance_outcomes()}")

    # create initial state
    state = game.new_initial_state()
    print(f"\ninitial state: {state}")
    print(f"current player: {state.current_player()}")
    print(f"is chance node: {state.is_chance_node()}")
    print(f"is terminal: {state.is_terminal()}")

    # check chance outcomes
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        print(f"\nchance outcomes: {len(outcomes)} total")
        print(f"first outcome: action={outcomes[0][0]}, prob={outcomes[0][1]}")

    # apply first chance action (select instance)
    state.apply_action(0)
    print(f"\nafter selecting instance:")
    print(f"current player: {state.current_player()}")
    print(f"is chance node: {state.is_chance_node()}")

    # check legal actions for player
    legal_actions = state.legal_actions()
    print(f"legal actions for player {state.current_player()}: {len(legal_actions)} total")

    # make a simple offer
    if legal_actions:
        state.apply_action(legal_actions[0])
        print(f"\nafter first offer:")
        print(f"current player: {state.current_player()}")
        print(state.observation_string(0))

    print("\ntest passed!")

if __name__ == "__main__":
    test_basic_game()
