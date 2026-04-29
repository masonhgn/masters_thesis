"""
test suite to verify python deal_or_no_deal matches c++ bargaining game.
"""

import sys
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/open_spiel-private/build/python')
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/games')

import pyspiel
import numpy as np
from deal_or_no_deal import DealOrNoDealGame


def test_instances_match():
    """verify both games load the same instances"""
    print("testing instance loading...")

    cpp_game = pyspiel.load_game("bargaining")
    py_game = DealOrNoDealGame()

    # check number of instances
    num_instances = min(len(py_game.all_instances), 100)
    print(f"  comparing first {num_instances} instances...")

    for i in range(num_instances):
        # get cpp instance through a state
        cpp_state = cpp_game.new_initial_state()
        cpp_state.apply_action(i)
        cpp_info = cpp_state.information_state_string(0)

        # get python instance
        py_instance = py_game.get_instance(i)

        # compare pool and values by parsing cpp string
        # cpp format: "Pool: Book: X, Hat: Y, Basketball: Z\nMy values: Book: A, Hat: B, Basketball: C\n..."
        lines = cpp_info.split('\n')
        cpp_pool_line = lines[0]
        cpp_values_line = lines[1]

        # extract pool values
        cpp_pool = [
            int(cpp_pool_line.split("Book: ")[1].split(",")[0]),
            int(cpp_pool_line.split("Hat: ")[1].split(",")[0]),
            int(cpp_pool_line.split("Basketball: ")[1])
        ]

        # extract player 0 values
        cpp_p0_values = [
            int(cpp_values_line.split("Book: ")[1].split(",")[0]),
            int(cpp_values_line.split("Hat: ")[1].split(",")[0]),
            int(cpp_values_line.split("Basketball: ")[1])
        ]

        # compare
        if cpp_pool != py_instance.pool:
            print(f"  FAIL: instance {i} pool mismatch: cpp={cpp_pool}, py={py_instance.pool}")
            return False

        if cpp_p0_values != py_instance.values[0]:
            print(f"  FAIL: instance {i} p0 values mismatch: cpp={cpp_p0_values}, py={py_instance.values[0]}")
            return False

    print("  PASS: all instances match")
    return True


def test_offers_match():
    """verify both games generate the same offers"""
    print("testing offer generation...")

    cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
    py_game = DealOrNoDealGame({"max_num_instances": 100})

    cpp_num_offers = cpp_game.num_distinct_actions() - 1  # subtract accept action
    py_num_offers = len(py_game.all_offers)

    if cpp_num_offers != py_num_offers:
        print(f"  FAIL: number of offers mismatch: cpp={cpp_num_offers}, py={py_num_offers}")
        return False

    print(f"  PASS: both games have {py_num_offers} offers")
    return True


def test_basic_gameplay():
    """test basic game progression with a simple scenario"""
    print("testing basic gameplay...")

    cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
    py_game = DealOrNoDealGame({"max_num_instances": 100})

    # start both games with instance 0
    cpp_state = cpp_game.new_initial_state()
    py_state = py_game.new_initial_state()

    cpp_state.apply_action(0)  # select instance 0
    py_state.apply_action(0)

    # check current player is 0
    if cpp_state.current_player() != 0 or py_state.current_player() != 0:
        print(f"  FAIL: initial player mismatch: cpp={cpp_state.current_player()}, py={py_state.current_player()}")
        return False

    # player 0 makes offer [0,0,0] (action 0 in both games)
    cpp_state.apply_action(0)
    py_state.apply_action(0)

    # check current player switched to 1
    if cpp_state.current_player() != 1 or py_state.current_player() != 1:
        print(f"  FAIL: player switch mismatch: cpp={cpp_state.current_player()}, py={py_state.current_player()}")
        return False

    # player 1 accepts (last action)
    accept_action = cpp_game.num_distinct_actions() - 1
    cpp_state.apply_action(accept_action)
    py_state.apply_action(accept_action)

    # both should be terminal
    if not cpp_state.is_terminal() or not py_state.is_terminal():
        print(f"  FAIL: terminal state mismatch: cpp={cpp_state.is_terminal()}, py={py_state.is_terminal()}")
        return False

    # compare returns
    cpp_returns = cpp_state.returns()
    py_returns = py_state.returns()

    if not np.allclose(cpp_returns, py_returns, atol=1e-6):
        print(f"  FAIL: returns mismatch: cpp={cpp_returns}, py={py_returns}")
        return False

    print(f"  PASS: basic gameplay matches (returns={py_returns})")
    return True


def test_returns_calculation():
    """test returns calculation for various scenarios"""
    print("testing returns calculation...")

    cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
    py_game = DealOrNoDealGame({"max_num_instances": 100})

    test_cases = [
        # (instance_idx, offer_sequence, accept_after)
        (0, [0], 1),      # p0 offers, p1 accepts
        (0, [0, 5], 2),   # p0 offers, p1 offers, p0 accepts
        (5, [10], 1),     # different instance
    ]

    for instance_idx, offers, accept_after in test_cases:
        cpp_state = cpp_game.new_initial_state()
        py_state = py_game.new_initial_state()

        # select instance
        cpp_state.apply_action(instance_idx)
        py_state.apply_action(instance_idx)

        # make offers
        for offer_action in offers:
            cpp_state.apply_action(offer_action)
            py_state.apply_action(offer_action)

        # accept
        accept_action = cpp_game.num_distinct_actions() - 1
        cpp_state.apply_action(accept_action)
        py_state.apply_action(accept_action)

        # compare returns
        cpp_returns = cpp_state.returns()
        py_returns = py_state.returns()

        if not np.allclose(cpp_returns, py_returns, atol=1e-6):
            print(f"  FAIL: returns mismatch for instance {instance_idx}, offers {offers}:")
            print(f"     cpp={cpp_returns}, py={py_returns}")
            return False

    print("  PASS: all return calculations match")
    return True


def test_legal_actions():
    """test legal actions at various game states"""
    print("testing legal actions...")

    cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
    py_game = DealOrNoDealGame({"max_num_instances": 100})

    # initial chance node
    cpp_state = cpp_game.new_initial_state()
    py_state = py_game.new_initial_state()

    cpp_legal = sorted(cpp_state.legal_actions())
    py_legal = sorted(py_state.legal_actions())

    if cpp_legal != py_legal:
        print(f"  FAIL: legal actions mismatch at initial chance node")
        print(f"     cpp has {len(cpp_legal)} actions, py has {len(py_legal)} actions")
        return False

    # after selecting instance
    cpp_state.apply_action(0)
    py_state.apply_action(0)

    cpp_legal = sorted(cpp_state.legal_actions())
    py_legal = sorted(py_state.legal_actions())

    # should not include accept action yet (no offers on table)
    accept_action = cpp_game.num_distinct_actions() - 1
    if accept_action in cpp_legal or accept_action in py_legal:
        print(f"  FAIL: accept action should not be legal before any offers")
        return False

    # after one offer
    cpp_state.apply_action(0)
    py_state.apply_action(0)

    cpp_legal = sorted(cpp_state.legal_actions())
    py_legal = sorted(py_state.legal_actions())

    # should include accept action now
    if accept_action not in cpp_legal or accept_action not in py_legal:
        print(f"  FAIL: accept action should be legal after an offer")
        return False

    if cpp_legal != py_legal:
        print(f"  FAIL: legal actions mismatch after first offer")
        print(f"     cpp={len(cpp_legal)} actions, py={len(py_legal)} actions")
        return False

    print("  PASS: legal actions match")
    return True


def test_information_state_tensors():
    """test information state tensor encoding"""
    print("testing information state tensors...")

    cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
    py_game = DealOrNoDealGame({"max_num_instances": 100})

    # create a game state with some history
    cpp_state = cpp_game.new_initial_state()
    py_state = py_game.new_initial_state()

    # select instance 0
    cpp_state.apply_action(0)
    py_state.apply_action(0)

    # make a few offers
    cpp_state.apply_action(0)  # p0 offers [0,0,0]
    py_state.apply_action(0)

    cpp_state.apply_action(5)  # p1 offers something
    py_state.apply_action(5)

    # compare information state tensors for both players
    for player in [0, 1]:
        cpp_tensor = np.array(cpp_state.information_state_tensor(player))
        py_tensor = py_state.information_state_tensor(player)

        if cpp_tensor.shape != py_tensor.shape:
            print(f"  FAIL: tensor shape mismatch for player {player}:")
            print(f"     cpp={cpp_tensor.shape}, py={py_tensor.shape}")
            return False

        if not np.allclose(cpp_tensor, py_tensor, atol=1e-6):
            # find first difference
            diff_indices = np.where(~np.isclose(cpp_tensor, py_tensor, atol=1e-6))[0]
            print(f"  FAIL: tensor values mismatch for player {player}:")
            print(f"     first difference at index {diff_indices[0]}")
            print(f"     cpp[{diff_indices[0]}]={cpp_tensor[diff_indices[0]]}")
            print(f"     py[{diff_indices[0]}]={py_tensor[diff_indices[0]]}")
            return False

    print("  PASS: information state tensors match")
    return True


def test_observation_tensors():
    """test observation tensor encoding"""
    print("testing observation tensors...")

    cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
    py_game = DealOrNoDealGame({"max_num_instances": 100})

    # create a game state
    cpp_state = cpp_game.new_initial_state()
    py_state = py_game.new_initial_state()

    cpp_state.apply_action(0)
    py_state.apply_action(0)

    cpp_state.apply_action(0)
    py_state.apply_action(0)

    # compare observation tensors for both players
    for player in [0, 1]:
        cpp_tensor = np.array(cpp_state.observation_tensor(player))
        py_tensor = py_state.observation_tensor(player)

        if cpp_tensor.shape != py_tensor.shape:
            print(f"  FAIL: observation tensor shape mismatch for player {player}:")
            print(f"     cpp={cpp_tensor.shape}, py={py_tensor.shape}")
            return False

        if not np.allclose(cpp_tensor, py_tensor, atol=1e-6):
            diff_indices = np.where(~np.isclose(cpp_tensor, py_tensor, atol=1e-6))[0]
            print(f"  FAIL: observation tensor values mismatch for player {player}:")
            print(f"     first difference at index {diff_indices[0]}")
            return False

    print("  PASS: observation tensors match")
    return True


def run_all_tests():
    """run all test cases"""
    print("=" * 60)
    print("testing python vs c++ game equivalence")
    print("=" * 60)

    tests = [
        test_instances_match,
        test_offers_match,
        test_basic_gameplay,
        test_returns_calculation,
        test_legal_actions,
        test_information_state_tensors,
        test_observation_tensors,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL: test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
