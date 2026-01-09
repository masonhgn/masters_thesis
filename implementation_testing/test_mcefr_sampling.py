"""
test script to verify both external and internal sampling work in mcefr
"""

import sys
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/games')
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/algos')

import pyspiel
from deal_or_no_deal import DealOrNoDealGame
from mcefr import MCEFRSolver


def test_external_sampling():
    """test external sampling mode"""
    print("testing external sampling...")

    game = DealOrNoDealGame({"max_num_instances": 10, "max_turns": 3})
    solver = MCEFRSolver(game, "blind action", sampling_mode="external")

    # run a few iterations
    for i in range(5):
        solver.iteration()
        print(f"  iteration {i+1} completed")

    # get average policy
    avg_policy = solver.average_policy()

    # check policy is valid for a sample state
    state = game.new_initial_state()
    state.apply_action(0)  # select instance 0

    action_probs = avg_policy.action_probabilities(state, player_id=0)
    prob_sum = sum(action_probs.values())

    print(f"  policy probability sum: {prob_sum:.6f}")
    print(f"  number of actions: {len(action_probs)}")

    if abs(prob_sum - 1.0) < 1e-5:
        print("  PASS: external sampling works")
        return True
    else:
        print(f"  FAIL: policy probabilities sum to {prob_sum}, expected 1.0")
        return False


def test_internal_sampling():
    """test internal sampling mode"""
    print("\ntesting internal sampling...")

    game = DealOrNoDealGame({"max_num_instances": 10, "max_turns": 3})
    solver = MCEFRSolver(game, "blind action", sampling_mode="internal")

    # run a few iterations
    for i in range(5):
        solver.iteration()
        print(f"  iteration {i+1} completed")

    # get average policy
    avg_policy = solver.average_policy()

    # check policy is valid for a sample state
    state = game.new_initial_state()
    state.apply_action(0)  # select instance 0

    action_probs = avg_policy.action_probabilities(state, player_id=0)
    prob_sum = sum(action_probs.values())

    print(f"  policy probability sum: {prob_sum:.6f}")
    print(f"  number of actions: {len(action_probs)}")

    if abs(prob_sum - 1.0) < 1e-5:
        print("  PASS: internal sampling works")
        return True
    else:
        print(f"  FAIL: policy probabilities sum to {prob_sum}, expected 1.0")
        return False


def test_comparison():
    """compare external vs internal sampling convergence"""
    print("\ncomparing external vs internal sampling...")

    game = DealOrNoDealGame({"max_num_instances": 10, "max_turns": 3})

    # run external sampling
    external_solver = MCEFRSolver(game, "blind action", sampling_mode="external")
    for i in range(10):
        external_solver.iteration()

    # run internal sampling
    internal_solver = MCEFRSolver(game, "blind action", sampling_mode="internal")
    for i in range(10):
        internal_solver.iteration()

    # check both produce valid policies
    state = game.new_initial_state()
    state.apply_action(0)

    ext_policy = external_solver.average_policy()
    int_policy = internal_solver.average_policy()

    ext_probs = ext_policy.action_probabilities(state, player_id=0)
    int_probs = int_policy.action_probabilities(state, player_id=0)

    print(f"  external policy sum: {sum(ext_probs.values()):.6f}")
    print(f"  internal policy sum: {sum(int_probs.values()):.6f}")
    print(f"  external infostate count: {len(external_solver._infostates)}")
    print(f"  internal infostate count: {len(internal_solver._infostates)}")

    # internal should visit fewer infostates (due to sampling)
    if len(internal_solver._infostates) <= len(external_solver._infostates):
        print("  PASS: internal sampling visits fewer or equal infostates")
        return True
    else:
        print("  FAIL: internal sampling visited more infostates than external")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("mcefr sampling mode test")
    print("=" * 60)

    results = []

    try:
        results.append(test_external_sampling())
    except Exception as e:
        print(f"  FAIL: external sampling test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_internal_sampling())
    except Exception as e:
        print(f"  FAIL: internal sampling test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_comparison())
    except Exception as e:
        print(f"  FAIL: comparison test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"results: {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
