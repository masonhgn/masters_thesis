"""test script for mcefr implementation."""

import pyspiel
from masters_thesis.dealornodeal.algos.mcefr import MCEFRSolver


def test_mcefr_kuhn_poker():
    """test mcefr on kuhn poker."""
    print("testing mcefr on kuhn poker...")
    print("=" * 60)

    # create game
    game = pyspiel.load_game("kuhn_poker")
    print(f"game: {game.get_type().long_name}")
    print(f"players: {game.num_players()}")
    print()

    # test with blind cf deviations (should be equivalent to vanilla cfr)
    print("running mcefr with blind cf deviations...")
    solver = MCEFRSolver(game, "blind cf")

    # run iterations
    num_iterations = 100
    for i in range(num_iterations):
        solver.iteration()
        if (i + 1) % 10 == 0 or i == 0:
            print(f"iteration {i + 1}/{num_iterations}")

    print("\ncomputing average policy...")
    avg_policy = solver.average_policy()

    # print some example policies
    print("\nexample policies:")
    print("-" * 60)
    state = game.new_initial_state()

    # show a few example info states
    print("\nsample of learned strategies:")
    shown = 0
    for info_state_key in list(solver._infostates.keys())[:5]:
        info_node = solver._infostates[info_state_key]
        policy = {}
        cumsum = info_node.cumulative_policy.sum()
        if cumsum > 0:
            for i, action in enumerate(info_node.legal_actions):
                policy[action] = info_node.cumulative_policy[i] / cumsum
        else:
            num_actions = len(info_node.legal_actions)
            for action in info_node.legal_actions:
                policy[action] = 1.0 / num_actions

        print(f"\ninfo state: {info_state_key}")
        print(f"  policy: {policy}")
        shown += 1

    print(f"\ntotal info states visited: {len(solver._infostates)}")
    print("=" * 60)
    print("mcefr test completed successfully!")


def test_mcefr_blind_action():
    """test mcefr with blind action deviations."""
    print("\n\ntesting mcefr with blind action deviations...")
    print("=" * 60)

    game = pyspiel.load_game("kuhn_poker")
    solver = MCEFRSolver(game, "blind action")

    # run a few iterations
    num_iterations = 50
    for i in range(num_iterations):
        solver.iteration()
        if (i + 1) % 10 == 0 or i == 0:
            print(f"iteration {i + 1}/{num_iterations}")

    print(f"\ninfo states visited: {len(solver._infostates)}")
    print("blind action test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_mcefr_kuhn_poker()
    test_mcefr_blind_action()
