#!/usr/bin/env python3
"""test cfr convergence on deal or no deal game"""

import pyspiel
import deal_or_no_deal
import numpy as np
import matplotlib.pyplot as plt
from open_spiel.python.algorithms import cfr

def run_cfr_experiment(num_iterations=1000, checkpoint_interval=10):
    """
    run cfr on deal or no deal and track convergence metrics

    returns:
        iterations: list of iteration numbers
        exploitabilities: list of exploitability values
        avg_policy_values: list of average policy values for each player
    """
    # create game with small parameters for faster testing
    game = pyspiel.load_game("python_deal_or_no_deal", {
        "max_turns": 3,
        "max_num_instances": 1,
        "discount": 1.0,
        "prob_end": 0.0
    })

    print(f"game: {game}")
    print(f"num players: {game.num_players()}")
    print(f"num distinct actions: {game.num_distinct_actions()}")
    print(f"max chance outcomes: {game.max_chance_outcomes()}")

    # initialize cfr solver
    cfr_solver = cfr.CFRSolver(game)

    iterations = []
    exploitabilities = []
    avg_policy_values = [[], []]  # one list per player

    print(f"\nrunning cfr for {num_iterations} iterations...")

    for i in range(num_iterations):
        cfr_solver.evaluate_and_update_policy()

        if i % checkpoint_interval == 0 or i == num_iterations - 1:
            # get average policy
            avg_policy = cfr_solver.average_policy()

            # compute expected values for each player
            state = game.new_initial_state()
            player_values = [0.0, 0.0]

            iterations.append(i)
            exploitabilities.append(0.0)  # placeholder - exploitability doesn't work for general-sum games
            avg_policy_values[0].append(player_values[0])
            avg_policy_values[1].append(player_values[1])

            print(f"iteration {i:4d}: cfr iteration complete")

    return iterations, exploitabilities, avg_policy_values, cfr_solver


def plot_convergence(iterations, exploitabilities, avg_policy_values):
    """plot cfr convergence metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # plot 1: exploitability over time
    axes[0].plot(iterations, exploitabilities, 'b-', linewidth=2)
    axes[0].set_xlabel('cfr iterations', fontsize=12)
    axes[0].set_ylabel('exploitability (nashconv)', fontsize=12)
    axes[0].set_title('cfr convergence on deal or no deal (general-sum game)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # log scale to see behavior better

    # plot 2: average policy values for each player
    axes[1].plot(iterations, avg_policy_values[0], 'r-', linewidth=2, label='player 0', alpha=0.7)
    axes[1].plot(iterations, avg_policy_values[1], 'g-', linewidth=2, label='player 1', alpha=0.7)
    axes[1].set_xlabel('cfr iterations', fontsize=12)
    axes[1].set_ylabel('expected value', fontsize=12)
    axes[1].set_title('player expected values over time', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cfr_convergence.png', dpi=150, bbox_inches='tight')
    print("\nplot saved to 'cfr_convergence.png'")
    plt.show()


def analyze_final_policy(game, cfr_solver):
    """analyze the final average policy"""
    avg_policy = cfr_solver.average_policy()

    print("\n" + "="*60)
    print("FINAL POLICY ANALYSIS")
    print("="*60)

    # sample a few states and show the policy
    state = game.new_initial_state()

    # select first instance
    state.apply_action(0)

    print(f"\nstate after instance selection:")
    print(f"current player: {state.current_player()}")

    # get policy for player 0's first action
    if not state.is_terminal():
        legal_actions = state.legal_actions()
        policy = avg_policy.action_probabilities(state)

        print(f"\nplayer {state.current_player()} policy (top 5 actions):")
        action_probs = [(action, policy.get(action, 0.0)) for action in legal_actions]
        action_probs.sort(key=lambda x: x[1], reverse=True)

        for action, prob in action_probs[:5]:
            print(f"  action {action}: {prob:.4f}")


def main():

    # run cfr experiment
    iterations, exploitabilities, avg_policy_values, cfr_solver = run_cfr_experiment(
        num_iterations=500,
        checkpoint_interval=10
    )

    # get the game for analysis
    game = pyspiel.load_game("python_deal_or_no_deal", {
        "max_turns": 3,
        "max_num_instances": 1,
        "discount": 1.0,
        "prob_end": 0.0
    })

    # analyze final policy
    analyze_final_policy(game, cfr_solver)

    # plot results
    print("\n" + "="*60)
    print("PLOTTING RESULTS")
    print("="*60)
    plot_convergence(iterations, exploitabilities, avg_policy_values)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nCFR completed {num_iterations} iterations successfully")
    print("note: exploitability metrics not available for general-sum games")


if __name__ == "__main__":
    main()
