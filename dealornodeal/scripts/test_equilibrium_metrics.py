import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from open_spiel.python.algorithms import cfr, efr
from open_spiel.python import policy
import os
import datetime


def run_algorithm(game, algo_name, num_iterations=200, checkpoint_interval=20):
    """run cfr or efr and track equilibrium distances."""

    # create solver
    if algo_name == 'cfr':
        solver = cfr.CFRPlusSolver(game)
    else:
        solver = efr.EFRSolver(game, deviations_name=algo_name)

    metrics = {
        'iterations': [],
        'cce_dist': [],
        'ce_dist': [],
    }

    strategies = []

    print(f"\nrunning {algo_name}...")

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()

        if i % checkpoint_interval == 0 or i == num_iterations - 1:
            # convert policy and add to strategies
            current_policy = policy.python_policy_to_pyspiel_policy(solver.current_policy())
            strategies.append(current_policy)

            # compute equilibrium distances
            corr_dev = pyspiel.uniform_correlation_device(strategies)
            cce_dist_info = pyspiel.cce_dist(game, corr_dev)
            det_corr_dev = pyspiel.sampled_determinize_corr_dev(corr_dev, 100)
            ce_dist_info = pyspiel.ce_dist(game, det_corr_dev)

            metrics['iterations'].append(i)
            metrics['cce_dist'].append(cce_dist_info.dist_value)
            metrics['ce_dist'].append(ce_dist_info.dist_value)

            print(f"  iter {i:4d}: cce={cce_dist_info.dist_value:.6f}, ce={ce_dist_info.dist_value:.6f}")

    return metrics


def plot_comparison(cfr_metrics, tips_metrics, output_dir):
    """plot cfr vs tips comparison."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # cce distance
    ax1.plot(cfr_metrics['iterations'], cfr_metrics['cce_dist'],
             'b-o', linewidth=2, label='cfr', markersize=4)
    ax1.plot(tips_metrics['iterations'], tips_metrics['cce_dist'],
             'r-s', linewidth=2, label='tips', markersize=4)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('cce distance')
    ax1.set_title('coarse correlated equilibrium distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # ce distance
    ax2.plot(cfr_metrics['iterations'], cfr_metrics['ce_dist'],
             'b-o', linewidth=2, label='cfr', markersize=4)
    ax2.plot(tips_metrics['iterations'], tips_metrics['ce_dist'],
             'r-s', linewidth=2, label='tips', markersize=4)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('ce distance')
    ax2.set_title('correlated equilibrium distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nsaved plot to: {filepath}")


if __name__ == "__main__":
    # load bargaining game (c++ deal-or-no-deal)
    game = pyspiel.load_game("bargaining")
    print(f"game: {game}")
    print(f"num players: {game.num_players()}")

    # run both algorithms
    cfr_metrics = run_algorithm(game, 'cfr', num_iterations=200, checkpoint_interval=20)
    tips_metrics = run_algorithm(game, 'tips', num_iterations=200, checkpoint_interval=20)

    # plot results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/bargaining_{timestamp}"
    plot_comparison(cfr_metrics, tips_metrics, output_dir)

    print("\ndone!")
