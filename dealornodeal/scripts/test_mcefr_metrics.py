

import sys
import os


import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from dealornodeal.algos import mcefr
from open_spiel.python import policy as policy_module
import dealornodeal.games.deal_or_no_deal as deal_or_no_deal
import random


def compute_policy_delta(game, solver, prev_values):
    """
    compute measure of strategy change using expected value changes.

    measures how much the policy's performance is changing.
    should decrease if converging to stable policy.

    args:
        game: the game instance
        solver: the solver instance
        prev_values: tuple of (ev_p0, ev_p1) from previous iteration

    returns:
        delta: l2 norm of expected value changes
        current_values: current (ev_p0, ev_p1) for next iteration
    """
    # evaluate current policy
    avg_policy = solver.average_policy()
    ev_p0, ev_p1 = evaluate_policy_values(game, avg_policy, num_samples=30)
    current_values = (ev_p0, ev_p1)

    if prev_values is None:
        # first iteration
        return 0.0, current_values

    # compute l2 norm of value changes
    delta = np.sqrt((ev_p0 - prev_values[0])**2 + (ev_p1 - prev_values[1])**2)

    return delta, current_values


def evaluate_policy_values(game, policy, num_samples=100):
    """
    evaluate expected utility for each player under the given policy.
    simulates games using the policy and computes average returns.

    args:
        game: the game instance
        policy: the policy to evaluate
        num_samples: number of game samples to average over

    returns:
        (expected_value_p0, expected_value_p1): tuple of utilities across num_samples games
    """
    total_returns = [0.0, 0.0]

    for _ in range(num_samples):
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.is_chance_node():
                # sample from chance distribution
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = np.random.choice(actions, p=probs)
            else:
                # sample from policy
                legal_actions = state.legal_actions()
                action_probs = policy.action_probabilities(state)

                # build probability distribution over legal actions
                probs = [action_probs.get(a, 0.0) for a in legal_actions]
                prob_sum = sum(probs)

                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                    action = np.random.choice(legal_actions, p=probs)
                else:
                    # uniform random if no policy defined
                    action = np.random.choice(legal_actions)

            state.apply_action(action)

        returns = state.returns()
        total_returns[0] += returns[0]
        total_returns[1] += returns[1]

    return total_returns[0] / num_samples, total_returns[1] / num_samples


def compute_policy_variance(metrics_history, n=5):
    """
    compute variance in recent expected values.
    low variance indicates stable/converged policy.

    args:
        metrics_history: dictionary containing metric histories

    returns:
        variance_p0: variance in player 0's recent expected values
        variance_p1: variance in player 1's recent expected values
    """
    if len(metrics_history['expected_value_p0']) < n:
        return 0.0, 0.0

    # compute variance over last 5 measurements
    recent_p0 = metrics_history['expected_value_p0'][-n:]
    recent_p1 = metrics_history['expected_value_p1'][-n:]

    variance_p0 = np.var(recent_p0)
    variance_p1 = np.var(recent_p1)

    return variance_p0, variance_p1


def compute_cumulative_regret_mcefr(solver, iteration):
    """
    compute aggregate average external regret for mcefr.

    for each info state, takes the maximum positive deviation regret.
    this measures how much each player could have improved.

    args:
        solver: the mcefr solver instance
        iteration: current iteration number (for normalization)

    returns:
        avg_external_regret: normalized aggregate regret
    """
    nodes = solver._infostates
    raw_sum = 0.0

    for key, info_state_node in nodes.items():
        # cumulative_regret is an array of regrets per deviation
        max_reg = np.maximum(info_state_node.cumulative_regret, 0.0)
        raw_sum += float(np.max(max_reg))

    # normalize by iteration count
    return raw_sum / max(iteration, 1)


def run_mcefr_with_metrics(game, deviation_type, num_iterations=500, checkpoint_interval=10, num_policy_samples=50):
    """
    run mcefr and track convergence metrics.

    args:
        game: the game to solve
        deviation_type: type of deviations to use
        num_iterations: number of mcefr iterations
        checkpoint_interval: how often to compute metrics
        num_policy_samples: number of samples for policy evaluation

    returns:
        metrics: dictionary containing all tracked metrics
        mcefr_solver: the trained mcefr solver
    """
    mcefr_solver = mcefr.MCEFRSolver(game, deviation_name=deviation_type)

    # initialize metric tracking
    metrics = {
        'iterations': [],
        'policy_delta': [],
        'expected_value_p0': [],
        'expected_value_p1': [],
        'social_welfare': [],
        'variance_p0': [],
        'variance_p1': [],
        'cumulative_regret': []
    }

    prev_values = None

    print(f"\nrunning mcefr for {num_iterations} iterations...")
    print(f"computing metrics every {checkpoint_interval} iterations")
    print("=" * 60)

    import time

    print(f"\nStarting MCEFR at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    for i in range(num_iterations):
        mcefr_solver.iteration()

        if i % checkpoint_interval == 0 or i == num_iterations - 1:
            # compute all metrics
            delta, prev_values = compute_policy_delta(game, mcefr_solver, prev_values)
            ev_p0, ev_p1 = prev_values  # reuse from delta computation
            social_welfare = ev_p0 + ev_p1

            # store metrics
            metrics['iterations'].append(i)
            metrics['policy_delta'].append(delta)
            metrics['expected_value_p0'].append(ev_p0)
            metrics['expected_value_p1'].append(ev_p1)
            metrics['social_welfare'].append(social_welfare)

            # compute variance (needs history)
            var_p0, var_p1 = compute_policy_variance(metrics)
            metrics['variance_p0'].append(var_p0)
            metrics['variance_p1'].append(var_p1)

            # compute cumulative regret
            cum_regret = compute_cumulative_regret_mcefr(mcefr_solver, i + 1)
            metrics['cumulative_regret'].append(cum_regret)

            # print progress
            print(f"iteration {i:4d}: "
                  f"delta={delta:.6f}, "
                  f"ev_p0={ev_p0:.3f}, "
                  f"ev_p1={ev_p1:.3f}, "
                  f"welfare={social_welfare:.3f}, "
                  f"var={np.sqrt(var_p0 + var_p1):.4f}, "
                  f"regret={cum_regret:.1f}")

    total_time = time.time() - start_time
    metrics['total_time'] = total_time

    print("=" * 60)

    return metrics, mcefr_solver


def plot_metrics(metrics, algo_name, output_dir):
    """
    plot all convergence metrics as separate image files with lines and dots.

    args:
        metrics: dictionary of metrics
        output_dir: directory to save plots
    """

    os.makedirs(output_dir, exist_ok=True)
    iterations = metrics['iterations']
    saved_files = []

    # plot 1: policy delta
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics['policy_delta'], 'b-o', linewidth=1,
             markersize=2, markerfacecolor='blue', markeredgecolor='darkblue')
    plt.xlabel(f'{algo_name} iteration', fontsize=12)
    plt.ylabel('policy delta (l2 norm)', fontsize=12)
    plt.title('strategy convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    if max(metrics['policy_delta']) > 0:
        plt.yscale('log')
    filepath = f"{output_dir}/policy_delta.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)

    # plot 2: expected values
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics['expected_value_p0'], 'r-o',
             linewidth=1, label='player 0', alpha=0.8, markersize=2)
    plt.plot(iterations, metrics['expected_value_p1'], 'g-s',
             linewidth=1, label='player 1', alpha=0.8, markersize=2)
    plt.xlabel(f'{algo_name} iteration', fontsize=12)
    plt.ylabel('expected utility', fontsize=12)
    plt.title('player expected values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    filepath = f"{output_dir}/expected_values.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)

    # plot 3: social welfare
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics['social_welfare'], 'o-', color='purple',
             linewidth=1, markersize=2, markerfacecolor='purple',
             markeredgecolor='darkviolet')
    plt.xlabel(f'{algo_name} iteration', fontsize=12)
    plt.ylabel('total utility (p0 + p1)', fontsize=12)
    plt.title('social welfare', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    filepath = f"{output_dir}/social_welfare.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)

    # plot 4: policy variance
    total_variance = [np.sqrt(v0 + v1) for v0, v1 in
                      zip(metrics['variance_p0'], metrics['variance_p1'])]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, total_variance, 'o-', color='orange',
             linewidth=1, markersize=2, markerfacecolor='orange',
             markeredgecolor='darkorange')
    plt.xlabel(f'{algo_name} iteration', fontsize=12)
    plt.ylabel('policy variance (std dev)', fontsize=12)
    plt.title('policy stability', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    filepath = f"{output_dir}/policy_variance.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)

    # plot 5: cumulative regret
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, metrics['cumulative_regret'], 'o-', color='brown',
             linewidth=1, markersize=2, markerfacecolor='brown',
             markeredgecolor='saddlebrown')
    plt.xlabel(f'{algo_name} iteration', fontsize=12)
    plt.ylabel('total absolute regret', fontsize=12)
    plt.title('cumulative regret', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    if max(metrics['cumulative_regret']) > 0:
        plt.yscale('log')
    filepath = f"{output_dir}/cumulative_regret.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)

    print(f"\nsaved {len(saved_files)} plots:")
    for f in saved_files:
        print(f"  - {f}")


def save_experiment(save_params, base_dir='output'):
    """
    Save experiment results to a new directory with JSON and CSV files.

    Args:
        save_params: dictionary containing all experiment parameters
        base_dir: base directory for all experiments

    Returns:
        experiment_dir: path to the created experiment directory
    """
    import datetime
    import os

    # Create unique experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # replace spaces with underscores in algo name for clean filenames
    algo_name_clean = save_params['algo_metadata']['algo_name'].replace(' ', '_')
    experiment_dir = os.path.join(base_dir, f"{algo_name_clean}_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save JSON
    json_path = save_to_json(save_params, output_dir=experiment_dir)

    # Plot metrics
    plot_metrics(save_params['metrics'], save_params['algo_metadata']['algo_longname'], experiment_dir)

    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  - JSON: {os.path.basename(json_path)}")

    return experiment_dir


def save_to_json(save_params, output_dir):
    """
    save complete experiment to json file with all metrics so that we can replot them how we want

    Args:
        save_params: dictionary containing all experiment parameters
        output_dir: directory to save the file

    Returns:
        filepath: path to the saved JSON file
    """
    import json
    import datetime

    filepath = os.path.join(output_dir, 'experiment_data.json')

    num_iterations = save_params['num_iterations']
    metrics = save_params['metrics']

    experiment_data = {
        'game_metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'game': 'Deal or No Deal'
        },
        'algo_metadata': save_params['algo_metadata'],
        'game_parameters': save_params['game_params'],
        'timing': {
            'total_iterations': num_iterations,
            'wall_clock_seconds': metrics['total_time'],
            'seconds_per_iteration': metrics['total_time'] / num_iterations,
            'iterations_per_second': num_iterations / metrics['total_time']
        },
        'final_metrics': {
            'policy_delta': float(metrics['policy_delta'][-1]),
            'ev_player0': float(metrics['expected_value_p0'][-1]),
            'ev_player1': float(metrics['expected_value_p1'][-1]),
            'social_welfare': float(metrics['social_welfare'][-1]),
            'cumulative_regret': float(metrics['cumulative_regret'][-1]),
            'total_variance': float(np.sqrt(metrics['variance_p0'][-1] + metrics['variance_p1'][-1]))
        },
        'convergence_assessment': {
            'strategy_convergence': 'STRONG' if metrics['policy_delta'][-1] < 0.1
                                   else 'MODERATE' if metrics['policy_delta'][-1] < 0.5
                                   else 'WEAK',
            'policy_stability': 'STABLE' if np.sqrt(metrics['variance_p0'][-1] + metrics['variance_p1'][-1]) < 0.5
                               else 'MODERATE' if np.sqrt(metrics['variance_p0'][-1] + metrics['variance_p1'][-1]) < 1.0
                               else 'UNSTABLE'
        },
        'iteration_history': {
            'iterations': metrics['iterations'],
            'policy_delta': [float(x) for x in metrics['policy_delta']],
            'ev_player0': [float(x) for x in metrics['expected_value_p0']],
            'ev_player1': [float(x) for x in metrics['expected_value_p1']],
            'social_welfare': [float(x) for x in metrics['social_welfare']],
            'variance_p0': [float(x) for x in metrics['variance_p0']],
            'variance_p1': [float(x) for x in metrics['variance_p1']],
            'cumulative_regret': [float(x) for x in metrics['cumulative_regret']]
        }
    }

    # write json
    with open(filepath, 'w') as f:
        json.dump(experiment_data, f, indent=2)

    return filepath


def run_mcefr(params):
    """run mcefr with comprehensive metrics on deal or no deal"""

    random.seed(params['random_seed'])

    game_params = {
        "max_turns": params['max_turns'],
        "max_num_instances": params['max_num_instances'],
        "discount": params['discount'],
        "prob_end": params['prob_end']
    }

    # create game with small parameters for testing
    game = pyspiel.load_game("python_deal_or_no_deal", game_params)

    print("MCEFR CONVERGENCE METRICS - DEAL OR NO DEAL")

    print(f"game: {game}")
    print(f"num players: {game.num_players()}")
    print(f"num distinct actions: {game.num_distinct_actions()}")
    print(f"max chance outcomes: {game.max_chance_outcomes()}")

    metrics, mcefr_solver = run_mcefr_with_metrics(
        game=game,
        deviation_type=params['deviation_type'],
        num_iterations=params['num_iterations'],
        checkpoint_interval=params['checkpoint_interval'],
        num_policy_samples=params['num_ev_samples']
    )

    save_params = {
        'algo_metadata': {
            'algo_name': f"mcefr_{params['deviation_type']}",
            'algo_longname': f"MCEFR ({params['deviation_type']})"
        },
        'game_params': game_params,
        'num_iterations': params['num_iterations'],
        'metrics': metrics,
    }

    experiment_dir = save_experiment(save_params=save_params)

    print("COMPLETE!")


if __name__ == "__main__":

    run_mcefr(
        params={
        'num_iterations': 10000,
        'deviation_type': 'csps',
        'checkpoint_interval': 100,
        'num_ev_samples': 10,
        "max_turns": 2,
        "max_num_instances": 3,
        "discount": 1.0,
        "prob_end": 0.0,
        "random_seed": 42,
        }
    )
