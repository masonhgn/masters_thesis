#main.py

from open_spiel.python.algorithms import efr, cfr
import deal_or_no_deal
import numpy as np
import pyspiel
import matplotlib.pyplot as plt 


def compute_exploitability(game, policy):
    """Compute NashConv (exploitability) for a Python TabularPolicy."""
    info_state_map = {}
    for info_state_str, state_idx in policy.state_lookup.items():
        action_probs = policy.action_probability_array[state_idx]
        actions = list(range(len(action_probs)))
        probs = [(a, float(p)) for a, p in zip(actions, action_probs) if p > 0]
        info_state_map[info_state_str] = probs
    return pyspiel.nash_conv(game, info_state_map)



def play_once(game, policy):
    """one one self play iteration with a specific policy"""
    state = game.new_initial_state()
    while not state.is_terminal():
        cur_player = state.current_player()

        if cur_player == pyspiel.PlayerId.CHANCE:
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
        else:
            probs_dict = policy.action_probabilities(state, cur_player)
            actions, probs = zip(*probs_dict.items())
            action = np.random.choice(actions, p=probs)

        state.apply_action(action)

    return state.returns()


#cfr

def run_cfr(num_iterations=500):
    print("RUNNING CFR")
    game = pyspiel.load_game("python_deal_or_no_deal")
    solver = cfr.CFRSolver(game)

    exploitabilities = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()

        if (i + 1) % eval_every == 0:
            avg_policy = solver.average_policy()
            expl = compute_exploitability(game, avg_policy)
            exploitabilities.append(expl)
            print(f"Iter {i+1:>4}: Exploitability = {expl:.6f}")

    #plot exploitability
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(eval_every, num_iterations + 1, eval_every),
        exploitabilities,
        marker="."
    )
    plt.title("CFR Convergence: Exploitability (NashConv)")
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f'cfr_{num_iterations}_itr_latest.png')
    plt.show()



#efr

def run_efr(num_iterations: int = 1000, deviation: str = "tips"):
    print("RUNNING EFR")
    game = pyspiel.load_game("python_deal_or_no_deal")
    solver = efr.EFRSolver(game, deviations_name=deviation)

    exploitabilities = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()

        if (i + 1) % eval_every == 0:
            avg_policy = solver.average_policy()
            expl = compute_exploitability(game, avg_policy)
            exploitabilities.append(expl)
            print(f"Iter {i+1:>4}: Exploitability = {expl:.6f}")

    #plot exploitability
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(eval_every, num_iterations + 1, eval_every),
        exploitabilities,
        marker=".",
        color = "red"
    )
    plt.title(f"EFR Convergence: Exploitability (NashConv) deviation = {deviation}")
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.3)
    plt.tight_layout()
    deviation = ''.join([c if c != ' ' else '_' for c in list(deviation)])
    plt.savefig(f'efr_{num_iterations}_itr_{deviation}_latest.png')
    plt.show()

    #confirm zero sum
    print("\nTraining finished!")
    print(f"Final Exploitability = {exploitabilities[-1]:.6f}")
    returns = np.mean([play_once(game, solver.average_policy()) for _ in range(20)], axis=0)
    print("  Mean self-play returns:", returns)
    print("  Sum check (should be around 0):", np.sum(returns))



np.random.seed(0)

run_efr(num_iterations=500, deviation='csps')
# run_efr(num_iterations=1000)
