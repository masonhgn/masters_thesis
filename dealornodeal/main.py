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
    """Run one self-play episode with a given policy."""
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
    print("=== CFR Exploitability Convergence ===")
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

    # --- Plot exploitability curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(eval_every, num_iterations + 1, eval_every),
        exploitabilities,
        marker="o"
    )
    plt.title("CFR Convergence: Exploitability (NashConv)")
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.yscale("log")  # log scale shows exponential decay nicely
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()



#efr

def run_efr(num_iterations: int = 1000):
    print("=== EFR Test Run ===")
    game = pyspiel.load_game("python_deal_or_no_deal")
    print("Game loaded:", game.get_type().long_name)

    solver = efr.EFRSolver(game, deviations_name="blind cf")
    print("Solver initialized.")

    nash_convs = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()
        avg_policy = solver.average_policy()

        if (i + 1) % eval_every == 0:
            nash_conv = compute_nash_conv(game, avg_policy)
            nash_convs.append(nash_conv)
            print(f"Iteration {i+1:>4}: NashConv = {nash_conv:.4f}")


    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(eval_every, num_iterations + 1, eval_every), nash_convs, marker='o')
    plt.title("EFR Convergence on Deal or No Deal")
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability (NashConv)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nTraining finished!")
    print(f"Final NashConv = {nash_convs[-1]:.4f}")
    returns = np.mean([play_once(game, solver.average_policy()) for _ in range(20)], axis=0)
    print("  Mean self-play returns:", returns)
    print("  Sum check (should be ~0):", np.sum(returns))


np.random.seed(0)

run_cfr(num_iterations=500)
# run_efr(num_iterations=1000)
