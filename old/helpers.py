#main.py

from open_spiel.python.algorithms import efr, cfr
import games.deal_or_no_deal as deal_or_no_deal, old.deal_or_no_deal_zerosum as deal_or_no_deal_zerosum
import old.deal_or_no_deal_mini as deal_or_no_deal_mini, old.deal_or_no_deal_mini_zerosum as deal_or_no_deal_mini_zerosum
import numpy as np
import pyspiel
import matplotlib.pyplot as plt 


def compute_exploitability(game, policy):
    """Compute NashConv for a TabularPolicy."""
    info_state_map = {}
    for info_state_str, state_idx in policy.state_lookup.items():
        action_probs = policy.action_probability_array[state_idx]
        actions = list(range(len(action_probs)))
        probs = [(a, float(p)) for a, p in zip(actions, action_probs) if p > 0]
        info_state_map[info_state_str] = probs
    return pyspiel.nash_conv(game, info_state_map)




def average_positive_regret(solver):
    """Compute average positive regret across all information states."""
    if hasattr(solver, "_info_state_nodes"):
        nodes = solver._info_state_nodes.values()
    elif hasattr(solver, "_info_states"):
        nodes = solver._info_states.values()
    else:
        raise AttributeError("Solver has no info state registry.")

    total_regret = 0.0
    num_infosets = 0

    for info_state in nodes:
        if hasattr(info_state, "cumulative_regrets"):
            regrets = np.array(info_state.cumulative_regrets)
        elif hasattr(info_state, "cumulative_regret"):
            regrets = np.array(list(info_state.cumulative_regret.values()))
        else:
            continue

        total_regret += np.sum(np.maximum(regrets, 0))
        num_infosets += 1

    return float(total_regret / max(num_infosets, 1))



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


#cfr with exploitability metric 

def run_cfr(game_name: str, num_iterations=500):
    print("RUNNING CFR")
    game = pyspiel.load_game(game_name)
    solver = cfr.CFRSolver(game)

    exploitabilities = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()

        if (i + 1) % eval_every == 0:
            avg_policy = solver.average_policy()
            expl = compute_exploitability(game, avg_policy)
            exploitabilities.append(expl)
            print(f"Iter {i+1:>4}: Exploitability = {expl}")

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
    plt.savefig(f'output/{game_name}_cfr.png')
    plt.show()












#efr with exploitability metric

def run_efr(game_name: str, num_iterations: int = 500, deviation: str = "tips"):
    print("RUNNING EFR")
    game = pyspiel.load_game(game_name)
    solver = efr.EFRSolver(game, deviations_name=deviation)

    exploitabilities = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()

        if (i + 1) % eval_every == 0:
            avg_policy = solver.average_policy()
            expl = compute_exploitability(game, avg_policy)
            exploitabilities.append(expl)
            print(f"Iter {i+1:>4}: Exploitability = {expl}")

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
    plt.savefig(f'output/{game_name}_efr_{deviation}.png')
    plt.show()

    #confirm zero sum
    print("\nTraining finished!")
    print(f"Final Exploitability = {exploitabilities[-1]}")
    returns = np.mean([play_once(game, solver.average_policy()) for _ in range(20)], axis=0)
    print("  Mean self-play returns:", returns)
    print("  Sum check (should be around 0):", np.sum(returns))












def run_cfr_regret(game_name: str, num_iterations=1000):
    print("RUNNING CFR (tracking regret)")
    game = pyspiel.load_game(game_name)
    solver = cfr.CFRSolver(game)

    avg_regrets = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()
        if (i + 1) % eval_every == 0:
            # get total regret from all infosets
            total_regret = 0.0
            num_infosets = 0
            for info_state in solver._info_state_nodes.values():
                regrets = info_state.cumulative_regret.values()
                total_regret += sum(max(r, 0) for r in regrets)

                num_infosets += 1

            avg_regret = total_regret / max(num_infosets, 1)
            avg_regrets.append(avg_regret)
            print(f"Iter {i+1:>4}: Avg regret = {avg_regret}")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(eval_every, num_iterations + 1, eval_every), avg_regrets, marker='.')
    plt.title("CFR Convergence: Average Regret")
    plt.xlabel("Iteration")
    plt.ylabel("Average Regret")
    plt.yscale("log")
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f"output/{game_name}_cfr.png")
    plt.show()

    return solver


def run_efr_regret(game_name: str, num_iterations=1000, deviation="csps"):
    print("RUNNING EFR (tracking regret)")
    game = pyspiel.load_game(game_name)
    solver = efr.EFRSolver(game, deviations_name=deviation)

    avg_regrets = []
    eval_every = 10

    for i in range(num_iterations):
        solver.evaluate_and_update_policy()
        if (i + 1) % eval_every == 0:
            total_regret = 0.0
            num_infosets = 0
            for info_state in solver._info_state_nodes.values():
                regrets = info_state.cumulative_regret.values()
                total_regret += sum(max(r, 0) for r in regrets)

                num_infosets += 1

            avg_regret = total_regret / max(num_infosets, 1)
            avg_regrets.append(avg_regret)
            print(f"Iter {i+1:>4}: Avg regret = {avg_regret}")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(eval_every, num_iterations + 1, eval_every),
             avg_regrets, marker='.', color='red')
    plt.title(f"EFR Convergence: Average Regret ({deviation})")
    plt.xlabel("Iteration")
    plt.ylabel("Average Regret")
    plt.yscale("log")
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f'output/{game_name}_efr_{deviation}.png')
    plt.show()

    return solver










from open_spiel.python.algorithms import outcome_sampling_mccfr as mccfr

def run_mccfr(game_name: str, num_iterations=1000):
    print("RUNNING MCCFR")
    game = pyspiel.load_game(game_name)
    solver = mccfr.OutcomeSamplingSolver(game)
    
    regrets = []
    eval_every = 100
    
    for i in range(num_iterations):
        solver.iteration()
        if (i + 1) % eval_every == 0:
            avg_policy = solver.average_policy()
            total_regret = solver.average_regret()
            regrets.append(total_regret)
            print(f"Iter {i+1}: Avg Regret = {total_regret}")

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(eval_every, num_iterations + 1, eval_every), regrets, marker='.')
    plt.title("MCCFR (Outcome Sampling) Average Regret")
    plt.xlabel("Iteration")
    plt.ylabel("Average Regret")
    plt.yscale("log")
    plt.grid(True, linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f'output/{game_name}_mccfr_.png')
    plt.show()

    return solver






def print_policy_summary(game, policy, num_samples=100):
    print("\n=== Policy Summary ===")
    infos = list(policy.state_lookup.keys())
    np.random.shuffle(infos)
    for info_state in infos[:num_samples]:
        actions = list(range(policy.action_probability_array.shape[1]))
        probs = policy.action_probability_array[policy.state_lookup[info_state]]
        top_actions = [(a, p) for a, p in zip(actions, probs) if p > 0.01]
        print(f"{info_state}\n  " + ", ".join([f"a{a}: {p:.2f}" for a, p in top_actions]))
    print("======================\n")








if __name__ == "__main__":

    np.random.seed(42)

    # #zero sum mini cfr
    # print('running zero sum mini cfr')
    # run_cfr(game_name='python_deal_or_no_deal_mini_zerosum')




    # #zero sum mini efr tips
    # print('running zero sum mini efr tips')
    # run_efr(game_name='python_deal_or_no_deal_mini_zerosum', deviation='tips')




    # #zero sum mini efr causal
    # print('running zero sum mini efr causal')
    # run_efr(game_name='python_deal_or_no_deal_mini_zerosum', deviation='csps')




    #original mini cfr
    print('running original mini cfr')
    run_cfr(game_name='python_deal_or_no_deal_mini')




    # #original mini efr tips
    # print('running original mini efr tips')
    # run_efr_regret(game_name='python_deal_or_no_deal_mini', deviation='tips')



    # #original mini efr causal
    # print('running original mini efr causal')
    # run_efr_regret(game_name='python_deal_or_no_deal_mini', deviation='csps')




    # #original full mccfr
    # print('running original full mccfr')
    # run_mccfr(game_name='python_deal_or_no_deal')





    #original full mcefr tips
    print('running original full mcefr tips')
    print('mcefr not ready')




    #original full mcefr causal
    print('running original full mcefr causal')
    print('mcefr not ready')