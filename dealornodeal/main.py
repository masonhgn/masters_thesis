from open_spiel.python.algorithms import efr
#from deal_or_no_deal import DealOrNoDealGame
import deal_or_no_deal  
import numpy as np
import pyspiel



def play_once(game, policy):
    """Run one self-play episode with a given policy."""
    state = game.new_initial_state()
    while not state.is_terminal():
        cur_player = state.current_player()

        if cur_player == pyspiel.PlayerId.CHANCE:
            # Sample from chance distribution
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
        else:
            # Query policy for decision player
            probs_dict = policy.action_probabilities(state, cur_player)
            actions, probs = zip(*probs_dict.items())
            action = np.random.choice(actions, p=probs)

        state.apply_action(action)

    return state.returns()




def run_efr():
    game = pyspiel.load_game("python_deal_or_no_deal")
    
    # Pick a deviation type
    solver = efr.EFRSolver(game, deviations_name="blind cf")  
    # "blind cf" = vanilla CFR behavior
    # Try "informed action" or "cfps" for richer deviation sets

    NUM_ITERATIONS = 1000
    for i in range(NUM_ITERATIONS):
        solver.evaluate_and_update_policy()
        avg_policy = solver.average_policy()

        # Sample some self-play returns
        returns = [play_once(game, avg_policy) for _ in range(10)]
        mean_returns = np.mean(returns, axis=0)

        print(f"Iteration {i+1}")
        print("  Mean self-play returns:", mean_returns)


run_efr()
