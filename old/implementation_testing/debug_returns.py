"""
debug script to understand returns calculation differences
"""

import sys
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/open_spiel-private/build/python')
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/games')

import pyspiel
from deal_or_no_deal import DealOrNoDealGame


cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
py_game = DealOrNoDealGame({"max_num_instances": 100})

# test case: instance 0, offers [0, 5]
cpp_state = cpp_game.new_initial_state()
py_state = py_game.new_initial_state()

# select instance 0
cpp_state.apply_action(0)
py_state.apply_action(0)

# print instance details
print("instance 0 details (from cpp):")
print(cpp_state.information_state_string(0))
print()

print("instance 0 details (from python):")
py_instance = py_game.get_instance(0)
print(f"pool: {py_instance.pool}")
print(f"p0 values: {py_instance.values[0]}")
print(f"p1 values: {py_instance.values[1]}")
print()

# player 0 makes offer 0 (which is [0,0,0])
print("p0 makes offer 0 ([0,0,0])")
cpp_state.apply_action(0)
py_state.apply_action(0)
print(f"cpp current player: {cpp_state.current_player()}")
print(f"py current player: {py_state.current_player()}")
print()

# player 1 makes offer 5 (which is [0,0,5])
print("p1 makes offer 5 ([0,0,5])")
print(f"offer 5 in python: {py_game.all_offers[5]}")
cpp_state.apply_action(5)
py_state.apply_action(5)
print(f"cpp current player: {cpp_state.current_player()}")
print(f"py current player: {py_state.current_player()}")
print()

# player 0 accepts
accept_action = cpp_game.num_distinct_actions() - 1
print(f"p0 accepts (action {accept_action})")
cpp_state.apply_action(accept_action)
py_state.apply_action(accept_action)

print(f"cpp is_terminal: {cpp_state.is_terminal()}")
print(f"py is_terminal: {py_state.is_terminal()}")
print()

# compare returns
cpp_returns = cpp_state.returns()
py_returns = py_state.returns()

print(f"cpp returns: {cpp_returns}")
print(f"py returns: {py_returns}")
print()

# manually calculate expected returns
print("manual calculation:")
print(f"last offer was [0,0,5] made by player 1")
print(f"player 0 accepted, so player 1 gets [0,0,5], player 0 gets remaining")
print(f"pool: {py_instance.pool}")
print(f"player 1 gets: [0,0,5], utility = 0*{py_instance.values[1][0]} + 0*{py_instance.values[1][1]} + 5*{py_instance.values[1][2]}")
print(f"player 0 gets: [{py_instance.pool[0]},_{py_instance.pool[1]},{py_instance.pool[2]-5}], utility = {py_instance.pool[0]}*{py_instance.values[0][0]} + {py_instance.pool[1]}*{py_instance.values[0][1]} + {py_instance.pool[2]-5}*{py_instance.values[0][2]}")
