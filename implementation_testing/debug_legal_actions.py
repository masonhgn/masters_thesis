"""
debug script to check legal actions
"""

import sys
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/open_spiel-private/build/python')
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/games')

import pyspiel
from deal_or_no_deal import DealOrNoDealGame


cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
py_game = DealOrNoDealGame({"max_num_instances": 100})

# create states after first offer
cpp_state = cpp_game.new_initial_state()
py_state = py_game.new_initial_state()

cpp_state.apply_action(0)  # instance 0
py_state.apply_action(0)

cpp_state.apply_action(0)  # p0 offers [0,0,0]
py_state.apply_action(0)

# get legal actions
cpp_legal = sorted(cpp_state.legal_actions())
py_legal = sorted(py_state.legal_actions())

print(f"cpp legal actions ({len(cpp_legal)}): {cpp_legal}")
print(f"py legal actions ({len(py_legal)}): {py_legal}")
print()

# find differences
cpp_set = set(cpp_legal)
py_set = set(py_legal)

in_cpp_not_py = cpp_set - py_set
in_py_not_cpp = py_set - cpp_set

if in_cpp_not_py:
    print(f"in cpp but not py: {sorted(in_cpp_not_py)}")
if in_py_not_cpp:
    print(f"in py but not cpp: {sorted(in_py_not_cpp)}")

# check if action 5 is legal
print()
print("instance 0 pool: [1, 2, 3]")
print(f"offer 5 is: {py_game.all_offers[5]}")
print(f"is action 5 legal in cpp? {5 in cpp_legal}")
print(f"is action 5 legal in py? {5 in py_legal}")
