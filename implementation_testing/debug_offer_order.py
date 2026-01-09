"""
debug script to check offer ordering
"""

import sys
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/open_spiel-private/build/python')
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/games')

import pyspiel
from deal_or_no_deal import DealOrNoDealGame


cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
py_game = DealOrNoDealGame({"max_num_instances": 100})

# print first 20 offers from each game
print("first 20 offers:")
print("action | cpp offer | py offer")
print("-" * 40)

cpp_state = cpp_game.new_initial_state()
for i in range(min(20, 120)):
    cpp_offer_str = cpp_game.action_to_string(0, i)
    py_offer = py_game.all_offers[i]
    print(f"{i:6d} | {cpp_offer_str:25s} | {py_offer}")

print()
print("checking specific actions that differ:")
print()

# check actions that are in cpp but not py
in_cpp_not_py = [64, 65, 70, 71]
for action in in_cpp_not_py[:4]:
    cpp_offer_str = cpp_game.action_to_string(0, action)
    py_offer = py_game.all_offers[action]
    print(f"action {action}:")
    print(f"  cpp: {cpp_offer_str}")
    print(f"  py: {py_offer}")
    print()
