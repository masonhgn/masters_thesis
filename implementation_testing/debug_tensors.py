"""
debug script to check tensor encoding
"""

import sys
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/open_spiel-private/build/python')
sys.path.append('/Users/fluffy/projects/brown/thesis/masters_thesis/dealornodeal/games')

import pyspiel
import numpy as np
from deal_or_no_deal import DealOrNoDealGame


cpp_game = pyspiel.load_game("bargaining", {"max_num_instances": 100})
py_game = DealOrNoDealGame({"max_num_instances": 100})

# create a simple game state
cpp_state = cpp_game.new_initial_state()
py_state = py_game.new_initial_state()

cpp_state.apply_action(0)  # instance 0
py_state.apply_action(0)

cpp_state.apply_action(0)  # p0 offers [0,0,0]
py_state.apply_action(0)

cpp_state.apply_action(5)  # p1 offers [5,0,0]
py_state.apply_action(5)

# get information state tensors for player 0
cpp_tensor = np.array(cpp_state.information_state_tensor(0))
py_tensor = py_state.information_state_tensor(0)

print("information state tensor comparison:")
print(f"cpp shape: {cpp_tensor.shape}, py shape: {py_tensor.shape}")
print()

# print first 50 values side by side
print("index | cpp | py | match")
print("-" * 40)
for i in range(min(50, len(cpp_tensor))):
    match = "YES" if np.isclose(cpp_tensor[i], py_tensor[i]) else "NO"
    if not np.isclose(cpp_tensor[i], py_tensor[i]):
        print(f"{i:5d} | {cpp_tensor[i]:3.1f} | {py_tensor[i]:3.1f} | {match}")

print()
print("understanding tensor structure:")
print("1. agreement reached (1 bit)")
print("2. number of offers (max_turns + 1 bits)")
print(f"   - we have {len(py_state._offers)} offers")
print("3. pool (3 * (pool_max + 1) bits with thermometer encoding)")
print(f"   - pool is {py_state._instance.pool}")
print("4. my values (3 * (total_value + 1) bits with thermometer encoding)")
print(f"   - player 0 values: {py_state._instance.values[0]}")
print("5. all offers (max_turns * 3 * (pool_max + 1) bits)")
print(f"   - offers so far: {py_state._offers}")

print()
print("checking observation tensor:")
cpp_obs = np.array(cpp_state.observation_tensor(0))
py_obs = py_state.observation_tensor(0)

print(f"cpp shape: {cpp_obs.shape}, py shape: {py_obs.shape}")
print()

print("index | cpp | py | match")
print("-" * 40)
for i in range(min(30, len(cpp_obs))):
    match = "YES" if np.isclose(cpp_obs[i], py_obs[i]) else "NO"
    if not np.isclose(cpp_obs[i], py_obs[i]):
        print(f"{i:5d} | {cpp_obs[i]:3.1f} | {py_obs[i]:3.1f} | {match}")
