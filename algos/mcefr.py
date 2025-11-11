# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified: 2023 James Flynn
# Original based on:
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/outcome_sampling_mccfr.py

"""Python implementation of Monte Carlo Extensive-Form Regret Minimization (MCEFR).

This combines the Monte Carlo sampling approach from MCCFR with the deviation-based
regret framework from EFR. Instead of traversing the entire game tree, MCEFR samples
trajectories through the tree while maintaining EFR's deviation-based regret matching.

See:
- "Efficient Deviation Types and Learning for Hindsight Rationality in Extensive-Form Games",
  Morrill et al. 2021b, https://arxiv.org/abs/2102.06973
- "Monte Carlo Sampling for Regret Minimization in Extensive Games",
  Lanctot et al. 2009

One iteration of MCEFR consists of:
1) Sample a trajectory through the game tree (outcome sampling)
2) Compute deviation values along the sampled trajectory
3) Update regrets for deviations at sampled information states
4) Update current strategy using regret matching

The average policy converges to an equilibrium (type depends on deviation type used).
"""

import collections
import copy
import numpy as np
from scipy import linalg

from open_spiel.python import policy
import pyspiel

# Import deviation types and helper functions from efr.py
from algos.efr import (
    LocalDeviationWithTimeSelection,
    _InfoStateNode,
    _update_average_policy,
    strat_dict_to_array,
    array_to_strat_dict,
    create_probs_from_index,
    return_blind_action,
    return_informed_action,
    return_blind_cf,
    return_informed_cf,
    return_blind_partial_sequence,
    return_cf_partial_sequence,
    return_cs_partial_sequence,
    return_twice_informed_partial_sequence,
    return_behavourial,
)


class MCEFRSolver(object):
  """Monte Carlo Extensive-Form Regret Minimization solver.

  This solver uses outcome sampling to estimate regrets for deviations
  rather than computing exact values over the entire game tree.

  Usage:
  ```python
    game = pyspiel.load_game("game_name")
    solver = MCEFRSolver(game, deviations_name="blind cf")
    for i in range(num_iterations):
      solver.iteration()
      solver.average_policy()  # Access the average policy
  ```
  """

  def __init__(self, game, deviations_name):
    """Initializer.

    Args:
      game: The `pyspiel.Game` to run on.
      deviations_name: the name of the deviation type to use for
        accumulating regrets and calculating the strategy at the next timestep.

    Deviation types implemented are "blind action", "informed action",
    "blind cf", "informed counterfactual", "blind partial sequence",
    "counterfactual partial sequence", "casual partial sequence",
    "twice informed partial sequence", "single target behavioural".

    See "Efficient Deviation Types and Learning for Hindsight Rationality in
    Extensive-Form Games" by D. Morrill et al. 2021b
    for the full definition of each type.
    """
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "MCEFR requires sequential games. If you're trying to run it "
        + "on a simultaneous (or normal-form) game, please first transform it "
        + "using turn_based_simultaneous_game."
    )

    self._game = game
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()

    # This is for returning the current policy and average policy to a caller
    self._current_policy = policy.TabularPolicy(game)
    self._average_policy = self._current_policy.__copy__()

    # Set up deviation generator
    self._external_only = False
    deviation_sets = None

    if deviations_name in {"blind action"}:
      deviation_sets = return_blind_action
      self._external_only = True
    elif deviations_name in {"informed action"}:
      deviation_sets = return_informed_action
    elif deviations_name in {"blind cf", "blind counterfactual"}:
      deviation_sets = return_blind_cf
      self._external_only = True
    elif deviations_name in {"informed cf", "informed counterfactual"}:
      deviation_sets = return_informed_cf
    elif deviations_name in {"bps", "blind partial sequence"}:
      deviation_sets = return_blind_partial_sequence
      self._external_only = True
    elif deviations_name in {
        "cfps",
        "cf partial sequence",
        "counterfactual partial sequence",
    }:
      deviation_sets = return_cf_partial_sequence
    elif deviations_name in {"csps", "casual partial sequence"}:
      deviation_sets = return_cs_partial_sequence
    elif deviations_name in {"tips", "twice informed partial sequence"}:
      deviation_sets = return_twice_informed_partial_sequence
    elif deviations_name in {"bhv", "single target behavioural", "behavioural"}:
      deviation_sets = return_behavourial
    else:
      raise ValueError(
          "Unsupported Deviation Set Passed As Constructor Argument"
      )

    self._deviation_gen = deviation_sets

    # Store information state nodes (created lazily during sampling)
    self._info_state_nodes = {}

    # Initialize root structure for generating deviations
    hist = {player: [] for player in range(self._num_players)}
    empty_path_indices = [[] for _ in range(self._num_players)]
    self._initialize_info_state_nodes(self._root_node, hist, empty_path_indices)

    self._iteration = 1  # For possible linear-averaging

  def _initialize_info_state_nodes(self, state, history, path_indices):
    """Initializes info_state_nodes.

    Create one _InfoStateNode per infoset. This is similar to EFR's initialization
    but adapted for lazy creation during Monte Carlo sampling.

    Args:
      state: The current state in the tree traversal.
      history: an array of the preceding actions taken prior to the state
        for each player.
      path_indices: a 3d array [player number]x[preceding state]x[legal actions
        for state, index of the policy for this state in TabularPolicy].
    """
    if state.is_terminal():
      return

    if state.is_chance_node():
      for action, unused_action_prob in state.chance_outcomes():
        self._initialize_info_state_nodes(
            state.child(action), history, path_indices
        )
      return

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)
    info_state_node = self._info_state_nodes.get(info_state)
    if info_state_node is None:
      legal_actions = state.legal_actions(current_player)
      info_state_node = _InfoStateNode(
          legal_actions=legal_actions,
          index_in_tabular_policy=self._current_policy.state_lookup[info_state],
          relizable_deviations=None,
          history=history[current_player].copy(),
          current_history_probs=copy.deepcopy(path_indices[current_player]),
      )
      prior_possible_actions = []
      for i in range(len(info_state_node.current_history_probs)):
        prior_possible_actions.append(
            info_state_node.current_history_probs[i][0]
        )
      prior_possible_actions.append(info_state_node.legal_actions)

      info_state_node.relizable_deviations = self._deviation_gen(
          len(info_state_node.legal_actions),
          info_state_node.history,
          prior_possible_actions,
      )
      self._info_state_nodes[info_state] = info_state_node

    legal_actions = state.legal_actions(current_player)

    for action in info_state_node.legal_actions:
      new_path_indices = copy.deepcopy(path_indices)
      new_path_indices[current_player].append(
          [legal_actions, info_state_node.index_in_tabular_policy]
      )
      new_history = copy.deepcopy(history)
      new_history[current_player].append(action)
      assert len(new_history[current_player]) == len(
          new_path_indices[current_player]
      )

      self._initialize_info_state_nodes(
          state.child(action), new_history, new_path_indices
      )

  def _update_current_policy(self, state, current_policy):
    """Updated the current policy.

    Updated in order so that memory reach probs are defined wrt to the new
    strategy. This is similar to EFR's policy update.

    Args:
      state: the state of which to update the strategy.
      current_policy: the (t+1)th strategy that is being recursively computed.
    """

    if state.is_terminal():
      return
    elif not state.is_chance_node():
      current_player = state.current_player()
      info_state = state.information_state_string(current_player)
      info_state_node = self._info_state_nodes[info_state]
      deviations = info_state_node.relizable_deviations
      for devation in range(len(deviations)):
        mem_reach_probs = create_probs_from_index(
            info_state_node.current_history_probs, current_policy
        )
        deviation_reach_prob = deviations[
            devation
        ].player_deviation_reach_probability(mem_reach_probs)
        y_increment = (
            max(0, info_state_node.cumulative_regret[devation])
            * deviation_reach_prob
        )
        info_state_node.y_values[deviations[devation]] = (
            info_state_node.y_values[deviations[devation]] + y_increment
        )

      state_policy = current_policy.policy_for_key(info_state)
      for action, value in self._regret_matching(info_state_node).items():
        state_policy[action] = value

      for action in info_state_node.legal_actions:
        new_state = state.child(action)
        self._update_current_policy(new_state, current_policy)
    else:
      for action, _ in state.chance_outcomes():
        new_state = state.child(action)
        self._update_current_policy(new_state, current_policy)

  def _regret_matching(self, info_set_node):
    """Returns an info state policy using regret matching.

    The info state policy returned is the one obtained by applying
    regret-matching over all deviations and time selection functions.

    Args:
      info_set_node: the info state node to compute the policy for.

    Returns:
      A dict of action -> prob for all legal actions of the
      info_set_node.
    """
    legal_actions = info_set_node.legal_actions
    num_actions = len(legal_actions)
    info_state_policy = None
    z = sum(info_set_node.y_values.values())

    # The fixed point solution can be directly obtained through the
    # weighted regret matrix if only external deviations are used.
    if self._external_only and z > 0:
      weighted_deviation_matrix = np.zeros((num_actions, num_actions))
      for dev in list(info_set_node.y_values.keys()):
        weighted_deviation_matrix += (
            info_set_node.y_values[dev] / z
        ) * dev.return_transform_matrix()
      new_strategy = weighted_deviation_matrix[:, 0]
      info_state_policy = dict(zip(legal_actions, new_strategy))

    # Full regret matching by finding the least squares solution to the
    # fixed point of the EFR regret matching function.
    elif z > 0:
      weighted_deviation_matrix = -np.eye(num_actions)

      for dev in list(info_set_node.y_values.keys()):
        weighted_deviation_matrix += (
            info_set_node.y_values[dev] / z
        ) * dev.return_transform_matrix()

      normalisation_row = np.ones(num_actions)
      weighted_deviation_matrix = np.vstack(
          [weighted_deviation_matrix, normalisation_row]
      )
      b = np.zeros(num_actions + 1)
      b[num_actions] = 1
      b = np.reshape(b, (num_actions + 1, 1))

      strategy = linalg.lstsq(weighted_deviation_matrix, b)[0]

      # Adopt same clipping strategy as paper author's code.
      np.clip(strategy, a_min=0, a_max=1, out=strategy)
      strategy = strategy / np.sum(strategy)

      info_state_policy = dict(zip(legal_actions, strategy[:, 0]))
    # Use a uniform strategy as sum of all regrets is negative.
    else:
      unif_policy_value = 1.0 / num_actions
      info_state_policy = {
          legal_actions[index]: unif_policy_value
          for index in range(num_actions)
      }
    return info_state_policy

  def _sample_action_from_policy(self, state, policy_dict, rng):
    """Sample an action from a policy distribution.

    Args:
      state: the current state.
      policy_dict: a dictionary mapping actions to probabilities.
      rng: a random number generator.

    Returns:
      A sampled action.
    """
    actions = list(policy_dict.keys())
    probs = [policy_dict[a] for a in actions]
    # Normalize in case of numerical errors
    probs = np.array(probs)
    probs = probs / probs.sum()
    return rng.choice(actions, p=probs)

  def _outcome_sampling(self, state, player, reach_probs, sample_probs, rng):
    """Runs outcome sampling MCEFR.

    Args:
      state: the current state.
      player: the player for which to compute regrets (update player).
      reach_probs: reach probabilities for each player [p0, p1, ..., chance].
      sample_probs: sampling probabilities along the trajectory.
      rng: random number generator.

    Returns:
      The utility for the update player at this state.
    """
    if state.is_terminal():
      return np.asarray(state.returns())[player] / sample_probs

    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action, prob = outcomes[rng.choice(len(outcomes))]
      new_state = state.child(action)
      new_reach_probs = reach_probs.copy()
      new_reach_probs[-1] *= prob
      new_sample_probs = sample_probs * prob
      return self._outcome_sampling(
          new_state, player, new_reach_probs, new_sample_probs, rng
      )

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)
    info_state_node = self._info_state_nodes[info_state]
    legal_actions = state.legal_actions()

    # Get current policy for this information state
    info_state_policy = self._get_infostate_policy(info_state)

    # Sample an action according to current policy
    sampled_action = self._sample_action_from_policy(
        state, info_state_policy, rng
    )
    action_prob = info_state_policy[sampled_action]

    # Update average policy (weighted by reach probability)
    reach_prob = reach_probs[current_player]
    info_state_node.cumulative_policy[sampled_action] += reach_prob

    # Recurse with sampled action
    new_state = state.child(sampled_action)
    new_reach_probs = reach_probs.copy()
    new_reach_probs[current_player] *= action_prob
    new_sample_probs = sample_probs * action_prob

    sampled_value = self._outcome_sampling(
        new_state, player, new_reach_probs, new_sample_probs, rng
    )

    # Update regrets only if this is the update player's information state
    if current_player == player:
      # Reset y values for this iteration
      info_state_node.y_values = collections.defaultdict(float)

      # Compute counterfactual reach probability
      counterfactual_reach_prob = np.prod(
          reach_probs[:current_player]
      ) * np.prod(reach_probs[current_player + 1:])

      # For each deviation, compute its counterfactual value
      deviations = info_state_node.relizable_deviations
      for deviation_index in range(len(deviations)):
        deviation = deviations[deviation_index]

        # Get deviation strategy
        deviation_strategy = deviation.deviate(
            strat_dict_to_array(info_state_policy)
        )

        # Compute deviation's expected value by sampling
        # We use the sampled action's value and importance sampling
        deviation_action_prob = deviation_strategy[legal_actions.index(sampled_action)][0]

        # Importance sampling correction
        if action_prob > 0:
          deviation_cf_value = sampled_value * (deviation_action_prob / action_prob)
        else:
          deviation_cf_value = 0.0

        # Compute memory reach probability for this deviation
        memory_reach_probs = create_probs_from_index(
            info_state_node.current_history_probs, self.current_policy()
        )
        player_current_memory_reach_prob = (
            deviation.player_deviation_reach_probability(memory_reach_probs)
        )

        # Compute regret for this deviation
        # regret = memory_reach_prob * counterfactual_reach_prob * (deviation_value - sampled_value)
        deviation_regret = player_current_memory_reach_prob * (
            counterfactual_reach_prob * deviation_cf_value
            - counterfactual_reach_prob * sampled_value
        )

        info_state_node.cumulative_regret[deviation_index] += deviation_regret

    return sampled_value

  def iteration(self, rng=None):
    """Performs one iteration of outcome sampling MCEFR.

    Args:
      rng: Optional random number generator. If None, uses numpy default.
    """
    if rng is None:
      rng = np.random.RandomState()

    # Sample trajectory for each player
    for player in range(self._num_players):
      reach_probs = np.ones(self._num_players + 1)
      sample_probs = 1.0
      self._outcome_sampling(
          self._root_node.clone(), player, reach_probs, sample_probs, rng
      )

    # Update current policy based on new regrets
    self._update_current_policy(self._root_node, self._current_policy)
    self._iteration += 1

  def current_policy(self):
    """Returns the current policy as a TabularPolicy.

    WARNING: The same object, updated in-place will be returned! You can copy
    it (or its `action_probability_array` field).

    For MCEFR, this policy does not necessarily have to converge.
    """
    return self._current_policy

  def average_policy(self):
    """Returns the average of all policies iterated.

    WARNING: The same object, updated in-place will be returned! You can copy it
    (or its `action_probability_array` field).

    This average policy converges to an equilibrium policy as the number
    of iterations increases (equilibrium type depends on learning
    deviations used).

    Returns:
      A `policy.TabularPolicy` object giving the time averaged policy.
    """
    _update_average_policy(self._average_policy, self._info_state_nodes)
    return self._average_policy

  def _get_infostate_policy(self, info_state_str):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
    info_state_node = self._info_state_nodes[info_state_str]
    prob_vec = self._current_policy.action_probability_array[
        info_state_node.index_in_tabular_policy
    ]
    return {
        action: prob_vec[action] for action in info_state_node.legal_actions
    }

  def return_cumulative_regret(self):
    """Returns a dictionary mapping.

    The mapping is from every information state to its associated regret
    (accumulated over all iterations).
    """
    return {
        list(self._info_state_nodes.keys())[i]: list(
            self._info_state_nodes.values()
        )[i].cumulative_regret
        for i in range(len(self._info_state_nodes.keys()))
    }
