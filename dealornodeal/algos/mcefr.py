

"""

mcefr combines efr's deviation-based regret minimization with mccfr's sampling
approach for improved efficiency on large games.

one iteration of mcefr consists of:
1) for each player, sample a trajectory using external sampling
2) compute deviation-based regrets along the sampled path
3) update strategy using efr's regret matching
"""

import collections
import copy
import numpy as np
from scipy import linalg
from open_spiel.python import policy
import pyspiel

# import deviation generation functions from efr
from dealornodeal.algos.efr import (
    return_blind_action,
    return_informed_action,
    return_blind_cf,
    return_informed_cf,
    return_blind_partial_sequence,
    return_cf_partial_sequence,
    return_cs_partial_sequence,
    return_twice_informed_partial_sequence,
    return_behavourial,
    array_to_strat_dict,
    create_probs_from_index,
)


class _InfoStateNode(object):
    """information state data for mcefr.

    stores deviation information and cumulative statistics for an info state.
    """

    def __init__(self, legal_actions, deviations, history, current_history_probs):
        self.legal_actions = legal_actions
        self.deviations = deviations
        self.history = history
        self.current_history_probs = current_history_probs

        # cumulative regrets per deviation (not per action like cfr)
        self.cumulative_regret = np.zeros(len(deviations), dtype=np.float64)

        # cumulative policy (per action, for average policy)
        self.cumulative_policy = np.zeros(len(legal_actions), dtype=np.float64)

        # y-values for current iteration (reset each iteration)
        self.y_values = {}


class AveragePolicy(policy.Policy):
    """average policy for mcefr."""

    def __init__(self, game, player_ids, infostates):
        super().__init__(game, player_ids)
        self._infostates = infostates

    def action_probabilities(self, state, player_id=None):
        """returns the mcefr average policy for a player in a state.

        if the policy is not defined for the provided state, a uniform
        random policy is returned.
        """
        if player_id is None:
            player_id = state.current_player()
        legal_actions = state.legal_actions()
        info_state_key = state.information_state_string(player_id)
        info_state_node = self._infostates.get(info_state_key, None)

        if info_state_node is None:
            return {a: 1 / len(legal_actions) for a in legal_actions}

        cumsum = info_state_node.cumulative_policy.sum()
        if cumsum == 0:
            return {a: 1 / len(legal_actions) for a in legal_actions}

        avg_policy = info_state_node.cumulative_policy / cumsum
        return {legal_actions[i]: avg_policy[i] for i in range(len(legal_actions))}


class MCEFRSolver(object):
    """monte carlo extensive-form regret minimization solver.

    implements external sampling mcefr with various deviation types.
    """

    def __init__(self, game, deviation_name):
        """initializer.

        args:
            game: the `pyspiel.game` to run on.
            deviation_name: the name of the deviation type to use.
                options: "blind action", "informed action", "blind cf",
                "informed counterfactual", "blind partial sequence",
                "counterfactual partial sequence", "casual partial sequence",
                "twice informed partial sequence", "single target behavioural"
        """
        assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
            "mcefr requires sequential games."
        )

        self._game = game
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()

        # map from info state string to _InfoStateNode
        self._infostates = {}

        # determine deviation generation function and external_only flag
        self._external_only = False

        if deviation_name in {"blind action"}:
            self._deviation_gen = return_blind_action
            self._external_only = True
        elif deviation_name in {"informed action"}:
            self._deviation_gen = return_informed_action
        elif deviation_name in {"blind cf", "blind counterfactual"}:
            self._deviation_gen = return_blind_cf
            self._external_only = True
        elif deviation_name in {"informed cf", "informed counterfactual"}:
            self._deviation_gen = return_informed_cf
        elif deviation_name in {"bps", "blind partial sequence"}:
            self._deviation_gen = return_blind_partial_sequence
            self._external_only = True
        elif deviation_name in {
            "cfps",
            "cf partial sequence",
            "counterfactual partial sequence",
        }:
            self._deviation_gen = return_cf_partial_sequence
        elif deviation_name in {"csps", "casual partial sequence"}:
            self._deviation_gen = return_cs_partial_sequence
        elif deviation_name in {"tips", "twice informed partial sequence"}:
            self._deviation_gen = return_twice_informed_partial_sequence
        elif deviation_name in {"bhv", "single target behavioural", "behavioural"}:
            self._deviation_gen = return_behavourial
        else:
            raise ValueError(f"unsupported deviation type: {deviation_name}")

    def iteration(self):
        """performs one iteration of external sampling mcefr.

        an iteration consists of one episode for each player as the update player.
        """
        for player in range(self._num_players):
            # reset yvalues for all info states at start of iteration
            for info_state_node in self._infostates.values():
                info_state_node.y_values = {}

            # run one episode with this player as the update player
            history = {p: [] for p in range(self._num_players)}
            history_info_states = {p: [] for p in range(self._num_players)}
            history_legal_actions = {p: [] for p in range(self._num_players)}
            self._update_regrets(
                self._root_node,
                player,
                history,
                history_info_states,
                history_legal_actions,
                reach_probs=np.ones(self._num_players + 1)
            )

        # update strategies for all visited info states using y-values
        self._update_all_strategies()

    def _lookup_or_create_infostate(self, state, current_player, history, history_info_states, history_legal_actions):
        """looks up or creates an info state node with lazy deviation generation.

        args:
            state: current game state
            current_player: player at this info state
            history: dict mapping player -> list of actions taken
            history_info_states: dict mapping player -> list of info_state_keys visited
            history_legal_actions: dict mapping player -> list of legal_actions at each prior state

        returns:
            _InfoStateNode for this information state
        """
        info_state_key = state.information_state_string(current_player)

        # if we've seen this info state before, return it
        if info_state_key in self._infostates:
            return self._infostates[info_state_key]

        # create new info state node with lazy deviation generation
        legal_actions = state.legal_actions(current_player)

        # use actual tracked legal actions from history for proper deviation generation
        prior_possible_actions = history_legal_actions[current_player].copy()
        prior_possible_actions.append(legal_actions)

        deviations = self._deviation_gen(
            len(legal_actions),
            history[current_player].copy(),
            prior_possible_actions,
        )

        # create and store the info state node
        # store history of info state keys for memory reach probability computation
        info_state_node = _InfoStateNode(
            legal_actions=legal_actions,
            deviations=deviations,
            history=history[current_player].copy(),
            current_history_probs=history_info_states[current_player].copy(),
        )

        self._infostates[info_state_key] = info_state_node
        return info_state_node

    def _get_current_policy(self, info_state_node):
        """computes current policy from regrets using efr regret matching.

        args:
            info_state_node: the info state node

        returns:
            numpy array of probabilities for each action
        """
        num_actions = len(info_state_node.legal_actions)

        # compute y-values if not already computed this iteration
        if not info_state_node.y_values:
            # y-values need to be computed using current policy, but we're computing
            # current policy... for now, use regret matching directly
            # this will be updated properly in _update_all_strategies
            pass

        z = sum(info_state_node.y_values.values()) if info_state_node.y_values else 0

        # external-only deviations: direct computation
        if self._external_only and z > 0:
            weighted_deviation_matrix = np.zeros((num_actions, num_actions))
            for dev in info_state_node.y_values.keys():
                weighted_deviation_matrix += (
                    info_state_node.y_values[dev] / z
                ) * dev.return_transform_matrix()
            policy_array = weighted_deviation_matrix[:, 0]
            return policy_array

        # general deviations: least squares solution
        elif z > 0:
            weighted_deviation_matrix = -np.eye(num_actions)
            for dev in info_state_node.y_values.keys():
                weighted_deviation_matrix += (
                    info_state_node.y_values[dev] / z
                ) * dev.return_transform_matrix()

            normalisation_row = np.ones(num_actions)
            weighted_deviation_matrix = np.vstack(
                [weighted_deviation_matrix, normalisation_row]
            )
            b = np.zeros(num_actions + 1)
            b[num_actions] = 1
            b = np.reshape(b, (num_actions + 1, 1))

            strategy = linalg.lstsq(weighted_deviation_matrix, b)[0]
            np.clip(strategy, a_min=0, a_max=1, out=strategy)
            strategy = strategy / np.sum(strategy)

            return strategy[:, 0]
        else:
            # uniform policy if no positive regrets
            return np.ones(num_actions, dtype=np.float64) / num_actions

    def _update_regrets(self, state, player, history, history_info_states, history_legal_actions, reach_probs):
        """runs external sampling episode and updates regrets.

        uses external sampling: samples opponent actions, iterates over our actions.

        args:
            state: current game state
            player: player to update regrets for
            history: dict mapping player_id -> list of actions
            history_info_states: dict mapping player_id -> list of info_state_keys
            history_legal_actions: dict mapping player_id -> list of legal_actions at prior states
            reach_probs: array of reach probabilities [p0, p1, ..., chance]

        returns:
            value for the updating player
        """
        if state.is_terminal():
            return state.player_return(player)

        if state.is_chance_node():
            # sample chance outcome
            outcomes, probs = zip(*state.chance_outcomes())
            outcome = np.random.choice(outcomes, p=probs)
            new_reach_probs = reach_probs.copy()
            new_reach_probs[-1] *= probs[outcomes.index(outcome)]
            return self._update_regrets(
                state.child(outcome), player, history, history_info_states, history_legal_actions, new_reach_probs
            )

        current_player = state.current_player()
        info_state_key = state.information_state_string(current_player)

        # lookup or create info state node (lazy generation)
        info_state_node = self._lookup_or_create_infostate(
            state, current_player, history, history_info_states, history_legal_actions
        )

        # get current policy using simple regret matching
        # (we'll update with proper y-values later)
        if info_state_node.cumulative_regret.sum() > 0:
            positive_regrets = np.maximum(info_state_node.cumulative_regret, 0)
            regret_sum = positive_regrets.sum()
            if regret_sum > 0:
                # map deviation regrets to action regrets (approximate)
                action_regrets = np.zeros(len(info_state_node.legal_actions))
                for dev_idx, deviation in enumerate(info_state_node.deviations):
                    # for external deviations, map to corresponding action
                    if hasattr(deviation.local_swap_transform, 'target_action'):
                        target_action = deviation.local_swap_transform.target_action
                        action_regrets[target_action] += positive_regrets[dev_idx]
                policy_array = action_regrets / action_regrets.sum() if action_regrets.sum() > 0 else np.ones(len(info_state_node.legal_actions)) / len(info_state_node.legal_actions)
            else:
                policy_array = np.ones(len(info_state_node.legal_actions)) / len(info_state_node.legal_actions)
        else:
            policy_array = np.ones(len(info_state_node.legal_actions)) / len(info_state_node.legal_actions)

        legal_actions = info_state_node.legal_actions
        num_actions = len(legal_actions)

        value = 0
        child_values = np.zeros(num_actions, dtype=np.float64)

        if current_player != player:
            # opponent node: sample one action
            action_idx = np.random.choice(np.arange(num_actions), p=policy_array)

            # update history for this action
            new_history = copy.deepcopy(history)
            new_history[current_player].append(legal_actions[action_idx])

            # track info state history for memory reach prob computation
            new_history_info_states = copy.deepcopy(history_info_states)
            new_history_info_states[current_player].append(info_state_key)

            # track legal actions at this state
            new_history_legal_actions = copy.deepcopy(history_legal_actions)
            new_history_legal_actions[current_player].append(legal_actions)

            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= policy_array[action_idx]

            value = self._update_regrets(
                state.child(legal_actions[action_idx]),
                player,
                new_history,
                new_history_info_states,
                new_history_legal_actions,
                new_reach_probs
            )

            # update average policy (simple averaging at opponent nodes)
            # CRITICAL: must update ALL actions, not just the sampled one
            # this matches mccfr external sampling simple averaging
            for action_idx_avg in range(num_actions):
                info_state_node.cumulative_policy[action_idx_avg] += policy_array[action_idx_avg]

        else:
            # our node: iterate over all actions (needed for deviation regrets)
            for action_idx in range(num_actions):
                new_history = copy.deepcopy(history)
                new_history[current_player].append(legal_actions[action_idx])

                # track info state history
                new_history_info_states = copy.deepcopy(history_info_states)
                new_history_info_states[current_player].append(info_state_key)

                # track legal actions at this state
                new_history_legal_actions = copy.deepcopy(history_legal_actions)
                new_history_legal_actions[current_player].append(legal_actions)

                new_reach_probs = reach_probs.copy()
                new_reach_probs[current_player] *= policy_array[action_idx]

                child_values[action_idx] = self._update_regrets(
                    state.child(legal_actions[action_idx]),
                    player,
                    new_history,
                    new_history_info_states,
                    new_history_legal_actions,
                    new_reach_probs
                )
                value += policy_array[action_idx] * child_values[action_idx]

            # update deviation regrets
            self._update_deviation_regrets(
                info_state_node,
                info_state_key,
                child_values,
                value,
                policy_array,
                reach_probs,
                current_player,
                history_info_states
            )

            # update average policy
            reach_prob = reach_probs[current_player]
            for action_idx in range(num_actions):
                info_state_node.cumulative_policy[action_idx] += (
                    reach_prob * policy_array[action_idx]
                )

        return value

    def _update_deviation_regrets(
        self, info_state_node, info_state_key, child_values,
        state_value, policy_array, reach_probs, current_player,
        history_info_states
    ):
        """updates cumulative regrets for all deviations at this info state.

        args:
            info_state_node: the info state node
            info_state_key: info state string key
            child_values: values of all child states
            state_value: value of current state
            policy_array: current policy as array
            reach_probs: reach probabilities
            current_player: current player
            history_info_states: dict mapping player -> list of info_state_keys
        """
        # compute counterfactual reach probability
        counterfactual_reach_prob = (
            np.prod(reach_probs[:current_player]) *
            np.prod(reach_probs[current_player + 1:])
        )

        # construct memory reach probabilities from visited info states
        # this maps info state keys to their current policies
        memory_reach_probs = []
        for info_key in history_info_states[current_player]:
            if info_key in self._infostates:
                hist_info_node = self._infostates[info_key]
                # compute current policy for this historical state
                hist_policy_array = self._get_current_policy_simple(hist_info_node)
                # convert to dict mapping action -> prob
                policy_dict = {
                    hist_info_node.legal_actions[i]: hist_policy_array[i]
                    for i in range(len(hist_info_node.legal_actions))
                }
                memory_reach_probs.append(policy_dict)

        # for each deviation, compute regret
        for deviation_idx, deviation in enumerate(info_state_node.deviations):
            # compute value of deviating
            deviation_strategy = deviation.deviate(
                policy_array.reshape(-1, 1)
            )
            deviation_value = np.inner(
                deviation_strategy[:, 0],
                child_values
            )

            # compute memory reach probability using deviation's method
            memory_reach_prob = deviation.player_deviation_reach_probability(
                memory_reach_probs
            )

            # compute regret for this deviation
            deviation_regret = memory_reach_prob * (
                counterfactual_reach_prob * (deviation_value - state_value)
            )

            # accumulate regret
            info_state_node.cumulative_regret[deviation_idx] += deviation_regret

    def _get_current_policy_simple(self, info_state_node):
        """computes current policy from cumulative regrets using simple regret matching.

        this is a simplified version for computing memory reach probabilities.

        args:
            info_state_node: the info state node

        returns:
            numpy array of probabilities for each action
        """
        num_actions = len(info_state_node.legal_actions)

        # simple regret matching on action-aggregated regrets
        if info_state_node.cumulative_regret.sum() > 0:
            # aggregate deviation regrets to action regrets
            action_regrets = np.zeros(num_actions)
            for dev_idx, deviation in enumerate(info_state_node.deviations):
                if hasattr(deviation.local_swap_transform, 'target_action'):
                    target_action = deviation.local_swap_transform.target_action
                    action_regrets[target_action] += max(
                        0, info_state_node.cumulative_regret[dev_idx]
                    )

            regret_sum = action_regrets.sum()
            if regret_sum > 0:
                return action_regrets / regret_sum

        # uniform policy if no positive regrets
        return np.ones(num_actions, dtype=np.float64) / num_actions

    def _update_all_strategies(self):
        """updates strategies for all info states using y-values.

        this is called after all episodes in an iteration to properly
        compute y-values and update strategies using efr's regret matching.
        """
        # for each info state, compute y-values from cumulative regrets
        for info_state_key, info_state_node in self._infostates.items():
            # compute current policy for memory reach probabilities
            current_policy = self._get_current_policy_simple(info_state_node)

            # compute y-values for each deviation
            for deviation_idx, deviation in enumerate(info_state_node.deviations):
                # get memory reach probability
                memory_reach_probs = []
                for hist_info_key in info_state_node.current_history_probs:
                    if hist_info_key in self._infostates:
                        hist_node = self._infostates[hist_info_key]
                        hist_policy = self._get_current_policy_simple(hist_node)
                        policy_dict = {
                            hist_node.legal_actions[i]: hist_policy[i]
                            for i in range(len(hist_node.legal_actions))
                        }
                        memory_reach_probs.append(policy_dict)

                memory_reach_prob = deviation.player_deviation_reach_probability(
                    memory_reach_probs
                )

                # y-value = max(0, cumulative_regret) * memory_reach_prob
                y_increment = (
                    max(0, info_state_node.cumulative_regret[deviation_idx])
                    * memory_reach_prob
                )
                info_state_node.y_values[deviation] = y_increment

        # note: current policies during traversal already use regret matching
        # the y-values are primarily used for the efr regret matching formula
        # in future iterations

    def average_policy(self):
        """returns the average policy.

        returns:
            an AveragePolicy object
        """
        return AveragePolicy(
            self._game,
            list(range(self._num_players)),
            self._infostates
        )
