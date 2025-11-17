import pyspiel
from typing import List, Dict, Tuple
import numpy as np


import pdb

import faulthandler
import sys
faulthandler.enable(file=sys.stderr)


_NUM_PLAYERS = 2
_MAX_TURNS = 10
_NUM_ITEM_TYPES = 3
_POOL_MAX_NUM_ITEMS = 7  # maximum items per type in pool
_TOTAL_VALUE_ALL_ITEMS = 10  # total value constraint for each player


class Instance:
    """represents a single game instance with pool and player values"""
    def __init__(self, pool: List[int], values: List[List[int]]):
        """
        pool: list of 3 integers, quantities of each item type
        values: list of 2 lists, each with 3 integers (values for each player)
        """
        self.pool = pool
        self.values = values  # values[player][item_type]

    def __str__(self):
        return f"pool={self.pool} p0={self.values[0]} p1={self.values[1]}"


# temporary placeholders for num distinct actions and max chance outcomes
# these will be set properly once we create all offers and load instances
_NUM_DISTINCT_ACTIONS = 200  # placeholder: 120 offers + 1 accept, set high for safety
_MAX_CHANCE_NODE_OUTCOMES = 1002  # placeholder: 1000 instances + continue + end


_GAME_TYPE = pyspiel.GameType(
    short_name="python_deal_or_no_deal",
    long_name="Python Deal or No Deal Negotiation",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_observation_string=True,
    provides_information_state_tensor=True,
    provides_observation_tensor=True,
    parameter_specification={
        "max_turns": _MAX_TURNS,
        "discount": 1.0,  # discount factor applied after turn 2
        "prob_end": 0.0,  # probability of early termination after turn 2
        "max_num_instances": 100,  # maximum number of instances to load
    },
)


_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_DISTINCT_ACTIONS,
    max_chance_outcomes=_MAX_CHANCE_NODE_OUTCOMES,
    num_players=_NUM_PLAYERS,
    min_utility=0.0,
    max_utility=_TOTAL_VALUE_ALL_ITEMS,  # max utility is 10 per player
    utility_sum=None,
    max_game_length=_MAX_TURNS + 1,  # max_turns offers + initial chance node
)




##### 1  DEFINE STATE

'''
runtime model (HOW THE IMPLEMENTATION WORKS, AKA HOW DealOrNoDealGame and DealOrNoDealState work together):

DealOrNoDealGame is just a ruleset that describes the rule parameters of the game, nothing more.
It does not store the actual game progression itself, that is what DealOrNoDealState is for.

To start: A DealOrNoDealState is instantiated once, and then as the game progresses, it calls:

    1. legal_actions(state) #to see what possible actions can be made by the player whose turn it is
    2. state.apply_action(action) #apply action and update state in place
    3. check if state.is_terminal() and if so, end the game


'''


class DealOrNoDealState(pyspiel.State):
    """state for deal or no deal game"""

    def __init__(self, game) -> None:
        super().__init__(game)

        self._num_players = _NUM_PLAYERS

        # cache game parameters to avoid repeated get_game() calls
        self._max_turns = game.max_turns
        self._prob_end = game.prob_end
        self._discount_factor = game.discount

        # game instance (pool and player values)
        # set by first chance node selecting an instance
        self._instance = None

        # game progression
        self._cur_player = pyspiel.PlayerId.CHANCE  # start at chance node
        self._next_player = 0  # tracks who goes after a chance node
        self._turn = 0  # counts player offers (not chance nodes)
        self._offers = []  # history of offers
        self._agreement = False  # did someone accept?
        self._game_ended = False  # probabilistic early termination flag
        self._discount = 1.0  # cumulative discount factor applied to returns

        

    def __str__(self) -> str:
        """for internal use by open_spiel"""
        if self.is_terminal():
            return "terminal"
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return "chance"
        return self.information_state_string(self.current_player())



    def current_player(self) -> int:
        """returns the current player id or special values for chance/terminal"""
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        return self._cur_player  # can be CHANCE, 0, or 1
    



    def is_terminal(self) -> bool:
        """returns true if game has ended"""
        return self._agreement or self._game_ended or len(self._offers) >= self._max_turns


    def returns(self) -> List[float]:
        """compute returns based on last accepted offer"""
        if not self._agreement:
            # no agreement reached, both players get 0
            return [0.0, 0.0]

        # proposer is determined by number of offers (proposer made last offer)
        # offers are indexed 0, 1, 2, ... so player 0 makes even-indexed offers
        proposing_player = (len(self._offers) - 1) % _NUM_PLAYERS #get last offer before acceptance
        other_player = 1 - proposing_player

        last_offer = self._offers[-1]
        returns = [0.0, 0.0]

        # proposer gets utility of items in the offer
        # other player gets utility of remaining items

        #for each item, calculate the utility * quantity of each item in each player's final basket
        for i in range(_NUM_ITEM_TYPES):
            returns[proposing_player] += self._instance.values[proposing_player][i] * last_offer[i] 
            returns[other_player] += self._instance.values[other_player][i] * (
                self._instance.pool[i] - last_offer[i]
            )

        # apply cumulative discount
        returns[0] *= self._discount
        returns[1] *= self._discount

        return returns






    def _apply_action(self, action: int) -> None:
        """apply an action to the state"""
        game = self.get_game()

        if self.current_player() == pyspiel.PlayerId.CHANCE:
            # first chance node: select instance
            if self._instance is None:
                self._instance = game.get_instance(action)
                self._cur_player = 0  # player 0 starts after instance selected
                return

            # later chance nodes: continue or end the game
            if action == game.continue_outcome():
                # game continues, switch to next player
                self._cur_player = self._next_player
            else:
                # game ends early
                self._game_ended = True
                self._cur_player = pyspiel.PlayerId.TERMINAL
            return

        # player action: either accept or make an offer
        if action == self._accept_action_id():
            self._agreement = True
        else:
            offer = self._decode_offer_action(action)
            self._offers.append(offer)

            # apply discount after turn 2 (after first 2 offers have been made)
            if len(self._offers) >= 3 and self._discount_factor < 1.0:
                self._discount *= self._discount_factor

            # after turn 2 (i.e., after both players have made at least one offer),
            # insert a chance node if prob_end > 0
            if len(self._offers) >= 2 and self._prob_end > 0.0:
                self._next_player = 1 - self._cur_player
                self._cur_player = pyspiel.PlayerId.CHANCE
            else:
                # no chance node, just switch players
                self._cur_player = 1 - self._cur_player



    def _is_legal_offer(self, offer: List[int]) -> bool:
        """check if an offer is legal given the current pool"""
        for i in range(_NUM_ITEM_TYPES):
            if offer[i] > self._instance.pool[i]: #if a player tried to propose a quantity greater than available
                return False
        return True

    def _legal_actions(self, player: int) -> List[int]:
        """returns list of legal action ids for the given player"""
        if self.is_terminal():
            return []

        game = self.get_game()

        if player == pyspiel.PlayerId.CHANCE:
            # first chance node: select from all instances
            if self._instance is None:
                return list(range(len(game.all_instances))) #select a starting state (utility vectors, etc)
            # later chance nodes: continue or end
            else:
                return [game.continue_outcome(), game.end_outcome()]

        # player can accept if there's an offer on the table
        actions = []
        if len(self._offers) > 0:
            actions.append(self._accept_action_id())

        # player can make any offer that doesn't exceed the pool
        for i, offer in enumerate(game.all_offers):
            if self._is_legal_offer(offer):
                actions.append(i)

        return actions



    def chance_outcomes(self) -> List[Tuple[int, float]]:
        """returns list of (action, probability) pairs for chance node"""
        if self.current_player() != pyspiel.PlayerId.CHANCE:
            return []

        game = self.get_game()

        # first chance node: uniformly select instance
        if self._instance is None:
            num_instances = len(game.all_instances)
            prob = 1.0 / num_instances
            return [(i, prob) for i in range(num_instances)]

        # later chance nodes: continue or end
        return [
            (game.continue_outcome(), 1.0 - self._prob_end),
            (game.end_outcome(), self._prob_end)
        ]





    def information_state_string(self, player: int = None) -> str:
        """returns information state string for the given player"""
        if self.is_terminal():
            return "terminal"
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return "chance"
        if player is None:
            player = self.current_player()

        # player's private information: their values and the pool
        pool_str = ",".join(map(str, self._instance.pool))
        values_str = ",".join(map(str, self._instance.values[player]))

        # public information: offer history
        offers_str = " ".join([f"P{i%2}:{','.join(map(str, o))}" for i, o in enumerate(self._offers)])

        return f"pool:{pool_str} my_values:{values_str} offers:[{offers_str}] agreement:{self._agreement}"







    def observation_string(self, player: int = None) -> str:
        """return observation string visible to the given player"""
        if self.is_terminal():
            return "terminal"
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return "chance"
        if player is None:
            player = self.current_player()

        # player can see pool, their own values, and most recent offer
        pool_str = ",".join(map(str, self._instance.pool))
        values_str = ",".join(map(str, self._instance.values[player]))
        last_offer_str = ",".join(map(str, self._offers[-1])) if self._offers else "none"

        return f"pool:{pool_str} my_values:{values_str} last_offer:{last_offer_str} agreement:{self._agreement}"









    def information_state_tensor(self, player: int) -> np.ndarray:
        """vectorized information state for the given player using thermometer encoding"""
        game = self.get_game()
        tensor_size = game.information_state_tensor_shape()[0]
        values = np.zeros(tensor_size, dtype=np.float32)

        # no observations at chance nodes
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return values

        offset = 0

        # agreement reached?
        if self._agreement:
            values[offset] = 1.0
        offset += 1

        # how many trade offers have happened? (one-hot encoding)
        values[offset + len(self._offers)] = 1.0
        offset += self._max_turns + 1

        # pool (thermometer encoding)
        for i in range(_NUM_ITEM_TYPES):
            for j in range(self._instance.pool[i] + 1):
                values[offset + j] = 1.0
            offset += _POOL_MAX_NUM_ITEMS + 1

        # my values (thermometer encoding)
        for i in range(_NUM_ITEM_TYPES):
            for j in range(self._instance.values[player][i] + 1):
                values[offset + j] = 1.0
            offset += _TOTAL_VALUE_ALL_ITEMS + 1

        # all offers (thermometer encoding)
        for k in range(self._max_turns):
            if k < len(self._offers):
                for i in range(_NUM_ITEM_TYPES):
                    for j in range(self._offers[k][i] + 1):
                        values[offset + j] = 1.0
                    offset += _POOL_MAX_NUM_ITEMS + 1
            else:
                # no offer at this position, leave as zeros
                offset += (_POOL_MAX_NUM_ITEMS + 1) * _NUM_ITEM_TYPES

        return values


    def observation_tensor(self, player: int) -> np.ndarray:
        """vectorized observation visible to the given player using thermometer encoding"""
        game = self.get_game()
        tensor_size = game.observation_tensor_shape()[0]
        values = np.zeros(tensor_size, dtype=np.float32)

        # no observations at chance nodes
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return values

        offset = 0

        # agreement reached?
        if self._agreement:
            values[offset] = 1.0
        offset += 1

        # how many trade offers have happened? (one-hot encoding)
        values[offset + len(self._offers)] = 1.0
        offset += self._max_turns + 1

        # pool (thermometer encoding)
        for i in range(_NUM_ITEM_TYPES):
            for j in range(self._instance.pool[i] + 1):
                values[offset + j] = 1.0
            offset += _POOL_MAX_NUM_ITEMS + 1

        # my values (thermometer encoding)
        for i in range(_NUM_ITEM_TYPES):
            for j in range(self._instance.values[player][i] + 1):
                values[offset + j] = 1.0
            offset += _TOTAL_VALUE_ALL_ITEMS + 1

        # just the last offer (thermometer encoding)
        if self._offers:
            for i in range(_NUM_ITEM_TYPES):
                for j in range(self._offers[-1][i] + 1):
                    values[offset + j] = 1.0
                offset += _POOL_MAX_NUM_ITEMS + 1
        else:
            # no offer yet, leave as zeros
            offset += (_POOL_MAX_NUM_ITEMS + 1) * _NUM_ITEM_TYPES

        return values







    def _accept_action_id(self) -> int:
        """returns the action id for accepting the current offer"""
        game = self.get_game()
        # accept action is the last action (after all offers)
        return len(game.all_offers)

    def _decode_offer_action(self, action: int) -> List[int]:
        """decode action id into offer vector"""
        game = self.get_game()
        return game.all_offers[action]















##### 2 DEFINE GAME


class DealOrNoDealGame(pyspiel.Game):
    """deal or no deal negotiation game"""

    def __init__(self, params=None) -> None:
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
        game_parameters = self.get_parameters()

        # game parameters
        self.max_turns = int(game_parameters.get("max_turns", _MAX_TURNS))
        self.discount = float(game_parameters.get("discount", 1.0))
        self.prob_end = float(game_parameters.get("prob_end", 0.0))
        self.max_num_instances = int(game_parameters.get("max_num_instances", 100))

        # create all possible offers (0 to pool_max items per type)
        self._create_offers()

        # load or generate instances
        self._load_instances()

    def num_distinct_actions(self) -> int:
        """returns number of distinct player actions (all offers + agree)"""
        return len(self.all_offers) + 1

    def max_chance_outcomes(self) -> int:
        """returns maximum number of chance outcomes (instances + continue + end)"""
        return len(self.all_instances) + 2

    def max_game_length(self) -> int:
        """returns maximum game length"""
        return self.max_turns + 1

    def _create_offers(self):
        """generate all possible offers up to pool_max items per type"""
        self.all_offers = []
        # generate all combinations of items from 0 to _POOL_MAX_NUM_ITEMS
        for i0 in range(_POOL_MAX_NUM_ITEMS + 1):
            for i1 in range(_POOL_MAX_NUM_ITEMS + 1):
                for i2 in range(_POOL_MAX_NUM_ITEMS + 1):
                    # only include offers with total items <= pool max
                    if i0 + i1 + i2 <= _POOL_MAX_NUM_ITEMS:
                        self.all_offers.append([i0, i1, i2])

    def _load_instances(self):
        """load game instances from file"""
        from pathlib import Path

        self.all_instances = []

        # path to instances file relative to this file
        current_file = Path(__file__).resolve()
        instances_file = current_file.parent / "cpp" / "bargaining_instances1000.txt"

        with open(instances_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # skip empty lines
                    continue

                # parse: "pool p0_values p1_values"
                parts = line.split()
                if len(parts) != 3:
                    continue

                # parse pool
                pool = list(map(int, parts[0].split(',')))

                # parse player values
                p0_values = list(map(int, parts[1].split(',')))
                p1_values = list(map(int, parts[2].split(',')))

                self.all_instances.append(Instance(pool, [p0_values, p1_values]))

                # stop if we've loaded enough instances
                if len(self.all_instances) >= self.max_num_instances:
                    break

    def get_instance(self, idx: int) -> Instance:
        """get instance by index"""
        return self.all_instances[idx]

    def continue_outcome(self) -> int:
        """action id for 'continue' outcome in probabilistic termination"""
        return len(self.all_instances)

    def end_outcome(self) -> int:
        """action id for 'end' outcome in probabilistic termination"""
        return len(self.all_instances) + 1

    def new_initial_state(self) -> DealOrNoDealState:
        return DealOrNoDealState(self)

    def observation_tensor_shape(self):
        """returns the shape of observation tensors"""
        return [
            1 +                                          # agreement reached?
            self.max_turns + 1 +                         # how many offers have happened (one-hot)
            (_POOL_MAX_NUM_ITEMS + 1) * _NUM_ITEM_TYPES +  # pool (thermometer encoding)
            (_TOTAL_VALUE_ALL_ITEMS + 1) * _NUM_ITEM_TYPES +  # my values (thermometer encoding)
            (_POOL_MAX_NUM_ITEMS + 1) * _NUM_ITEM_TYPES    # most recent offer (thermometer encoding)
        ]

    def information_state_tensor_shape(self):
        """returns the shape of information state tensors"""
        return [
            1 +                                          # agreement reached?
            self.max_turns + 1 +                         # how many offers have happened (one-hot)
            (_POOL_MAX_NUM_ITEMS + 1) * _NUM_ITEM_TYPES +  # pool (thermometer encoding)
            (_TOTAL_VALUE_ALL_ITEMS + 1) * _NUM_ITEM_TYPES +  # my values (thermometer encoding)
            self.max_turns * (_POOL_MAX_NUM_ITEMS + 1) * _NUM_ITEM_TYPES  # all offers (thermometer encoding)
        ]

    def make_py_observer(self, iig_obs_type=None, params=None):
        """returns an observer for this game (openspiel compatibility)"""
        # use default observer for now
        return None



















pyspiel.register_game(_GAME_TYPE, DealOrNoDealGame)