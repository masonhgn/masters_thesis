import pyspiel
from typing import List, Dict, Tuple
import numpy as np


_NUM_PLAYERS = 2
_MAX_TURNS = 10
_NUM_ITEM_TYPES = 3
_NUM_ITEMS_PER_TYPE = 2 #quantities of items
_MAX_SINGLE_ITEM_VALUE = 10






_NUM_DISTINCT_ACTIONS = 1 + (_NUM_ITEMS_PER_TYPE + 1) ** _NUM_ITEM_TYPES #accept + all possible offers
_MAX_CHANCE_NODE_OUTCOMES = (_MAX_SINGLE_ITEM_VALUE + 1) ** _NUM_ITEM_TYPES 


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
        "num_item_types": _NUM_ITEM_TYPES,
        "items_per_type": _NUM_ITEMS_PER_TYPE,
        "utility_max":_MAX_SINGLE_ITEM_VALUE # !NOTE: is this for sure the maximum single item value? or is it the same utility_max as in _GAME_INFO?
    },
)


_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions = _NUM_DISTINCT_ACTIONS,
    max_chance_outcomes = _MAX_CHANCE_NODE_OUTCOMES,
    num_players = _NUM_PLAYERS,
    min_utility = 0,
    max_utility = _MAX_SINGLE_ITEM_VALUE * _NUM_ITEM_TYPES * _NUM_ITEMS_PER_TYPE,
    utility_sum = 0.0,
    max_game_length= _MAX_TURNS, #each player acts on alternating timesteps
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
    '''state for deal or no deal game'''

    def __init__(self, game) -> None:
        super().__init__(game)

        #params
        self._num_players = game.num_players()
        self._max_turns = game.max_turns
        self._num_item_types = game.num_item_types
        self._items_per_type = game.items_per_type
        self._utility_max = game.utility_max

        #random stuff
        self._cur_player = pyspiel.PlayerId.CHANCE   # randomly decide first player
        self._turn = 0
        self._offers = [] #history of offers (list of allocations)
        self._agreement = False #did someone accept?

        #pool of items: vector length = num_item_types
        # example: if 3 types, each with 2 items -> [2,2,2]
        self._pool = [self._items_per_type] * self._num_item_types

        # Utilities: list of utility vectors, one per player
        #filled at chance node later
        self._utilities = [None for _ in range(self._num_players)]

        # store payoffs when terminal
        self._returns = [0.0 for _ in range(self._num_players)]


    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL #if game is over
        if any(u is None for u in self._utilities):
            return pyspiel.PlayerId.CHANCE # if game hasn't started yet, pick a random player to go first
        return self._cur_player #game is ongoing, whose turn is it?
    

    def is_terminal(self) -> bool:
        return self._agreement or self._turn >= self._max_turns


    def returns(self) -> List[float]:

        '''
        POSSIBLE CONDITIONS:

        1. game is not terminal:
            return Nonetype

        2. turn limit has been reached:
            return 0 utility for all players

        3. last proposal was not valid (e.g. player asked for higher quantity of an item than available):
            return return 0 utility for all players

        4. last proposal was valid:
            - let the player who made the last proposal be player A
            - player A final utility = dot product of player A utilities and last proposal (e.g. last_proposal = [2,0,1], utilities = [4,1,3], return 8+0+3 = 11)
            - player B final utility = dot product of player B utilities and (total allocation vector - last proposal)
        '''

        #1
        if not self.is_terminal(): #if we haven't reached an agreement, don't return anything
            return [0.0 for _ in range(self._num_players)]
        
        #2,3
        #if agreement hasn't been reached and we've run out of turns (which is implied here since self.is_terminal is true but self._agreement is false)
        #or if there have been no offers
        if not self._agreement or len(self._offers) == 0:
            return [0.0 for _ in range(self._num_players)]
        
        last_offer = self._offers[-1]  # e.g. [2,0,1]
        total_allocation = [self._items_per_type] * self._num_item_types


        #4
        returns = []
        for pid in range(self._num_players):
            if pid == self._cur_player:
                # proposer payoff
                util = np.dot(self._utilities[pid], last_offer)
            else:
                # responder payoff = remainder
                remainder = [t - o for t, o in zip(total_allocation, last_offer)]
                util = np.dot(self._utilities[pid], remainder)
            returns.append(float(util))

        return returns
    






    def _apply_action(self, action: int) -> None:
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            #if we are still assigning utilities, then get the next player with no utility, and assign utility vector
            next_pid = [i for i, u in enumerate(self._utilities) if u is None][0]


            #print(action)
            self._utilities[next_pid] = self._decode_utility_action(action)
            # If still another player missing utilities, stay in chance
            if any(u is None for u in self._utilities):
                self._cur_player = pyspiel.PlayerId.CHANCE
            else:
                self._cur_player = 0  # first player starts
            return

        #have we reached an agreement?
        if action == self._accept_action_id():
            self._agreement = True
        else:
            offer = self._decode_offer_action(action)
            self._offers.append(offer)
            self._cur_player = 1 - self._cur_player  # alternate turn
        self._turn += 1



    def _legal_actions(self, player: int) -> List[int]:
        if self.is_terminal():
            return []

        if player == pyspiel.PlayerId.CHANCE:
            #all possible utility vectors
            return [self._encode_utility_action(v) for v in self._all_utility_vectors()]

        actions = []
        actions.append(self._accept_action_id()) #player could accept

        #or player could make any of these offers
        actions.extend([self._encode_offer_action(offer) for offer in self._all_offers()])
        return actions



    def chance_outcomes(self) -> List[Tuple[int,float]]:
        if self.current_player() != pyspiel.PlayerId.CHANCE:
            return []
        outcomes = []
        all_utils = self._all_utility_vectors()
        prob = 1.0 / len(all_utils)
        for vec in all_utils:
            outcomes.append((self._encode_utility_action(vec), prob))
        return outcomes






    def information_state_string(self, player: int = None) -> str:
        """return infoset string for given player or current player if None"""
        if self.is_terminal():
            return "terminal"

        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return "chance"

        if player is None:
            player = self.current_player()

        util_str = "?" if self._utilities[player] is None else str(self._utilities[player])
        offers_str = "; ".join([str(o) for o in self._offers])
        return f"Player {player}, Utilities: {util_str}, Offers: {offers_str}, Turn: {self._turn}"


    def observation_string(self, player: int = None) -> str:
        """return public observation string (player arg optional)"""
        if self.is_terminal():
            return "terminal"

        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return "chance"

        if player is None:
            player = self.current_player()

        offers_str = "; ".join([str(o) for o in self._offers])
        return f"Pool: {self._pool}, Offers: {offers_str}, Agreement: {self._agreement}"









    def information_state_tensor(self, player: int) -> np.ndarray:
        """Vectorized information state for CFR."""
        #private utilities
        util_vec = np.array(self._utilities[player] if self._utilities[player] else [0]*self._num_item_types)

        #last offer
        last_offer = self._offers[-1] if self._offers else [0]*self._num_item_types
        last_offer = np.array(last_offer)

        #turn (normalized)
        turn_scalar = np.array([self._turn / self._max_turns])

        # agreement flag
        agreement_scalar = np.array([1.0 if self._agreement else 0.0])

        return np.concatenate([util_vec, last_offer, turn_scalar, agreement_scalar])


    def observation_tensor(self, player: int) -> np.ndarray:
        """Vectorized public observation."""
        #last offer
        last_offer = self._offers[-1] if self._offers else [0]*self._num_item_types
        last_offer = np.array(last_offer)

        #turn (normalized)
        turn_scalar = np.array([self._turn / self._max_turns])

        #agreement flag
        agreement_scalar = np.array([1.0 if self._agreement else 0.0])

        return np.concatenate([last_offer, turn_scalar, agreement_scalar])







    def _accept_action_id(self) -> int:
        return 0



    def _encode_offer_action(self, offer: List[int]) -> int:
        """encode offer into integer id > 0 """
        base = self._items_per_type + 1
        idx = 0
        for pos, count in enumerate(offer):
            idx += count * (base ** pos)
        return idx + 1  # +1 to reserve 0 for accept

    def _decode_offer_action(self, action: int) -> List[int]:
        """decode integer id back into offer vector"""
        action -= 1  # shift back since 0 is accept
        base = self._items_per_type + 1
        offer = []
        for _ in range(self._num_item_types):
            offer.append(action % base)
            action //= base
        return offer
    

    def _encode_utility_action(self, util: List[int]) -> int:
        """encode utility vector into int"""
        base = self._utility_max + 1
        idx = 0
        for pos, val in enumerate(util):
            idx += val * (base ** pos)
        return idx

    def _decode_utility_action(self, action: int) -> List[int]:
        """decode int back into utility vector"""
        base = self._utility_max + 1
        util = []
        for _ in range(self._num_item_types):
            util.append(action % base)
            action //= base
        return util
    




    def _all_offers(self) -> List[List[int]]:
        "generate every combination like [0,0,0], [1,0,0], ... [2,2,2]..."
        base = self._items_per_type + 1
        offers = []
        for idx in range(base ** self._num_item_types):
            offer = []
            tmp = idx
            for _ in range(self._num_item_types):
                offer.append(tmp % base)
                tmp //= base
            offers.append(offer)
        return offers




    def _all_utility_vectors(self) -> List[List[int]]:
        base = self._utility_max + 1
        utils = []
        for idx in range(base ** self._num_item_types):
            util = []
            tmp = idx
            for _ in range(self._num_item_types):
                util.append(tmp % base)
                tmp //= base
            utils.append(util)
        return utils















##### 2 DEFINE GAME


class DealOrNoDealGame(pyspiel.Game):
    '''deal or no deal'''

    def __init__(self, params = None) -> None:
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
        game_parameters = self.get_parameters()
        self.max_turns = int(game_parameters.get("max_turns", _MAX_TURNS))
        self.num_item_types = int(game_parameters.get("num_item_types", _NUM_ITEM_TYPES))
        self.items_per_type = int(game_parameters.get("items_per_type", _NUM_ITEMS_PER_TYPE))
        self.utility_max = int(game_parameters.get("utility_max", _MAX_SINGLE_ITEM_VALUE))

    def new_initial_state(self) -> DealOrNoDealState:
        return DealOrNoDealState(self)



    def _generate_allocations(self) -> None:
        '''generate starting allocations for both players'''
        pass




pyspiel.register_game(_GAME_TYPE, DealOrNoDealGame)