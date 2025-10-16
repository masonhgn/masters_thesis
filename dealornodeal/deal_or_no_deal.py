import pyspiel
from typing import List, Dict, Tuple
import numpy as np


_NUM_PLAYERS = 2
_MAX_TURNS = 3
_NUM_ITEM_TYPES = 2
_NUM_ITEMS_PER_TYPE = 1 #quantities of items
_MAX_SINGLE_ITEM_VALUE = 1






_NUM_DISTINCT_ACTIONS = 1 + (_NUM_ITEMS_PER_TYPE + 1) ** _NUM_ITEM_TYPES #accept + all possible offers
_MAX_CHANCE_NODE_OUTCOMES = (_MAX_SINGLE_ITEM_VALUE + 1) ** _NUM_ITEM_TYPES 


_GAME_TYPE = pyspiel.GameType(
    short_name="python_deal_or_no_deal",
    long_name="Python Deal or No Deal Negotiation",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
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
    min_utility = -_MAX_SINGLE_ITEM_VALUE * _NUM_ITEM_TYPES * _NUM_ITEMS_PER_TYPE,
    max_utility =  _MAX_SINGLE_ITEM_VALUE * _NUM_ITEM_TYPES * _NUM_ITEMS_PER_TYPE,
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



    def __str__(self) -> str:
        """for internal use by open_spiel"""
        if self.is_terminal():
            return "terminal"
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            return "chance"
        return self.information_state_string(self.current_player())



    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if any(u is None for u in self._utilities):
            return pyspiel.PlayerId.CHANCE
        if self._cur_player == pyspiel.PlayerId.CHANCE:
            return 0
        return self._cur_player
    



    def is_terminal(self) -> bool:
        return self._agreement or self._turn >= self._max_turns


    def returns(self) -> List[float]:
        """compute returns based on last valid offer"""
        if not self.is_terminal():
            return [0.0, 0.0]
        if not self._agreement or len(self._offers) == 0:
            return [0.0, 0.0]

        last_offer = self._offers[-1]
        total_allocation = [self._items_per_type] * self._num_item_types

        #identify proposer (previous player, since _cur_player switched)
        proposer = (self._cur_player + 1) % self._num_players

        #each player's subjective utility of the deal
        offer_util_p0 = np.dot(self._utilities[0], last_offer)
        remainder_util_p0 = np.dot(self._utilities[0],
                                [t - o for t, o in zip(total_allocation, last_offer)])
        offer_util_p1 = np.dot(self._utilities[1], last_offer)
        remainder_util_p1 = np.dot(self._utilities[1],
                                [t - o for t, o in zip(total_allocation, last_offer)])

        #values for proposer and responder
        value_p0 = offer_util_p0 if proposer == 0 else remainder_util_p0
        value_p1 = offer_util_p1 if proposer == 1 else remainder_util_p1

        #convert to zero-sum: relative advantage
        diff = value_p0 - value_p1

        #normalization to keep within declared range
        max_abs = _MAX_SINGLE_ITEM_VALUE * self._num_item_types * self._items_per_type
        diff = np.clip(diff, -max_abs, max_abs)

        return [float(diff), float(-diff)]






    def _apply_action(self, action: int) -> None:
        if self.current_player() == pyspiel.PlayerId.CHANCE:
            next_pid = [i for i, u in enumerate(self._utilities) if u is None][0]
            self._utilities[next_pid] = self._decode_utility_action(action)
            if any(u is None for u in self._utilities):
                self._cur_player = pyspiel.PlayerId.CHANCE
            else:
                #non deterministic behavior breaks EFR
                #self._cur_player = np.random.choice([0, 1]) #choose random player to start

                self._cur_player = 0
            return

        #have we reached an agreement? 
        if action == self._accept_action_id():
            self._agreement = True
            # don't increment turn here!!!!!
        else:
            offer = self._decode_offer_action(action)
            self._offers.append(offer)
            self._cur_player = 1 - self._cur_player
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


    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an observer for this game (OpenSpiel compatibility)."""
        return DealOrNoDealObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params
        )












class DealOrNoDealObserver:
    """
    this class is an "observer" that open_spiel expects any imperfect-information game to provide.
    its job is to give a consistent view of the game state from a given player's perspective,
    both as a human-readable string and as a numerical vector (tensor).

    in open_spiel, algorithms like CFR, EFR, or NashConv don't interact directly with your
    DealOrNoDealState internals. instead, they ask the game for an observer object and use it
    to extract what a given player can "see" at each point in the game.

    so this class serves as the adapter layer between your custom state representation and
    the standard open_spiel observation API.
    """

    def __init__(self, iig_obs_type, params):
        """
        constructor for the observer. called by your game's make_py_observer() method.

        iig_obs_type:  an object describing what kind of information this observer should include.
                       open_spiel passes this automatically when it needs to build an observer.
                       for example, it may specify whether perfect recall or private/public info
                       should be tracked.

        params: optional additional configuration. we don't need any, so we just raise an error
                if someone tries to pass them in.
        """
        if params:
            raise ValueError(f"observation parameters not supported; got {params}")

        # store what kind of info we should include in this observation
        # these flags will control what data we encode later.
        self.perfect_recall = iig_obs_type.perfect_recall
        self.include_private = (
            iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER
        )
        self.include_public = iig_obs_type.public_info

        # we now define what pieces of data will make up our observation.
        # each entry in "pieces" is a tuple of (name, number_of_elements, shape).
        # these will later be concatenated into a single flat tensor.
        pieces = []

        # 1. always include which player's perspective this observation is from.
        #    this is encoded as a one-hot vector, e.g. [1,0] for player 0 and [0,1] for player 1.
        pieces.append(("player", 2, (2,)))

        # 2. include private information (the player’s internal utility vector)
        #    only if the observer is configured to include private info.
        #    this is relevant because in imperfect-information games, only each player
        #    knows their own utility structure, not the opponent's.
        if self.include_private:
            # here we assume 2 item types. later you could generalize by passing
            # game.num_item_types instead.
            pieces.append(("utilities", 2, (2,)))

        # 3. include public information visible to both players.
        #    this includes the most recent offer, the current turn number,
        #    and whether an agreement has been reached.
        if self.include_public:
            pieces.append(("last_offer", 2, (2,)))
            pieces.append(("turn", 1, (1,)))
            pieces.append(("agreement", 1, (1,)))

        # compute total number of elements in the flattened observation tensor
        total_size = sum(size for _, size, _ in pieces)

        # create one flat tensor to hold all components.
        # open_spiel expects observers to expose a 1D float32 vector of fixed length.
        self.tensor = np.zeros(total_size, np.float32)

        # now build a dictionary of named views into this flat tensor.
        # this lets us conveniently access each piece by name (e.g. self.dict["turn"])
        # while keeping the flat layout that open_spiel expects.
        self.dict = {}
        idx = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[idx:idx + size].reshape(shape)
            idx += size

    def set_from(self, state, player):
        """
        this method updates the internal tensor representation to reflect
        what the given player sees in the provided game state.

        this is called by open_spiel whenever it needs to get a new observation
        (for example, when traversing the game tree or computing exploitability).

        inputs:
            state: a DealOrNoDealState instance (the full game state)
            player: the integer id (0 or 1) of the player whose perspective we’re encoding

        the output of this function is stored internally in self.tensor and self.dict.
        nothing is returned — other methods will later access these updated values.
        """
        # first, reset the tensor to zeros so we can write fresh data
        self.tensor.fill(0)

        # one-hot encode which player this observation belongs to
        self.dict["player"][player] = 1.0

        # if private info is included and the player already has a defined utility vector,
        # store that in the utilities part of the tensor.
        # this represents the player's "private knowledge" — their internal utility weights.
        if self.include_private and state._utilities[player] is not None:
            util_vec = np.array(state._utilities[player], dtype=np.float32)
            self.dict["utilities"][:len(util_vec)] = util_vec

        # now we encode public information, which both players can see.
        if self.include_public:
            # if there have been any offers, take the most recent one.
            if len(state._offers) > 0:
                last_offer = np.array(state._offers[-1], dtype=np.float32)
            else:
                # otherwise fill with zeros (no offers yet)
                last_offer = np.zeros(len(state._pool), dtype=np.float32)
            self.dict["last_offer"][:len(last_offer)] = last_offer

            # normalize the turn count to a 0–1 range, so it’s numerically stable
            self.dict["turn"][0] = state._turn / state._max_turns

            # binary indicator of whether an agreement has been reached
            self.dict["agreement"][0] = 1.0 if state._agreement else 0.0

    def string_from(self, state, player):
        """Return exactly the same information state string used by the game."""
        return state.information_state_string(player)



    def tensor_from(self, state, player):
        """
        this is the numerical counterpart of string_from().
        it returns a numpy vector encoding the same information numerically.

        open_spiel’s algorithms call this when they want the tensor version
        of the information state (e.g. for neural policy approximators).

        internally, it just calls set_from() to update self.tensor,
        and then returns a copy of that tensor as a numpy array.
        """
        # update the tensor with the current state information
        self.set_from(state, player)
        # return a copy so outside code doesn't accidentally mutate internal data
        return np.array(self.tensor, copy=True)











pyspiel.register_game(_GAME_TYPE, DealOrNoDealGame)