
'''
TODO:
Normalization Consistency

Utilities are divided by 10.0, pool counts and allocations by items_per_type.
This is good, but just to be clear:

If items_per_type changes, the relative scale of pool vs. utilities may differ.
This isnt wrong, but it can bias CFR learning.
Consider scaling everything into [0,1] more consistently, e.g.
divide utils by their max possible total (like 10), and pool counts also by items_per_type.


'''






import itertools
import numpy as np
import pyspiel
from open_spiel.python.observation import IIGObserverForPublicInfoGame

_NUM_PLAYERS = 2
_DEFAULT_MAX_TURNS = 10
_DEFAULT_ITEM_TYPES = 3
_DEFAULT_ITEMS_PER_TYPE = 2


_GAME_TYPE = pyspiel.GameType(
    short_name="python_deal_or_no_deal",
    long_name="Python Deal or No Deal Negotiation",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
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
        "max_turns": _DEFAULT_MAX_TURNS,
        "num_item_types": _DEFAULT_ITEM_TYPES,
        "items_per_type": _DEFAULT_ITEMS_PER_TYPE,
        "discount":1.0
    },

)

# conservative bounds
_MAX_UTILITY = 20.0
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=0,  # placeholder, updated later
    max_chance_outcomes=0,
    num_players=_NUM_PLAYERS,
    min_utility=0.0,
    max_utility=_MAX_UTILITY,
    utility_sum=None,
    max_game_length=_DEFAULT_MAX_TURNS,
)


class DealOrNoDealGame(pyspiel.Game):
    """Deal or No Deal game definition."""

    def __init__(self, params=None):
        params = params or {}
        
        # Extract params
        self.max_turns = int(params.get("max_turns", _DEFAULT_MAX_TURNS))
        self.num_item_types = int(params.get("num_item_types", _DEFAULT_ITEM_TYPES))
        self.items_per_type = int(params.get("items_per_type", _DEFAULT_ITEMS_PER_TYPE))
        self.discount = float(params.get("discount", 1.0))

        # Generate allocations and instances FIRST
        self._all_allocations = self._generate_allocations()
        self._accept_action_id = len(self._all_allocations)
        self._instances = self._generate_instances()

        self._obs_tensor_size = (
            1 +
            (self.max_turns + 1) +
            self.num_item_types +
            self.num_item_types +
            self.num_item_types
        )

        game_info = pyspiel.GameInfo(
            num_distinct_actions=len(self._all_allocations) + 1,
            max_chance_outcomes=len(self._instances),
            num_players=_NUM_PLAYERS,
            min_utility=0.0,
            max_utility=_MAX_UTILITY,
            utility_sum=None,
            max_game_length=self.max_turns,
        )

        # NOW call super after building _instances
        super().__init__(_GAME_TYPE, game_info, params)
        self._game_info = game_info
                

    def get_game_info(self):
        return self._game_info
    

    def observation_tensor_shape(self):
        return [self._obs_tensor_size]

    def information_state_tensor_shape(self):
        return [self._obs_tensor_size]



    def _generate_instances(self):
        pool = [self.items_per_type] * self.num_item_types
        utils = []
        # All utility vectors summing to 10
        for u in itertools.product(range(6), repeat=self.num_item_types):
            if sum(u) == 10:
                utils.append(list(u))
        instances = []
        for u0 in utils:
            for u1 in utils:
                instances.append((pool, u0, u1))
        return instances

    def _generate_allocations(self):
        pool = [self.items_per_type] * self.num_item_types
        allocations = []
        for alloc in itertools.product(*[range(c + 1) for c in pool]):
            allocations.append(list(alloc))
        return allocations

    def new_initial_state(self):
        return DealOrNoDealState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return IIGObserverForPublicInfoGame(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), params
        )


    def max_chance_nodes_in_history(self):
        return 1  # Only one chance event: assigning utilities









class DealOrNoDealState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._game = game
        self._cur_player = pyspiel.PlayerId.CHANCE
        self._turn = 0
        self._history = []
        self._agreement = None
        self._instance = None  # assigned by chance
        self._is_terminal = False
        self._discount_factor = 1.0 



    def _clone(self):
        """Creates a deep copy of the state (required for CFR/EFR)."""
        new_state = DealOrNoDealState(self._game)
        new_state._cur_player = self._cur_player
        new_state._turn = self._turn
        new_state._history = list(self._history)
        new_state._agreement = self._agreement
        new_state._instance = self._instance
        new_state._is_terminal = self._is_terminal
        new_state._discount_factor = self._discount_factor

        # Ensure _instances is available in the cloned state's game
        if not hasattr(new_state._game, "_instances"):
            new_state._game._instances = self._game._instances
        if not hasattr(new_state._game, "_all_allocations"):
            new_state._game._all_allocations = self._game._all_allocations
        if not hasattr(new_state._game, "_accept_action_id"):
            new_state._game._accept_action_id = self._game._accept_action_id

        return new_state




    def information_state_tensor(self, player):
        tensor = np.zeros(self._game._obs_tensor_size, dtype=np.float32)
        offset = 0

        # Agreement reached
        tensor[offset] = 1.0 if self._agreement is not None else 0.0
        offset += 1

        # Turn number one-hot
        tensor[offset + self._turn] = 1.0
        offset += (self._game.max_turns + 1)

        # My utilities (scaled 0–1)
        if self._instance is not None:
            utils = self._instance[1] if player == 0 else self._instance[2]
            for i, u in enumerate(utils):
                tensor[offset + i] = u / 10.0
        offset += self._game.num_item_types

        # Pool composition (raw counts scaled)
        pool = self._instance[0] if self._instance is not None else [0]*self._game.num_item_types
        for i, c in enumerate(pool):
            tensor[offset + i] = c / self._game.items_per_type
        offset += self._game.num_item_types

        # Last offer (if any)
        if self._history:
            alloc = self._game._all_allocations[self._history[-1][1]]
            for i, a in enumerate(alloc):
                tensor[offset + i] = a / self._game.items_per_type
        # else already zero
        offset += self._game.num_item_types

        return tensor





    def observation_tensor(self, player):
        tensor = np.zeros(self._game._obs_tensor_size, dtype=np.float32)
        offset = 0

        # Agreement reached
        tensor[offset] = 1.0 if self._agreement is not None else 0.0
        offset += 1

        # Turn number one-hot
        tensor[offset + self._turn] = 1.0
        offset += (self._game.max_turns + 1)

        # My utilities (scaled 0–1)
        if self._instance is not None:
            utils = self._instance[1] if player == 0 else self._instance[2]
            for i, u in enumerate(utils):
                tensor[offset + i] = u / 10.0
        offset += self._game.num_item_types

        # Pool composition (raw counts scaled)
        pool = self._instance[0] if self._instance is not None else [0]*self._game.num_item_types
        for i, c in enumerate(pool):
            tensor[offset + i] = c / self._game.items_per_type
        offset += self._game.num_item_types

        # Last offer (if any)
        if self._history:
            alloc = self._game._all_allocations[self._history[-1][1]]
            for i, a in enumerate(alloc):
                tensor[offset + i] = a / self._game.items_per_type
        # else already zero
        offset += self._game.num_item_types

        return tensor






    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        if self._instance is None:
            return pyspiel.PlayerId.CHANCE
        return self._cur_player

    def chance_outcomes(self):
        assert self.current_player() == pyspiel.PlayerId.CHANCE
        prob = 1.0 / len(self._game._instances)
        return [(idx, prob) for idx in range(len(self._game._instances))]

    def _apply_action(self, action):
        if self.is_chance_node():
            self._instance = self._game._instances[action]
            self._cur_player = 0
            return
        if action == self._game._accept_action_id:
            if not self._history:
                raise ValueError("Cannot accept without an offer.")
            self._agreement = self._history[-1][1]
            self._is_terminal = True
            return
        
        if action < 0 or action >= len(self._game._all_allocations):
            raise ValueError(f"Invalid action: {action}")        

        self._history.append((self._cur_player, action))

        # Apply discount after a proposal
        self._discount_factor *= self._game.discount 

        self._turn += 1
        if self._turn >= self._game.max_turns:
            self._is_terminal = True
        else:
            self._cur_player = 1 - self._cur_player


    def is_terminal(self):
        return self._is_terminal


    def is_chance_node(self):
        return self.current_player() == pyspiel.PlayerId.CHANCE

    def returns(self):
        if not self._is_terminal or self._agreement is None:
            return [0.0, 0.0]
        pool, utils0, utils1 = self._instance
        alloc = self._game._all_allocations[self._agreement]
        r0 = sum(a * u for a, u in zip(alloc, utils0))
        r1 = sum((p - a) * u for a, p, u in zip(alloc, pool, utils1))
        r0 *= self._discount_factor
        r1 *= self._discount_factor
        return [float(r0), float(r1)]

    def information_state_string(self, player):
        if self._instance is None:
            return "Unassigned utilities"

        pool, utils0, utils1 = self._instance
        util = utils0 if player == 0 else utils1

        info = []
        info.append(f"Turn={self._turn}")
        info.append(f"MyUtils={util}")
        info.append(f"Agreement={'Yes' if self._agreement is not None else 'No'}")

        if self._history:
            info.append("History:")
            for i, (pl, idx) in enumerate(self._history):
                offer = self._game._all_allocations[idx]
                info.append(f"  Round {i+1}: P{pl} offered {offer}")
            if self._agreement is not None:
                offer = self._game._all_allocations[self._agreement]
                info.append(f"  Accepted: {offer}")

        return "\n".join(info)
    


    def _legal_actions(self, player):
        if self._is_terminal or player != self._cur_player:
            return []
        actions = list(range(len(self._game._all_allocations)))
        if self._history:
            actions.append(self._game._accept_action_id)
        return actions

    def _action_to_string(self, player, action):
        if action == self._game._accept_action_id:
            return "Accept"
        alloc = self._game._all_allocations[action]
        return f"Offer {alloc}"




    def __str__(self):
        if self._instance is None:
            return "Initial chance node"
        hist = [
            f"P{pl}={self._game._all_allocations[idx]}"
            for pl, idx in self._history
        ]
        return f"Turn={self._turn}, Agreement={self._agreement}, Offers={hist}"



        

# Register the game
pyspiel.register_game(_GAME_TYPE, DealOrNoDealGame)
