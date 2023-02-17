import random
from typeguard import typechecked
from game import Action, BoardState, PlayerIx
import game
import math
from typing import NamedTuple, Optional, List

class ValuedAction(NamedTuple):
    action: Optional[Action]
    value: float

def get_value(va: ValuedAction):
    return va.value

class Policy:
    def __init__(self, player: PlayerIx, depth=3):
        self.player = player
        self.depth = depth

    def actions(self, state: BoardState, player: PlayerIx) -> List[Action]:
        return list(game.generate_valid_actions(state, player))

class Passive:
    def actions(self, state: BoardState, player: PlayerIx) -> List[Action]:
        actions = list(game.generate_valid_actions(state, player, False))
        random.shuffle(actions)
        return actions

class HueristicPolicy(Policy):
    def actions(self, state, player):
        actions = super().actions(state, player)
        random.shuffle(actions)
        return actions

    def pick_leaf_action(self, state: BoardState, player: PlayerIx) -> ValuedAction:
        return ValuedAction(None, 0)

no_min_action = ValuedAction(None, math.inf)
no_max_action = ValuedAction(None, -math.inf)

class MinimaxPolicy(HueristicPolicy):

    # The minimizing player is 1
    def min_action(self, state: BoardState, depth: int):
        if state.is_termination_state():
            return ValuedAction(None, 1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 1)
        return min((ValuedAction(a,
            self.max_action(game.next_state(state, a, 1),
            depth+1)[1]) for a in 
            self.actions(state, 1)), default=no_min_action, key=get_value)

    # The maximizing player is 0
    def max_action(self, state: BoardState, depth: int):
        if state.is_termination_state():
            return ValuedAction(None, 1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 0)
        return max((ValuedAction(a,
            self.min_action(game.next_state(state, a, 0),
            depth+1)[1]) for a in 
            self.actions(state, 0)), default=no_max_action, key=get_value)

    def policy(self, state):
        if self.player == 0:
            return self.max_action(state, 1)
        else:
            return self.min_action(state, 1)

class AlphaBeta(HueristicPolicy):

    def min_action(self, state: BoardState, depth: int, alpha: ValuedAction, beta: ValuedAction):
        if state.is_termination_state():
            return ValuedAction(None, 1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 1)
        for a in self.actions(state, 1):
            beta = min(beta, ValuedAction(a,
                self.max_action(game.next_state(state, a, 1),
                depth+1, alpha, beta)[1]), key=get_value)
            if alpha.value >= beta.value:
                return alpha
        return beta

    def max_action(self, state: BoardState, depth: int, alpha: ValuedAction, beta:ValuedAction):
        if state.is_termination_state():
            return ValuedAction(None, -1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 0)
        for a in self.actions(state, 0):
            alpha = max(alpha, ValuedAction(a,
                self.min_action(game.next_state(state, a, 0),
                depth+1, alpha, beta)[1]), key=get_value)
            if alpha.value >= beta.value:
                return beta
        return alpha

    def policy(self, state):
        if self.player == 0:
            return self.max_action(state, 1, no_max_action, no_min_action)
        else:
            return self.min_action(state, 1, no_max_action, no_min_action)

class PassiveAlphaBeta(Passive, AlphaBeta):
    pass

class PassiveMinimax(Passive, MinimaxPolicy):
    pass

class RandPolicy(Policy):
    def policy(self, state):
        actions = self.actions(state, self.player)
        chosen = random.randrange(len(actions))
        return (actions[chosen], 0)

class Edge(NamedTuple):
    action: Optional[Action]
    q: float
    n: int
    parent_n: int

def get_ucb(e: Edge):
    return e.q + math.sqrt(2) * math.sqrt(math.log(e.parent_n), e.n)

class MCTS(HueristicPolicy):
    def __init__(self, player, limit=500):
        self.player = player
        self.cache = dict()
        self.limit = limit

    # The minimizing player is 1
    def min_action(self, state: BoardState):
        statekey = tuple(state.state) # TODO: remove symmetries
        if state.is_termination_state():
            self.cache[statekey] = Edge(None, 1, 1, 1)
        if statekey in self.cache:
            choice = min(self.cache[statekey], key=get_ucb)
            return self.max_action(game.next_state(state, choice.action, 1))
        else:
            self.rollout(state)

    # The maximizing player is 0
    def max_action(self, state: BoardState):
        statekey = tuple(state.state) # TODO: remove symmetries
        if state.is_termination_state():
            self.cache[statekey] = Edge(None, -1, 1, 1)
        elif statekey in self.cache:
            choice = max(self.cache[statekey], key=get_ucb)
            self.min_action(game.next_state(state, choice.action, 0))
        else:
            self.rollout(state)

    def rollout(self, state):
        pass # TODO

    def lookup_action(self, state, a):
        cached_val = self.cache[game.next_state(state, a)]
        return ValuedAction(cached_val.action, cached_val.q)

    def policy(self, state):
        for _ in range(self.limit):
            if self.player == 0:
                self.max_action(state, 1)
            else:
                self.min_action(state, 1)
        actions = [self.lookup_action(state, a)
            for a in self.actions(state, self.player)]
        if self.player == 0:
            return max(actions, key=get_value)
        else:
            return min(actions, key=get_value)



# For MCTS, we should map to a space without symmetries.
# Perhaps flip the board to the orientation with the highest hash.

# Perhaps we could run MCTS for a long time without clearing its
# cache much. Then train a value function on its output to use as a hueristic
# for AlphaBeta
