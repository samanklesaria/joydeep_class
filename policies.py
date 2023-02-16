import random
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

class PassivePolicy(Policy):
    def actions(self, state: BoardState, player: PlayerIx) -> List[Action]:
        return list(game.generate_valid_actions(state, player, False))

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

    def min_action(self, state: BoardState, depth: int, alpha, beta):
        if state.is_termination_state():
            return ValuedAction(None, 1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 1)
        for a in self.actions(state, 1):
            beta = min(beta, ValuedAction(a,
                self.max_action(game.next_state(state, a, 1),
                depth+1, alpha, beta)[1]), key=get_value)
            if alpha >= beta:
                return alpha
        return beta

    def max_action(self, state: BoardState, depth: int, alpha, beta):
        if state.is_termination_state():
            return ValuedAction(None, -1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 0)
        for a in self.actions(state, 0):
            alpha = max(alpha, ValuedAction(a,
                self.min_action(game.next_state(state, a, 0),
                depth+1, alpha, beta)[1]), key=get_value)
            if alpha >= beta:
                return beta
        return alpha

    def policy(self, state):
        if self.player == 0:
            return self.max_action(state, 1, -math.inf, math.inf)
        else:
            return self.min_action(state, 1, -math.inf, math.inf)


class RandPolicy(Policy):
    def policy(self, state):
        actions = self.actions(state, self.player)
        chosen = random.randrange(len(actions))
        return (actions[chosen], 0)


