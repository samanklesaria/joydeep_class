import random
from typeguard import typechecked
from game import Action, BoardState, PlayerIx, GameSimulator, EncState
import game
import math
from typing import NamedTuple, Optional, List, Dict, Set

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

class Node(NamedTuple):
    counts: int
    edges: List[Edge]
    parents: Set[EncState]

def get_ucb(parent_n: int, e: Edge):
    return e.q + math.sqrt(2) * math.sqrt(math.log(parent_n) / e.n)

# TODO: cache gc
# TODO: make rollouts use AlphaBeta internally, with our cache as hueristic
# TODO: implement max_view and backprop

def max_view(state: EncState, player) -> EncState:
    """
    Transforms the state to the one in which player is player 0.
    Also returns a way to map actions from the fictional state back to the real one.
    """
    # TODO
    return map_action, tuple(state)

class MCTS(Policy):
    def __init__(self, player, limit=500):
        self.player = player
        self.cache : Dict[EncState, Node] = dict()
        self.limit = limit

    def walk_dag(self, state: BoardState, player: PlayerIx, parent_key: EncState):
        map_action, statekey = max_view(state.state, player)
        if parent_key is not None:
            self.cache[statekey].parents.add(parent_key)
        if statekey in self.cache:
            n = self.cache[statekey].counts
            choice = max(enumerate(self.cache[statekey].edges), key=lambda x: get_ucb(n, x[1]))
            new_state = game.next_state(state, map_action(choice[1].action), player ^ 1)
            if not new_state.is_termination_state():
                self.walk_dag(new_state, player ^ 1, statekey)
        else:
            self.cache[statekey] = Node(1, [self.rollout(state, a, 0) for a in self.actions(state, 0)],
                set([parent_key]))
            self.backprop(statekey)

    def backprop(self, state):
        # TODO: update stuff
        pass
                
    def rollout(self, state, action, player):
        sim = GameSimulator([RandPolicy(0), RandPolicy(1)])
        sim.current_round = player
        sim.game_state = game.next_state(state, action, player)
        winner = sim.go()
        return Edge(action, 1 if winner == player else -1, 1)
            
    def lookup_action(self, state, a):
        map_action, statekey = max_view(game.next_state(state, a))
        cached_val = self.cache[statekey]
        return ValuedAction(map_action(cached_val.action), cached_val.q)

    def policy(self, state):
        for _ in range(self.limit):
            self.walk_dag(state, self.player)
        actions = [self.lookup_action(state, a)
            for a in self.actions(state, self.player)]
        return max(actions, key=get_value)

