import random
from functools import reduce
import numpy as np
from game import Action, BoardState, PlayerIx, GameSimulator, EncState
import game
import math
from typing import NamedTuple, Optional, List, Dict, Set, Tuple, Union
from queue import Queue

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
    parents: Set[Tuple[int, EncState]]

def get_ucb(parent_n: int, e: Edge):
    return (e.q / e.n) + math.sqrt(2) * math.sqrt(math.log(parent_n) / e.n)

# TODO: test MCTS
# TODO: cache gc
# TODO: make rollouts use AlphaBeta internally, with our cache as hueristic

swapped_locs = np.concatenate([np.arange(6, 12), np.arange(6)])
mean_pos = np.array([3, 0])
flip = np.array([-1, 1])

def inverse_perm(p):
    "Invert the permutation"
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

TokenPerm = np.ndarray
Flip = str
Composition = List["GroupElt"]
GroupElt = Union[TokenPerm, Flip, Composition]

def flip_coord(state):
    return game.encode(flip * (game.decode(state) - mean_pos) + mean_pos)

def apply_op(op, action):
    if op == "Flip":
        return (action[0], flip_coord(action[1]))
    if isinstance(op, TokenPerm):
        return (op[action[0]], action[1])
    if isinstance(op, Composition):
        return reduce(apply_op, reversed(op), initial=action)

def normalized_view(state: EncState, player: PlayerIx) -> (GroupElt, EncState):
    """
    Transforms the state to a normalized one.
    Returns a way to map actions from the fictional state back to the real one.
    """
    action_map = [] # sequence of permutations, applied right to left. 

    # Swap players so that the current player is 0. Action map is unchanged. 
    state = state if player == 0 else state[swapped_locs]

    # Flip the board left or right
    state2 = game.encode(flip * (game.decode(state) - mean_pos) + mean_pos)
    if hash(tuple(state2)) > hash(tuple(state)):
        state = state2
        action_map.append("Flip")

    # Sort the tokens of each player
    ix1 = np.argsort(state[BoardState.locs1])
    ix2 = 6 + np.argsort(state[BoardState.locs2])
    sort_perm = np.concatenate([ix1, [5], ix2, [11]])
    state = state[sort_perm]
    action_map.append(inverse_perm(ix1))
   
    return action_map, tuple(state)

class MCTS(Policy):
    def __init__(self, player, limit=500):
        self.player = player
        self.cache : Dict[EncState, Node] = dict()
        self.limit = limit

    def gc(self):
        pass
        # TODO: Keep a priority queue of all the states in my cache
        # If our cache goes beyond a specified size, delete the min priority one
        # A hueristic for cache eviction can be least recently used. This means every
        # time we encounter a state during walk_dag, we must change the priority in the queue.
        # To prevent a large number of things having the same priority and making the queue unbalanced,
        # we could add some random noise to the keys.

    def walk_dag(self, state: BoardState, player: PlayerIx, parent_key: tuple[int, EncState]):
        while True:
            unmap, statekey = normalized_view(state.state, player)
            if parent_key is not None:
                self.cache[statekey].parents.add(parent_key)
            if statekey in self.cache:
                n = self.cache[statekey].counts
                choice = max(enumerate(self.cache[statekey].edges), key=lambda x: get_ucb(n, x[1]))
                new_state = game.next_state(state, apply_op(unmap, choice[1].action), player ^ 1)
                if new_state.is_termination_state():
                    return
                state = new_state
                parent_key = (choice[0], statekey)
                player ^= 1
            else:
                self.cache[statekey] = Node(1, [self.rollout(state, a, player) for a in self.actions(state, player)],
                    set([parent_key]))
                self.backprop(statekey)
                return

    def backprop(self, statekey: EncState):
        node = self.cache[statekey]
        q = node.q
        to_process = Queue()
        for k in node.parents:
            to_process.put(k)
        while not to_process.empty():
            (i, p) = to_process.get()
            self.cache[p].edges[i].q += q
            self.cache[p].edges[i].n += 1
            for k in self.cache[p].parents:
                to_process.put(k)
                
    def rollout(self, state, action, player):
        sim = GameSimulator([RandPolicy(0), RandPolicy(1)])
        sim.current_round = player
        sim.game_state = game.next_state(state, action, player)
        winner = sim.go()
        return Edge(action, 1 if winner == player else -1, 1)
            
    def lookup_action(self, state, a):
        map_action, statekey = max_view(game.next_state(state, a))
        cached_val = self.cache[statekey]
        return ValuedAction(map_action(cached_val.action), cached_val.q / cached_val.n)

    def policy(self, state):
        for _ in range(self.limit):
            self.walk_dag(state, self.player)
        actions = [self.lookup_action(state, a)
            for a in self.actions(state, self.player)]
        return max(actions, key=get_value)

