import random
from dataclasses import dataclass
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

no_min_action = ValuedAction(None, 1)
no_max_action = ValuedAction(None, -1)

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

def min_right(a, b, key):
    "Min, but prefers the right element under equality"
    return a if key(a) < key(b) else b

def max_right(a, b, key):
    "Max, but prefers the right element under equality"
    return a if key(a) > key(b) else b

class AlphaBeta(HueristicPolicy):

    def min_action(self, state: BoardState, depth: int, alpha: ValuedAction, beta: ValuedAction):
        if state.is_termination_state():
            return ValuedAction(None, 1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 1)
        for a in self.actions(state, 1):
            beta = min_right(beta, ValuedAction(a,
                self.max_action(game.next_state(state, a, 1),
                depth+1, alpha, beta)[1]), get_value)
            if alpha.value > beta.value:
                return alpha
            if alpha.value == beta.value:
                return beta
        return beta

    def max_action(self, state: BoardState, depth: int, alpha: ValuedAction, beta:ValuedAction):
        if state.is_termination_state():
            return ValuedAction(None, -1)
        if depth >= self.depth:
            return self.pick_leaf_action(state, 0)
        for a in self.actions(state, 0):
            alpha = max_right(alpha, ValuedAction(a,
                self.min_action(game.next_state(state, a, 0),
                depth+1, alpha, beta)[1]), get_value)
            if alpha.value > beta.value:
                return beta
            if alpha.value == beta.value:
                return alpha
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

@dataclass
class Edge:
    action: Action
    q: float
    n: int

class Node(NamedTuple):
    counts: int
    edges: List[Edge]
    parents: Set[Tuple[int, EncState]]

def get_ucb(parent_n: int, e: Edge):
    return (e.q / e.n) + math.sqrt(2) * math.sqrt(math.log(parent_n) / e.n)


# TODO: if a child in the tree has a Q value of 1, just take it. No upper bound required,
# as that's the best we can do. Under this policy, the parent Q value should also be set to
# 1, as this is the average (only) reward possible. Never rollout a leaf with Q value 1,
# as we won't be able to learn anything new. If the root node has Q value 1, stop rollouts.

# TODO: test MCTS
# TODO: cache gc
# TODO: make rollouts use AlphaBeta internally, with our cache as hueristic

swapped_locs = np.concatenate([np.arange(6, 12), np.arange(6)])
mean_pos = np.array([3, 0])
flip = np.array([-1, 1])

def inverse_perm(p):
    "Invert the token permutation"
    s = np.empty_like(p, shape=6)
    s[p] = np.arange(5)
    s[5] = 5 # the ball position is unique
    return s

TokenPerm = np.ndarray
Flip = str
Composition = List
GroupElt = Union[TokenPerm, Flip, Composition]

def flip_coord(state):
    return game.encode(flip * (game.decode(state) - mean_pos) + mean_pos)

def apply_op(action: Action, op: GroupElt) -> Action:
    if isinstance(op, TokenPerm):
        return (op[action[0]], action[1])
    if op == "Flip":
        return (action[0], flip_coord(action[1]))
    if isinstance(op, Composition):
        return reduce(apply_op, reversed(op), action)
    raise ValueError("Unknown group element")

def normalized_view(state: EncState, player: PlayerIx) -> tuple[GroupElt, EncState]:
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
    def __init__(self, player: PlayerIx, limit : int =50):
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

    def walk_dag(self, state: BoardState, player: PlayerIx):
        parent_key: Optional[tuple[int, EncState]] = None
        seen : Set[EncState] = set()
        while True:
            unmap, statekey = normalized_view(state.state, player)
            # print("At state", statekey)
            if statekey in seen:
                return
            else:
                seen.add(statekey)
            if statekey in self.cache:
                if parent_key is not None:
                    self.cache[statekey].parents.add(parent_key)
                n = self.cache[statekey].counts
                choice = max(enumerate(self.cache[statekey].edges), key=lambda x: get_ucb(n, x[1]))
                new_state = game.next_state(state, apply_op(choice[1].action, unmap), player ^ 1)
                if new_state.is_termination_state():
                    return
                state = new_state
                parent_key = (choice[0], statekey)
                player ^= 1
            else:
                edges = [self.rollout(state, a, player) for a in self.actions(state, player)]
                total_q = sum(e.q for e in edges)
                parents = set([parent_key]) if parent_key is not None else set()
                self.cache[statekey] = Node(1, edges, parents)
                self.backprop(statekey, total_q, len(edges))
                return

    def backprop(self, statekey: EncState, q: float, n: int):
        print("Backprop from", statekey)
        node = self.cache[statekey]
        to_process : Queue[tuple[int, EncState]] = Queue()
        for k in node.parents:
            to_process.put(k)
        while not to_process.empty():
            i, p = to_process.get()
            self.cache[p].edges[i].q += q
            self.cache[p].edges[i].n += n
            for k in self.cache[p].parents:
                to_process.put(k)
                
    def rollout(self, state, action, player):
        sim = GameSimulator([RandPolicy(0), RandPolicy(1)],
            n_steps=300, validate=False)
        sim.current_round = player
        sim.game_state = game.next_state(state, action, player)
        winner = sim.go()
        if winner == player:
            q = 1
        elif winner is None:
            q = 0
        else:
            q = -1
        return Edge(action, q, 1)
            
    def lookup_action(self, state, a):
        map_action, statekey = normalized_view(state, self.player)
        cached_val = self.cache[statekey]
        return ValuedAction(map_action(cached_val.action), cached_val.q / cached_val.n)

    def policy(self, state):
        for _ in range(self.limit):
            # print("\nStarting Traversal")
            self.walk_dag(state, self.player)
        map_action, statekey = normalized_view(state.state, self.player)
        cached_val = self.cache[statekey]
        choice = max(cached_val.edges, key=lambda x: x.q)
        return ValuedAction(choice.action, choice.q)
