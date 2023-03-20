import numpy as np
from queue import Queue, PriorityQueue
from game import GameSimulator, PlayerIx, EncState, Action, BoardState, VALIDATE
import game
import typing
from typing import Iterable, Optional, Dict, Tuple, Union, NamedTuple, List
import random
import heapq
import math
from copy import deepcopy

# The state for which the game is Markovian
MarkovState = Tuple[EncState, PlayerIx]

# Back-pointer dictionary
BpDict = Dict[MarkovState, Tuple[Union[MarkovState, None], Union[Action,None]]]

# Planning output
StPath = Iterable[Tuple[MarkovState, Optional[Action]]]

class Problem:

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set


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
        actions = self.actions(state, player)
        return ValuedAction(random.choice(actions), 0)

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

class MCTS(HueristicPolicy):
    class Node:
        def __init__(self, board_state, parent, player_index):
            self.board_state = board_state
            self.q = 0
            self.n = 0
            self.win_count = 0
            self.parent = parent
            self.children= list()
            self.actions = list()
            self.player_index = player_index

    def __init__(self, player: PlayerIx, limit : int =200, rollout_limit = 75):
        super().__init__(None)
        self.player = player
        self.limit = limit
        self.root = None
        self.exploration = np.sqrt(2)
        self.rollout_limit = rollout_limit

    def walk_dag(self, state: BoardState, player: PlayerIx):
        current_node = self.root
        chosen_action = None

        # get to leaf (selection)
        while current_node.actions:
            actions = self.actions(current_node.board_state, current_node.player_index)

            best_score = -10000
            best_action = None
            best_node = None

            for action in actions:
                this_score = 0

                child_node = current_node
                if action in current_node.actions:
                    child_node = current_node.children[current_node.actions.index(action)]
                    if current_node.player_index == self.player:
                        this_score = child_node.q + self.exploration * math.sqrt(math.log(current_node.n)/child_node.n)
                    else:
                        this_score = -1 * child_node.q + self.exploration * math.sqrt(math.log(current_node.n)/child_node.n)
                else:
                    this_score = 10000

                if (this_score > best_score):
                    best_score = this_score
                    best_action = action
                    best_node = child_node
            if (best_node == current_node):
                chosen_action = best_action
                break

            current_node = best_node

        if chosen_action is None:
            actions = self.actions(current_node.board_state, current_node.player_index)
            chosen_action = actions[(int)(len(actions)*random.random())]
        
        next_state = game.next_state(current_node.board_state, chosen_action, current_node.player_index)
        next_node = self.Node(next_state, current_node, current_node.player_index ^ 1)
        current_node.actions.append(chosen_action)
        current_node.children.append(next_node)

        current_node = next_node
        self.rollout(current_node)

    def backprop(self, winner, current_node):
        while current_node:
            current_node.n += 1
            current_node.win_count = current_node.win_count + 1 if winner == self.player else current_node.win_count
            current_node.q = current_node.win_count / current_node.n
            current_node = current_node.parent

    def rollout(self, current_node):
        sim = GameSimulator([RandPolicy(0), RandPolicy(1)],
            n_steps=self.rollout_limit, validate=False, use_heuristic=True)
        sim.current_round = current_node.player_index
        sim.game_state = deepcopy(current_node.board_state)
        #print(str(sim.game_state.state))
        winner = sim.go()
        #print(winner)
        self.backprop(winner, current_node)

    def policy(self, state):
        board_state = BoardState(state)
        self.root = self.Node(board_state, None, self.player)
        
        for _ in range(self.limit):
            # print("\nStarting Traversal")
            self.walk_dag(state, self.player)
        
        best_action = None
        best_value = -10000
        for child, action in zip(self.root.children,self.root.actions):
            #print('---------------')
            #print(child.n)
            #print(child.q)
            if child.q > best_value:
                best_action = action
                best_value = child.q

        return (best_action, best_value)

class GameStateProblem(Problem):

    def __init__(self, initial_board_state: BoardState, goal_board_state: BoardState, player_idx: PlayerIx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        The form of initial state is:
        ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def validate_path(self, path):
        new_st = None
        for (st, action) in path:
            assert new_st == st or new_st is None
            if action is not None:
                new_st = self.execute(st, action)

    def set_search_alg(self, alg="bfs"):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """
        if alg == "astar":
            self.search_alg_fnc = self.astar
        else:
            self.search_alg_fnc = self.bfs

    def get_actions(self, state: MarkovState) -> Iterable[Action]:
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    @typing.no_type_check # mypy can't handle numpy's polymorphic indexing
    def execute(self, state: MarkovState, action: Action) -> MarkovState:
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))


    def bfs(self) -> StPath:
        frontier : Queue[MarkovState] = Queue()
        frontier.put(self.initial_state)
        parent : BpDict = {}
        parent[self.initial_state] = (None, None)
        while not frontier.empty():
            current = frontier.get()
            if current in self.goal_state_set:
                backwards = list(get_path(current, parent, None))
                backwards.reverse()
                if VALIDATE:
                    self.validate_path(backwards)
                return backwards
            for action in self.get_actions(current):
                if VALIDATE:
                    self.sim.validate_action(action, current[1])
                succ = self.execute(current, action)
                if not (succ in parent):
                    if VALIDATE:
                        assert BoardState(succ[0]).is_valid()
                    frontier.put(succ)
                    parent[succ] = (current, action)
        raise ValueError("No path to goal")

    def astar(self) -> StPath:
        frontier : PriorityQueue[tuple[float, MarkovState]] = PriorityQueue()
        frontier.put((0, self.initial_state))
        parent : BpDict = {}
        costs_so_far : Dict = {}
        costs_so_far[self.initial_state] = 0
        parent[self.initial_state] = (None, None)
        while not frontier.empty():
            current = frontier.get()[1]
            if current in self.goal_state_set:
                backwards = list(get_path(current, parent, None))
                backwards.reverse()
                if VALIDATE:
                    self.validate_path(backwards)
                return backwards
            for action in self.get_actions(current):
                if VALIDATE:
                    self.sim.validate_action(action, current[1])
                succ = self.execute(current, action)
                new_cost = costs_so_far[current] + 1
                if (not (succ in parent)) and ((not succ in costs_so_far) or (new_cost < costs_so_far[succ])):
                    if VALIDATE:
                        assert BoardState(succ[0]).is_valid()
                    costs_so_far[succ] = new_cost
                    frontier.put((new_cost + self.get_heuristic(succ), succ))
                    parent[succ] = (current, action)
        raise ValueError("No path to goal")

    def get_heuristic(self, current_state):
        minimum = 1000
        player_index = current_state[1]
        for goal in self.goal_state_set:
            if (goal[1] == player_index):
                distance = self.get_hamming(current_state, goal)
                if (distance < minimum):
                    minimum = distance
        return minimum
    
    def get_hamming(self, current, goal):
        distance = 0
        curr_state = current[0]
        goal_state = goal[0]
        for i in range(len(goal_state)):
            if (curr_state[i] != goal_state[i]):
                distance = distance + 1
        return distance
    
    def alpha_beta_policy(self, state, player, depth):
        alpha_beta = AlphaBeta(player, depth)
        return alpha_beta.policy(state)
    
    def minimax_policy(self, state, player):
        minimax = MinimaxPolicy(player)
        return minimax.policy(state)
    
    def mcts_policy(self, state, player):
        mcts = MCTS(player)
        return mcts.policy(state)

    def random_policy(self, state, player):
        random = RandPolicy(player)
        return random.policy(state)

def get_path(current: Optional[MarkovState], parent: BpDict,
        action: Optional[Action]) -> StPath:
    "Retrieve the shorest path to a state by following back-pointers"
    if current is not None:
        yield (current, action)
        pred, action = parent[current]
        yield from get_path(pred, parent, action)

