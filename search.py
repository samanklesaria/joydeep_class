import numpy as np
from queue import Queue, PriorityQueue
from game import GameSimulator, PlayerIx, EncState, Action, BoardState
import typing
from typing import Iterable, Optional, Dict, Tuple, Union
import heapq

VALIDATE = True # False

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
        frontier = PriorityQueue()
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

def get_path(current: Optional[MarkovState], parent: BpDict,
        action: Optional[Action]) -> StPath:
    "Retrieve the shorest path to a state by following back-pointers"
    if current is not None:
        yield (current, action)
        pred, action = parent[current]
        yield from get_path(pred, parent, action)

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 
    """ Here is an example:
    
    def my_snazzy_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)
    """

