from copy import deepcopy
import numpy as np
from collections import namedtuple, defaultdict
from typing import Set, Union, Iterable, Optional, Tuple, Dict, List
import random
from dataclasses import dataclass
from functools import reduce
import math
from queue import Queue
import numpy.random as npr


VALIDATE = False

# Representation of the state using EncPos
EncState = Union[tuple, np.ndarray]

# Representation of the state using NdPos
CoordState = np.ndarray

# Coordinates in the game, as a tuple
TupPos = Tuple[int,int]

# Coordinates in the game, as an ndarray
NdPos = np.ndarray

Pos = Union[TupPos, NdPos]

# Representation of the state as a list of  Pos
StupidState = List[Pos]

# Encoded position
EncPos = int

# Which player is playing?
PlayerIx = int # in [0,1]

# Represents a piece modulo PlayerIx
RelativePieceIx = int # in [0,5]

# Represents a piece
PieceIx = int # in [0, 11]

# A move in the game
Action = Tuple[RelativePieceIx, EncPos]

N_ROWS = 8
N_COLS = 7

limits = np.array([N_COLS, N_ROWS])

default_start_state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])

def decode(state: EncState) -> CoordState:
    result = np.divmod(state, N_COLS)
    return np.stack([result[1], result[0]]).T

def encode(state: CoordState) -> EncState:
    return state[...,0] + state[...,1] * N_COLS


def stupid_to_coordstate(ss: StupidState) -> CoordState:
    return np.stack([np.array(a) for a in ss])

class BoardState:
    """
    Represents a state in the game
    """

    locs1 = np.arange(5) # player 1 locs
    locs2 = 6 + locs1 # player 2 locs
    b = [5,11] # player ball ixs
    block_locs = np.concatenate([locs1, locs2])

    def __init__(self, state: Union[EncState, "BoardState", None] =None):
        """
        Initializes a fresh game state
        """

        if state is None:
            self.state = np.copy(default_start_state)
            self.stated = np.stack(self.make_state())
        elif not isinstance(state, BoardState):
            self.state = np.array(state)
            self.stated = np.stack(self.make_state())
        else:
            self.state = np.copy(state.state)
            self.stated = np.copy(state.stated)

        

    # Maintains Eric's silly list of tuples interface for backwards compat
    @property
    def decode_state(self):
        return [self.decode_single_pos(d) for d in self.state]

    @decode_state.setter
    def decode_state(self, value):
        self.stated = np.stack([np.array(a) for a in value])

    def update(self, idx: PieceIx, val: EncPos):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.stated[idx, :] = np.array(self.decode_single_pos(self.state[idx]))

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: Pos) -> EncPos:
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
        return cr[0] + cr[1] * N_COLS

    def decode_single_pos(self, n: EncPos) -> TupPos:
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        return (n % N_COLS, n // N_COLS)

    def is_termination_state(self) -> bool:
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.
        """
        if not self.is_valid(): return False
        r0 = self.stated[self.b[0]][1]
        r1 = self.stated[self.b[1]][1]
        return r0 == (N_ROWS - 1) or r1 == 0 

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        """

        outcome = True

        # All 12 pieces must be on the board (in [0,55])
        outcome &= (self.state >= 0).all() and (self.state <= 55).all()
        outcome &= len(self.state) == 12

        # Two blocks cannot occupy the same space
        block_state = self.state[self.block_locs]
        outcome &= len(np.unique(block_state)) == len(block_state)

        # Each ball must be held by a block of the same color
        outcome &= (self.state[self.locs1] == self.state[self.b[0]]).any()
        outcome &= not (self.state[self.locs2] == self.state[self.b[0]]).any()
        outcome &= (self.state[self.locs2] == self.state[self.b[1]]).any()
        outcome &= not (self.state[self.locs1] == self.state[self.b[1]]).any()
        return outcome

    def occupied(self, y):
        return (self.state == y).any()

# Wraps a potential move with its associated min semilattice element
PotentialMove = namedtuple('PotentialMove', ['norm', 'player', 'y'])

# Least upper bound in min norm semilattice
def lub(a: PotentialMove, b: PotentialMove) -> PotentialMove:
    if a.norm < b.norm:
        return a
    #if a.norm == b.norm:
    #    raise ValueError("Two pieces at the same position")
    return b

class Rules:

    @staticmethod
    def single_piece_actions(st: BoardState, piece_ix: PieceIx) -> Iterable[EncPos]:
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        """

        if st.state[st.b[piece_ix // 6]] != st.state[piece_ix]:
            x = st.stated[piece_ix]
            for move in (np.array([1, 2]), np.array([2, 1])):
                for d1 in (-1, 1):
                    for d2 in (-1, 1):
                        pos = x + move * np.array([d1, d2])
                        if (pos >= 0).all() and (pos <= np.array([6, 7])).all():
                            y = st.encode_single_pos(pos)
                            if not st.occupied(y):
                                yield y


    @staticmethod
    def single_ball_actions(st: BoardState, player_ix: PlayerIx) -> Iterable[EncPos]:
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for player_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        """
        return Rules.ball_actions_from(st, st.state[st.b[player_ix]], player_ix)

    @staticmethod
    def ball_actions_from(st: BoardState, y: EncPos, player_ix: PlayerIx) -> Set[EncPos]:
        # Look where we can pass ball. Then from there, look for any new places we can pass. And so on. 
        passes = set([y])

        def recurse(x: NdPos):
            for y in Rules.pass_actions(st, player_ix, x, passes):
                passes.add(y)
                recurse(np.array(st.decode_single_pos(y)))

        recurse(np.array(st.decode_single_pos(y)))
        passes.remove(y)
        return passes
        
    @staticmethod
    def pass_actions(st: BoardState, player_ix: PlayerIx, x: NdPos, passes: Set[EncPos]): 
        # For each person, get their vector to the ball.
        # Map each (vector, team) to its unit length version. Reverse the mapping, using lub defined above.
        # Then yield all the remaining vectors for the current team.

        ys = st.state[st.block_locs]
        vecs = st.stated[st.block_locs, :] - x[None,:]
        absvecs = np.abs(vecs)
        diagonals = absvecs[:, 0] == absvecs[:, 1]
        verticals = vecs[:, 0] == 0
        horizontals = vecs[:, 1] == 0
        norms = absvecs.max(axis=1)
        nz = norms > 0
        valid = (diagonals | verticals | horizontals) & nz

        players = np.floor_divide(st.block_locs[valid], 6)
        units = (tuple(u) for u in
            np.floor_divide(vecs[valid], norms[valid,None]))
        moves : Dict[tuple, PotentialMove] = dict()
        for (y, n, p, u) in zip(ys[valid], norms[valid], players, units):
            newmove = PotentialMove(n, p, y)
            if u in moves:
                moves[u] = lub(moves[u], newmove)
            else:
                moves[u] = newmove

        for (k,v) in moves.items():
            if v.player == player_ix and v.y not in passes:
                yield v.y

def next_state(state: BoardState, action: Action, player_idx: PlayerIx):
    offset_idx = player_idx * 6 ## Either 0 or 6
    idx, pos = action
    state2 = BoardState(state)
    state2.update(offset_idx + idx, pos)
    return state2

def generate_valid_actions(state: BoardState, player_ix: PlayerIx, can_move: bool = True) -> Iterable[Action]:
    if can_move:
        for i in range(5):
            piece = 6 * player_ix + i
            for action in Rules.single_piece_actions(state, piece):
               yield (i, action)
    for action in Rules.single_ball_actions(state, player_ix):
        yield (5, action)

# Posterior state density, given that player just moved
def get_posterior(obs: EncState, state: BoardState, player: PlayerIx) -> float:
    relevant_obs = obs[6 * player : 6 * player + 5]
    #print('player is ' + str(player) + ' relevant ' + str(relevant_obs))
    total_prob = 1.0
    for ix, y in enumerate(relevant_obs):
        piece_ix = 6 * player + ix
        probs = obs_probs(state, piece_ix)
        #print(probs[y])
        total_prob *= probs[y]
    return total_prob

def resample(particles, weights):
    N = len(weights)
    weights = weights / np.sum(weights)
    u = (np.arange(N) + npr.rand()) / N
    bins = np.cumsum(weights)
    return particles[np.digitize(u, bins)]

def particle_filter(obs, old_particles, player_index):
    weights = []
    player = player_index ^ 1
    print('STARTING!!!!!!!!!!!!')
    for i in range(len(old_particles)):
        temp = BoardState(obs)
        encoded_obs = [temp.encode_single_pos(d) for d in temp.state]
        for j in range(6):
            old_particles[i].state[j + 6 * player_index] = encoded_obs[j + 6 * player_index]
        actions = list(generate_valid_actions(old_particles[i], player))
        chosen = random.randrange(len(actions))
        offset, pos = actions[chosen]
        #print('offset ' + str(offset) + ' pos ' + str(pos))
        old_particles[i].update(offset + player * 6, pos)
        weights.append(get_posterior(obs, deepcopy(old_particles[i]), player))
    #print(weights)
    #old_particles = resample(old_particles, weights)
    for i in range(len(old_particles)):
        particle = old_particles[i]
        print(str(particle.state) + '    ' + str(weights[i]))
    #return old_particles

def obs_probs(game_state, piece_ix):
    pos = game_state.stated[piece_ix, :]
    enc_pos = game_state.encode_single_pos(pos)
    probs = defaultdict(lambda: 0)
    probs[enc_pos] = 0.6
    for direction in [np.array([1,0]), np.array([0,1])]:
        for magnitude in [-1, 1]:
            offset = direction * magnitude
            new_pos = pos + offset
            encoded = game_state.encode_single_pos(new_pos)
            if encoded in game_state.state or (new_pos < 0).any() or (new_pos >= limits).any():
                probs[enc_pos] += 0.1
            else:
                probs[encoded] = 0.1
    return probs

def smc(observations, player_index=0):
    particles = np.array([BoardState() for _ in range(500)])
    player = player_index
    for obs in observations:
        weights = []
        for state in particles:
            actions = list(generate_valid_actions(state, player))
            chosen = random.randrange(len(actions))
            offset, pos = actions[chosen]
            state.update(offset + player * 6, pos)
            weights.append(get_posterior(obs, state, player))
        particles = resample(particles, weights)
        player ^= 1
    return particles

def sample_observation(game_state, opposing_ix):
    chosen = []
    ball_loc = None
    for ix in range(6 * opposing_ix, 6 * opposing_ix + 5):
        probs = obs_probs(game_state, ix)
        choice = np.random.choice(list(probs.keys()), p=list(probs.values()))
        if game_state.state[ix] == game_state.state[6 * opposing_ix + 5]:
            ball_loc = choice
        chosen.append(choice)
    if opposing_ix == 0:
        new_state = np.concatenate(
            (chosen, [ball_loc], game_state.state[6:]))
    else:
        new_state = np.concatenate(
            (game_state.state[:6] , chosen , [ball_loc]))
    return new_state

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players, n_steps=200, log=False, validate=VALIDATE, use_heuristic=False, tries_per_round=7):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players
        self.log = log
        self.n_steps = n_steps
        self.validate = validate
        self.use_heuristic = use_heuristic
        self.max_tries_per_round = tries_per_round
    
    def sample_observation(self, opposing_ix):
        new_state = sample_observation(self.game_state, opposing_ix)
        return [self.game_state.decode_single_pos(d) for d in new_state]

    def winner(self):
        return self.current_round % 2

    def run_with_ground_truth(self):
        """
        Runs a game simulation
        """
        observations = []
        states = []
        actions = []
        while not (self.game_state.is_termination_state() or self.current_round > self.n_steps):
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            observation = self.sample_observation((player_idx + 1) % 2)
            is_valid_action = False
            tries = 0
            while (not is_valid_action) and (tries < self.max_tries_per_round):
                action, value = self.players[player_idx].policy(observation)
                try:
                    is_valid_action = self.validate_action(action, player_idx)
                except:
                    is_valid_action = False
                tries += 1
                print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}, Validity: {is_valid_action}")
                self.players[player_idx].process_feedback(observation, action, is_valid_action)
            if not is_valid_action:
                ## If an invalid action is provided, then the other player will be declared the winner
                return (observations, states, actions)
            else:
                actions.append(action)
                states.append(self.game_state)
                observations.append(encode(np.array(observation)))
            ## Updates the game state
            self.update(action, player_idx)
        return (observations, states, actions)


    def run(self):
        """
        Runs a game simulation
        """
        while not (self.game_state.is_termination_state() or self.current_round > self.n_steps):
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            observation = self.sample_observation((player_idx + 1) % 2)
            is_valid_action = False
            tries = 0
            while (not is_valid_action) and (tries < self.max_tries_per_round):
                action, value = self.players[player_idx].policy(observation)
                try:
                    is_valid_action = self.validate_action(action, player_idx)
                except:
                    is_valid_action = False
                tries += 1
                print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}, Validity: {is_valid_action}")
                self.players[player_idx].process_feedback(observation, action, is_valid_action)
            if not is_valid_action:
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black provided an invalid action"
            ## Updates the game state
            self.update(action, player_idx)
        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    """
    def run(self):
        "Backwards compat version of 'go' for Eric"
        self.go()
        player_idx = self.winner()
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    """
        
    def get_ball_heuristic(self, state, player_idx):
        ball_pos_row = ((state.state[player_idx*6+5]) // 8) % 7
        ball_start_row = player_idx * 8
        ball_dist = abs(ball_start_row - ball_pos_row) 

        opponent_ball_row = ((state.state[(1-player_idx) * 6 + 5]) // 8) % 7
        opponent_ball_start_row = (1-player_idx) * 8
        opponent_dist = abs(opponent_ball_start_row - opponent_ball_row)
    
        return (ball_dist - opponent_dist) / 8
    
    def get_pass_options(self, board_state, player_idx):
        explored_players = set()
        reachable_players = list()

        state = np.reshape(board_state.state, (2,-1))
        ball_pos =  state[player_idx][5] 

        reachable_players.append(ball_pos)
        while reachable_players:
            item = reachable_players.pop(0)    
            current_col, current_row = board_state.decode_single_pos(item)
            explored_players.add(item)
            for player_pos in state[player_idx][:5]: 
                if player_pos not in explored_players:
                    player_col, player_row = board_state.decode_single_pos(player_pos)
                    if abs(player_col - current_col) == abs(player_row - current_row):
                        reachable_players.append(player_pos)
                    elif abs(player_col - current_col) == 0:
                        reachable_players.append(player_pos)
                    elif abs(player_row - current_row) == 0:
                        reachable_players.append(player_pos)
        return explored_players

    def get_striking_distance(self, state, player_idx):
        reachable_players = self.get_pass_options(state,player_idx)
        furthest_player_row = None
        distance = 0
        if player_idx:
            _, furthest_player_row = state.decode_single_pos(min(reachable_players))
            distance = math.floor((8 - furthest_player_row) / 2) - 1
        else:
            _, furthest_player_row = state.decode_single_pos(max(reachable_players))
            distance = math.floor((furthest_player_row) / 2) - 1
        h = (distance - 1) / 2
        return h
    
    def get_heuristic(self, state,player_idx):
        return (self.get_striking_distance(state,player_idx) - self.get_striking_distance(state, 1-player_idx)) / 2
    
    def go(self):
        """
        Runs a game simulation
        """

        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2

            if self.current_round >= self.n_steps:
                if (self.use_heuristic):
                    val = self.get_heuristic(self.game_state, player_idx)
                    if (val > 0):
                        return player_idx
                    else:
                        return 1 - player_idx
                else:
                    return None

            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state )
            if self.log:
                print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if self.validate:
                self.validate_action(action, player_idx)

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        player_idx = self.current_round % 2
        if self.log:
            print("Winner is", player_idx)
        return player_idx
       
    def validate_action(self, action: Action, player_idx: PlayerIx) -> bool:
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        """
        assert player_idx in [0,1]
        assert action[0] < 6
        piece_ix = player_idx * 6 + action[0]
        if action[0] == 5:
            if action[1] not in Rules.ball_actions_from(self.game_state, self.game_state.state[piece_ix], player_idx):
                raise ValueError("Cannot move a ball that way")
        else:
            if action[1] not in set(Rules.single_piece_actions(self.game_state, piece_ix)):
                raise ValueError("Cannot move a block that way")
        return True
    
    def update(self, action: Action, player_idx: PlayerIx):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)

    def generate_valid_actions(self, player_ix: PlayerIx) -> Iterable[Action]:
        return generate_valid_actions(self.game_state, player_ix)
    
class Player:
    def __init__(self, policy_fnc):
        self.policy_fnc = policy_fnc
    def policy(self, decode_state):
        pass
    def process_feedback(self, observation, action, is_valid):
        pass

class MCTSPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.mcts_policy)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.invalid_actions = set()
    
    def policy(self, observation):
        obs = deepcopy(observation)
        temp = BoardState(obs)
        encoded_obs = [temp.encode_single_pos(d) for d in temp.state]
        state = BoardState(encoded_obs)
        return self.policy_fnc(state, self.player_idx, self.invalid_actions)
    
    def process_feedback(self, observation, action, is_valid):
        if (not is_valid):
            self.invalid_actions.add(action)
        else:
            self.invalid_actions = set()
    
class AlphaBetaPlayer(Player):
    def __init__(self, gsp, player_idx, depth=3):
        super().__init__(gsp.alpha_beta_policy)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.depth = depth
        self.invalid_actions = set()

    def policy(self, observation):
        obs = deepcopy(observation)
        temp = BoardState(obs)
        encoded_obs = [temp.encode_single_pos(d) for d in temp.state]
        state = BoardState(encoded_obs)
        return self.policy_fnc(state, self.player_idx, self.depth, self.invalid_actions)
    
    def process_feedback(self, observation, action, is_valid):
        if (not is_valid):
            self.invalid_actions.add(action)
        else:
            self.invalid_actions = set()
    
class MinimaxPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.minimax_policy)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.invalid_actions = set()

    def policy(self, observation):
        obs = deepcopy(observation)
        temp = BoardState(obs)
        encoded_obs = [temp.encode_single_pos(d) for d in temp.state]
        state = BoardState(encoded_obs)
        return self.policy_fnc(state, self.player_idx, self.invalid_actions)

    def process_feedback(self, observation, action, is_valid):
        if (not is_valid):
            self.invalid_actions.add(action)
        else:
            self.invalid_actions = set()

    
class RandPlayer(Player):
    def __init__(self, gsp, player_idx):
        super().__init__(gsp.random_policy)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
    def policy(self, observation):
        obs = deepcopy(observation)
        temp = BoardState(obs)
        encoded_obs = [temp.encode_single_pos(d) for d in temp.state]
        state = BoardState(encoded_obs)
        return self.policy_fnc(state, self.player_idx)
    

class RandLoggerPlayer(Player):
    def __init__(self, gsp, player_idx, log):
        """
        You can customize the signature of the constructor above to suit your needs.
        In this example, in the above parameters, gsp is a GameStateProblem, and
        gsp.adversarial_search_method is a method of that class.
        """
        super().__init__(gsp.random_logger_policy)
        self.gsp = gsp
        self.b = BoardState()
        self.player_idx = player_idx
        self.statelog = log

    def policy(self, observation):
        obs = deepcopy(observation)
        temp = BoardState(obs)
        encoded_obs = [temp.encode_single_pos(d) for d in temp.state]
        self.statelog.append(encoded_obs)
        state = BoardState(encoded_obs)
        return self.policy_fnc(state, self.player_idx, self.statelog)