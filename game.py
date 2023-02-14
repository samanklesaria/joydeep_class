import numpy as np
from collections import namedtuple
from typing import Set, Union, Iterable, Optional, Tuple, Dict

# Representation of the state using EncPos
EncState = Union[tuple, np.ndarray]

# Representation of the state using NdPos
CoordState = np.ndarray

# Coordinates in the game, as a tuple
TupPos = Tuple[int,int]

# Coordinates in the game, as an ndarray
NdPos = np.ndarray

Pos = Union[TupPos, NdPos]

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

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self, state: Optional[EncState] =None):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        if state is not None:
            self.state = np.array(state)
        else:
            self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])

        
        self.locs1 = np.arange(5) # player 1 locs
        self.locs2 = 6 + self.locs1 # player 2 locs
        self.b = [5,11] # player ball ixs
        self.block_locs = np.concatenate([self.locs1, self.locs2])

        self.stated: CoordState = np.stack([np.array(self.decode_single_pos(d)) for d in self.state])

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
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

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
        return cr[0] + cr[1] * self.N_COLS 

    def decode_single_pos(self, n: EncPos) -> TupPos:
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        return (n % self.N_COLS, n // self.N_COLS)

    def is_termination_state(self) -> bool:
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.
        """
        if not self.is_valid(): return False
        r0 = self.stated[self.b[0]][1]
        r1 = self.stated[self.b[1]][1]
        return r0 == (self.N_ROWS - 1) or r1 == 0 

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
    if a.norm == b.norm:
        raise ValueError("Two pieces at the same position")
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

        def generator():
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
        return list(generator())


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

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_ix: PlayerIx) -> Iterable[Action]:
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        """
        def generator():
            for i in range(5):
                piece = 6 * player_ix + i
                for action in Rules.single_piece_actions(self.game_state, piece):
                   yield (i, action)
            for action in Rules.single_ball_actions(self.game_state, player_ix):
                yield (5, action)
        return set(generator())
        
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

