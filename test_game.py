import numpy as np
import queue
import pytest
import game
from game import BoardState, GameSimulator, Rules
from search import GameStateProblem
from game import MCTSPlayer, AlphaBetaPlayer, MinimaxPlayer, RandPlayer

class TestGame:

    @pytest.mark.parametrize("p1_class,p2_class,encoded_state_tuple,exp_winner,exp_stat", [
    (MinimaxPlayer, AlphaBetaPlayer,
    (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52),
    "WHITE", "No issues")
    ])
    def test_adversarial_search(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        b1 = BoardState()
        b1.state = np.array(encoded_state_tuple)
        b1.decode_state = b1.make_state()
        players = [
        p1_class(GameStateProblem(b1, b1, 0), 0),
        p2_class(GameStateProblem(b1, b1, 0), 1)]
        sim = GameSimulator(players)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        assert winner == exp_winner and status == exp_stat