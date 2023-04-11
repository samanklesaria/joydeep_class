import game
import numpy as np
from game import BoardState, MCTSPlayer, AlphaBetaPlayer, MinimaxPlayer, RandPlayer, RandLoggerPlayer, smc
from search import GameStateProblem

def fake_observations():
    log = []
    b1 = BoardState()
    encoded_state_tuple = (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52)
    b1.state = np.array(encoded_state_tuple)
    b1.decode_state = b1.make_state()
    players = [RandLoggerPlayer(GameStateProblem(b1, b1, 0), 0, log), RandLoggerPlayer(GameStateProblem(b1, b1, 0), 1, log)]
    sim = game.GameSimulator(players, n_steps=10)
    sim.go()
    return log[1:] # The first state is known

def test_infer_last_state_from_seq():
    obs = fake_observations()
    particles = smc(obs)

test_infer_last_state_from_seq()

def test_observation():
    bs = game.GameSimulator([])
    bs.sample_observation(0)
    bs.sample_observation(1)

test_observation()