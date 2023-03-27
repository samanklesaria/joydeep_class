import game
import numpy as np
import util
from policies import RandLogger
from particle_filter import smc

def fake_observations():
    log = []
    players = [RandLogger(0, log), RandLogger(1, log)]
    sim = game.GameSimulator(players, n_steps=10)
    sim.go()
    return log[1:] # The first state is known

def test_infer_last_state_from_seq():
    obs = fake_observations()
    smc(obs)

def test_observation():
    bs = game.GameSimulator([])
    bs.sample_observation(0)
    bs.sample_observation(1)
