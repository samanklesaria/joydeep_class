from game import BoardState
import numpy as np
import numpy.random as npr
import random
import game

def get_posterior(obs, state):
    # TODO
    pass

def resample(particles, weights):
    N = len(weights)
    weights = weights / np.sum(weights)
    u = (np.arange(N) + npr.rand()) / N
    bins = np.cumsum(weights)
    return particles[np.digitize(u, bins)]

def smc(observations):
    particles = [BoardState() for _ in range(500)]
    player = 0
    for obs in observations:
        weights = []
        for state in particles:
            actions = list(game.generate_valid_actions(state, player))
            chosen = random.randrange(len(actions))
            offset, pos = actions[chosen]
            state.update(offset + player * 6, pos)
            weights.append(get_posterior(obs, state))
        particles = resample(particles, weights)
        player ^= 1
    # Each step:
    # Pick a random action.
    # Weigh the action by its posterior probability
    # Resample
