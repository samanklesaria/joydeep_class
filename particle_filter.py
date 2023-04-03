from game import BoardState, PlayerIx, EncState
import numpy as np
import numpy.random as npr
import random
import game

# Posterior state density, given that player just moved
def get_posterior(obs: EncState, state: BoardState, player: PlayerIx) -> float:
    relevant_obs = state.state[6 * player : 6 * player + 5]
    total_prob = 1.0
    for ix, y in enumerate(relevant_obs):
        piece_ix = 6 * player + ix
        probs = game.obs_probs(state, piece_ix)
        total_prob *= probs[y]
    return total_prob

def resample(particles, weights):
    N = len(weights)
    weights = weights / np.sum(weights)
    u = (np.arange(N) + npr.rand()) / N
    bins = np.cumsum(weights)
    return particles[np.digitize(u, bins)]

def smc(observations):
    particles = np.array([BoardState() for _ in range(500)])
    player = 0
    for obs in observations:
        weights = []
        for state in particles:
            actions = list(game.generate_valid_actions(state, player))
            chosen = random.randrange(len(actions))
            offset, pos = actions[chosen]
            state.update(offset + player * 6, pos)
            weights.append(get_posterior(obs, state, player))
        particles = resample(particles, weights)
        player ^= 1
    return particles