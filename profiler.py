from policies import MCTS, RandPolicy
import game

def mcts_game():
    players = [MCTS(0, limit=12), RandPolicy(1)]
    sim = game.GameSimulator(players, n_steps=1, log=True)
    sim.go()

mcts_game()
