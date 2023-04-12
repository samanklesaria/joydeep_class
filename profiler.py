from game import MCTSPlayer, AlphaBetaPlayer, MinimaxPlayer, RandPlayer
from game import BoardState, GameSimulator, Rules
from search import GameStateProblem
import game
def mcts_game():
    b1 = BoardState()
    gsp1 = GameStateProblem(b1, b1, 0)
    gsp2 = GameStateProblem(b1, b1, 0)
    players = [MCTSPlayer(gsp1, 0), RandPlayer(gsp2, 1)]
    sim = game.GameSimulator(players, n_steps=200, log=True)
    sim.run()

def minimax_game():
    b1 = BoardState()
    gsp1 = GameStateProblem(b1, b1, 0)
    gsp2 = GameStateProblem(b1, b1, 0)
    players = [MinimaxPlayer(gsp1, 0), RandPlayer(gsp2, 1)]
    sim = game.GameSimulator(players, n_steps=200, log=True)
    sim.run()

def alpha_beta_game():
    b1 = BoardState()
    gsp1 = GameStateProblem(b1, b1, 0)
    gsp2 = GameStateProblem(b1, b1, 0)
    players = [AlphaBetaPlayer(gsp1, 0), RandPlayer(gsp2, 1)]
    sim = game.GameSimulator(players, n_steps=200, log=True)
    sim.run()

#mcts_game()
#winner = minimax_game()
#alpha_beta_game()

count = 0
for i in range(100):
    winner = mcts_game()
    if (not winner):
        count = count + 1
print(count)