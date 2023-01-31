import numpy as np
import matplotlib.pyplot as plt # type: ignore
from game import CoordState, BoardState, Rules, PieceIx

# Grid lines for plotting the board
x_ticks = np.arange(0, 7)
y_ticks = np.arange(0, 8)

def printstate(locs: CoordState):
    "Prints a human readable representation of the state"
    itr = iter(locs)
    print("White Blocks")
    for a in range(5):
        print(next(itr))
    print("While ball", next(itr))
    print("Black Blocks")
    for a in range(5):
        print(next(itr))
    print("Black ball", next(itr))

def plotstate(locs: CoordState):
    "Plot the state and wait for user input"
    itr = iter(locs)
    for a in range(5):
        x,y = next(itr)
        plt.scatter([x], [y], c='r', s=100)
    x,y = next(itr)
    plt.scatter([x], [y], c='r', marker='s', s=100)
    for a in range(5):
        x,y = next(itr)
        plt.scatter([x], [y], c='b', s=100)
    x,y = next(itr)
    plt.scatter([x], [y], c='b', marker='s', s=100)
    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.axis('equal')
    plt.grid(which='both')
    plt.show()

def block_actions(st: BoardState, ix: PieceIx):
    "What actions are applicable for the block at ix given st?"
    return [st.decode_single_pos(y) for y in Rules.single_piece_actions(st, ix)]

def view_solution(steps):
    for (state, action) in steps:
        st = BoardState(state[0])
        print("Player", state[1])
        if action is not None:
            print("Action", st.decode_single_pos(action[1]))
        plotstate(st.stated)

