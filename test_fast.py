import game
import numpy as np
import util

def test_enc_dec():
    result = game.encode(game.decode(game.default_start_state))
    assert np.all(result == game.default_start_state)

def test_vectorized_dec():
    result = game.decode(game.default_start_state)
    assert np.all(result == game.BoardState().stated)

def test_vectorized_enc():
    for _ in range(20):
        s = np.random.randint(10, size=(12, 2))
        bs = game.BoardState()
        manual = np.array([bs.encode_single_pos(x) for x in s])
        vectorized = game.encode(s)
        assert np.all(vectorized == manual)

def test_non_vectorized_enc():
    for _ in range(100):
        s = np.random.randint(10, size=2)
        assert np.all(game.encode(s) == game.BoardState().encode_single_pos(s))

def test_non_vectorized_dec():
    for _ in range(100):
        s = np.random.randint(55)
        assert np.all(game.decode(s) == game.BoardState().decode_single_pos(s))

def test_observation():
    bs = game.GameSimulator([])
    print("Opponent 0")
    res0 = bs.sample_observation(0)
    cs = game.stupid_to_coordstate(res0)
    util.printstate(cs)
    print("\nOpponent 1")
    res0 = bs.sample_observation(1)
    cs = game.stupid_to_coordstate(res0)
    util.printstate(cs)