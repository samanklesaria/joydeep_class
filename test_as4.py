from copy import deepcopy
from typing import Iterable, Tuple

import pytest
import game
import numpy as np
from game import BoardState, EncPos, MCTSPlayer, AlphaBetaPlayer, MinimaxPlayer, RandPlayer, RandLoggerPlayer, smc, obs_probs, Rules, Action
from search import GameStateProblem
from collections import defaultdict
import random

def fake_observations(encoded_state_tuple, n_steps):
    b1 = BoardState()
    b1.state = np.array(encoded_state_tuple)
    b1.decode_state = b1.make_state()
    players = [RandPlayer(GameStateProblem(b1, b1, 0), 0), RandPlayer(GameStateProblem(b1, b1, 0), 1)]
    sim = game.GameSimulator(players, n_steps=n_steps)
    observations, states, actions = sim.run_with_ground_truth()
    return observations, states, actions # The first state is known

def get_single_piece_moves(position) -> Iterable[EncPos]:
    decoded_pos = (position % 7, position // 7)
    for move in (np.array([1, 2]), np.array([2, 1])):
        for d1 in (-1, 1):
            for d2 in (-1, 1):
                pos = decoded_pos + move * np.array([d1, d2])
                if (pos >= 0).all() and (pos <= np.array([6, 7])).all():
                    re_encode = pos[0] + pos[1] * 7
                    yield re_encode

def get_pieces_beliefs(observations):
    beliefs = []
    for i in range(len(observations)):
        observation = BoardState(observations[i])
        if (i == 0):
            new_belief = [defaultdict(lambda: 0) for x in range(10)]
            # First observation. Use obs probs for each dict
            # Iterate through each piece
            for j in range(10):
                piece_index = j if (j < 5) else j + 1
                new_belief[j] = obs_probs(observation, piece_index)
            beliefs.append(new_belief)
        else:
            new_belief = [defaultdict(lambda: 0) for x in range(10)]
            # Get the observation probs for each piece. Multiply each element by current belief, then normalize
            for j in range(10):
                piece_index = j if (j < 5) else j + 1
                probs_from_obs = obs_probs(observation, piece_index)
                prob_sum = 0
                for enc_index in range(56):
                    # Get reachable places from here:
                    actions = list(get_single_piece_moves(enc_index))
                    actions.append(enc_index)
                    transition_prob = 1 / len(actions)
                    for new_location in actions:
                        last_belief = beliefs[i - 1][j][enc_index]
                        new_prob = last_belief * probs_from_obs[new_location] * transition_prob
                        new_belief[j][new_location] += new_prob
                        prob_sum = prob_sum + new_prob
                norm_factor = 1 / prob_sum
                for enc_index in range(56):
                    new_belief[j][enc_index] *= norm_factor
            beliefs.append(new_belief)
    return beliefs

def update_states_for_balls(observations, piece_arrangements):
    for i in range(len(observations)):
        for j in range(2):
            start_player_offset = 6 * j
            ball_index = 6 * j + 5
            player_obs = observations[i][ball_index - 5 : ball_index]
            ball_obs = observations[i][ball_index]
            # If ball obs only appears once in player obs, 
            viable_players = []
            for player_index in range(len(player_obs)):
                if (player_obs[player_index] == ball_obs):
                    viable_players.append(piece_arrangements[i][player_index + start_player_offset])
            piece_arrangements[i][ball_index] = random.choice(viable_players)

def generate_state_sequence(observations):
    piece_beliefs = get_pieces_beliefs(observations)
    # Now, generate our list of states based on these beliefs
    temp_state_sequence = []
    for t in range(len(observations)):
        temp_state = [0 for x in range(12)]
        t_piece_belief = piece_beliefs[t]
        for belief_index in range(10):
            piece_index = belief_index if (belief_index < 5) else belief_index + 1
            mle = max(t_piece_belief[belief_index], key=t_piece_belief[belief_index].get)
            temp_state[piece_index] = mle
        temp_state_sequence.append(temp_state)
    # Now get ball locations in the sequence:
    update_states_for_balls(observations, temp_state_sequence)
    # Now convert to BoardStates and return
    board_states_sequence = []
    for temp_state in temp_state_sequence:
        board_states_sequence.append(BoardState(temp_state))
    return board_states_sequence

def get_full_beliefs(observations):
    piece_beliefs = get_pieces_beliefs(observations)
    full_beliefs = []
    for i in range(len(observations)):
        full_belief = [defaultdict(lambda: 0) for x in range(12)]
        for start in range(5):
            for mult in range(2):
                full_belief[mult * 6 + start] = piece_beliefs[i][5 * mult + start]
        for j in range(2):
            start_player_offset = 6 * j
            ball_index = 6 * j + 5
            player_obs = observations[i][ball_index - 5 : ball_index]
            ball_obs = observations[i][ball_index]
            # If ball obs only appears once in player obs, 
            viable_beliefs = []
            for player_index in range(len(player_obs)):
                if (player_obs[player_index] == ball_obs):
                    belief_index = player_index if (j == 0) else player_index + 5
                    viable_beliefs.append(piece_beliefs[i][belief_index])
            chosen_belief = random.choice(viable_beliefs)
            full_belief[ball_index] = chosen_belief
        full_beliefs.append(full_belief)
    
    return full_beliefs

def get_manhattan(pos1, pos2):
    return (abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1]))

def get_state_distance(state1, state2):
    decoded_state1 = state1.decode_state
    decoded_state2 = state2.decode_state
    distance = 0
    # We will use the manhattan distance metric
    for i in range(12):
        pos1 = decoded_state1[i]
        pos2 = decoded_state2[i]
        distance += get_manhattan(pos1, pos2)
    return distance

def get_most_likely_action(old_board_state, new_board_state):
    old_state = old_board_state.state
    new_state = new_board_state.state
    movers = []
    for i in range(12):
        if (old_state[i] != new_state[i]):
            movers.append(i)
    action = (-1, -1)
    if (len(movers) == 1):
        # Easy, this was the action
        relative_piece_index = movers[0] if movers[0] <= 5 else movers[0] - 6
        action = (relative_piece_index, new_state[movers[0]])
    else:
        best_distance = 1000000
        best_mover = -1
        for mover in movers:
            temp = deepcopy(old_state)
            temp[mover] = new_state[mover]
            temp = BoardState(temp)
            distance = get_state_distance(temp, new_board_state)
            if (distance < best_distance):
                best_distance = distance
                best_mover = mover
        relative_piece_index = best_mover if best_mover <= 5 else best_mover - 6
        action = (relative_piece_index, new_state[best_mover])
    return action

def belief_dict_to_np(belief):
    vector = []
    for i in range(56):
        vector.append(belief[i])
    return np.array(vector)

def get_most_likely_action_2(old_board_state, old_belief, new_belief):
    champ_piece = -1
    champ_index = -1
    champ_prob = -1000
    for i in range(12):
        old_vector = belief_dict_to_np(old_belief[i])
        new_vector = belief_dict_to_np(new_belief[i])
        multiplied = np.outer(old_vector, new_vector)
        for old_index in range(56):
            actions = list(Rules.single_piece_actions(old_board_state, i))
            for new_index in actions:
                prob = multiplied[old_index][new_index]
                if (prob > champ_prob):
                    champ_piece = i
                    champ_prob = prob
                    champ_index = new_index
    relative_piece_index = champ_piece if champ_piece <= 5 else champ_piece - 6
    return (relative_piece_index, champ_index)

            
def get_actions_with_beliefs(state_sequence, beliefs):
    actions = []
    for index in range(1, len(beliefs)):
        old_state = state_sequence[index - 1]
        actions.append(get_most_likely_action_2(old_state, beliefs[index - 1], beliefs[index]))
    return actions

def get_actions_from_states(state_sequence):
    actions = []
    for index in range(1, len(state_sequence)):
        old_state = state_sequence[index - 1]
        new_state = state_sequence[index]
        actions.append(get_most_likely_action(old_state, new_state))
    return actions

def score_actions(predictions, actual):
    total = len(predictions)
    count = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        gt = actual[i]
        if (prediction[0] == gt[0] and prediction[1] == gt[1]):
            count += 1
    return count / total

def test_infer_actions_from_seq():
    print('Average Score is ', infer_actions_from_seq(10, 100))

def infer_actions_from_seq(N, k):
        accuracy_sum = 0
        for i in range(N):
            encoded_state_tuple =(1,2,3,4,5,3,50,51,52,53,54,52)
            # (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52)
            obs, states, gt_actions = fake_observations(encoded_state_tuple, k)
            # belief_sequence = get_full_beliefs(obs)
            state_sequence = generate_state_sequence(obs)
            #actions_2 = get_actions_with_beliefs(state_sequence, belief_sequence)
            actions = get_actions_from_states(state_sequence)
            score = score_actions(actions, gt_actions)
            # print(score)
            accuracy_sum += score
        return accuracy_sum / N


def infer_actions_from_seq2(N, k):
        accuracy_sum = 0
        for i in range(N):
            encoded_state_tuple =(1,2,3,4,5,3,50,51,52,53,54,52)
            # (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52)
            obs, states, gt_actions = fake_observations(encoded_state_tuple, k)
            belief_sequence = get_full_beliefs(obs)
            state_sequence = generate_state_sequence(obs)
            actions_2 = get_actions_with_beliefs(state_sequence, belief_sequence)
            # actions = get_actions_from_states(state_sequence)
            score = score_actions(actions_2, gt_actions)
            # print(score)
            accuracy_sum += score
        return accuracy_sum / N

def test_infer_last_state_from_seq():
    return infer_last_state_from_seq(10, 100)

def infer_last_state_from_seq(N, k):
        encoded_state_tuple =(1,2,3,4,5,3,50,51,52,53,54,52)
        accuracy_sum = 0
        for i in range(N):
            obs, states, gt_actions = fake_observations(encoded_state_tuple, k)
            state_sequence = generate_state_sequence(obs)
            last_state = state_sequence[-1]
            vals = (obs[-1] == last_state.state)
            accuracy_sum += (vals.sum() / len(vals))
        return accuracy_sum / N


def get_last_state_plot():
    return [infer_last_state_from_seq(5, i) for i in range(1, 200, 20)]

def get_action_plot():
    return [infer_actions_from_seq(5, i) for i in range(1, 200, 20)]

def get_action_plot2():
    return [infer_actions_from_seq2(5, i) for i in range(1, 200, 20)]

# class TestAS4:

#     @pytest.mark.parametrize("encoded_state_tuple,exp_state,n_steps", [
#     (MinimaxPlayer, AlphaBetaPlayer,
#     (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52),
#     "WHITE", "No issues")
#     ])
#     def test_infer_actions_from_seq(self, encoded_state_tuple, exp_state, n_steps):
#         obs = fake_observations()
#         print(obs[-1])
#         belief_sequence = get_full_beliefs(obs)
#         state_sequence = generate_state_sequence(obs)
#         #actions_2 = get_actions_with_beliefs(state_sequence, belief_sequence)
#         actions = get_actions_from_states(state_sequence)
#         print(actions)

    
# def test_observation():
#     bs = game.GameSimulator([])
#     bs.sample_observation(0)
#     bs.sample_observation(1)
