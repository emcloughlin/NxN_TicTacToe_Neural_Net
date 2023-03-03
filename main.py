"""
main.py
Author: Evan McLoughlin
Created: 3.1.2023
"""

import numpy as np
import TicTacToe
import ReinforcementLearning
import os, psutil
ttt = TicTacToe.TicTacToe()
rl = ReinforcementLearning.ReinforcementLearning(pop_size=250, tests_per_model=50)
rl.create_gen_1()

# %%

def get_nn_output(game_matrix, neural_network):
    """ Get the output of a neural net """
    return neural_network.network_function(game_matrix)

# %%

def play_TTT(neural_network, isPlayer1):
    game_matrix = ttt.init_game_matrix(3)
    game_state = -1.0
    turn = 0
    #ttt.print_game_matrix(game_matrix)
    while (game_state == -1.0):
        if (turn % 2 == 0):
            game_matrix = ttt.two_d_game_matrix_to_vector(game_matrix)
            game_matrix = ttt.NN_player(
                game_matrix,
                get_nn_output(game_matrix, neural_network),
                isPlayer1)
            try:
                if (type(game_matrix) != np.ndarray):
                    return -2.0
            except (ValueError):
                pass
        else:
            game_matrix = ttt.random_move(game_matrix, not isPlayer1)
        game_state = ttt.check_end_state(game_matrix)
        #ttt.print_game_matrix(game_matrix)
        turn += 1
    return game_state

# %%

def collect_testing_data(isPlayer1=False):
    """ Returns a list of success ratings with elements corresponding to their
        respective models in network_list
        Keywords:
        isPlayer1 -- whether or not the model should be X (P1) or O (P2)
    """
    scores_list = np.zeros((0, 0))
    for network in rl.network_list:
        cumulative_score = 0
        for test_count in range(rl.tests_per_model):
            cumulative_score += game_result_to_score(
                    play_TTT(network, isPlayer1), isPlayer1)
        scores_list = np.append(scores_list, cumulative_score)
        #print(final_score)
    final_score_list = scores_list / rl.tests_per_model
    return final_score_list



# %%

def game_result_to_score(game_result, isPlayer1):
    """ Change the output of the game result into a score for training """
    match game_result:
        case -2.0:
            return 0.0
        case 0.0:
            return 0.666
        case 1.0:
            if isPlayer1:
                return 1.0
            else:
                return 0.333
        case 2.0:
            if not isPlayer1:
                return 1.0
            else:
                return 0.333

# %%

def sum_square_err_array(testing_data):
    """ Convert score elements to the sums of their square error"""
    return np.square(1 - testing_data)

# %%

def square_data(data):
    return np.square(data)

# %%

def determined_center_of_mass():
    """ Find the center of mass of the network weights where (mass) is the
        success rate of a network and (location) are the values of the weights
    """
    data = collect_testing_data()
    #TODO: im tired fix this so it works in the general case
    layer_weights = np.zeros_like(rl.network_list[0].network_weights)
    for index in range(np.size(data)):
        layer_weights[0] += rl.network_list[index].network_weights[0] * data[index]
        layer_weights[1] += rl.network_list[index].network_weights[1] * data[index]
    layer_weights[0] = layer_weights[0] / np.sum(data)
    layer_weights[1] = layer_weights[1] / np.sum(data)
    return data, layer_weights

# %%

def train_until_interrupted():
    """ Repeatedly trains and evolves the networks until stopped by user. In
        the future, will implement threading, but for now, just input 'y' to
        continue
    """
    continue_training = True
    training_round = 1
    while continue_training:
        print("Round {t_round}".format(t_round=training_round))
        data = collect_testing_data()
        print("Data collected. Evolving...")
        #print("Data collected. Current memory usage: {usage}\nEvolving..."
        #      .format(usage =
        #          psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
        rl.evolve_networks(data)
        #print("Memory usage after deletion: {usage}".format(usage =
        #     psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
        training_round += 1



























