"""
TicTacToe.py
Author: Evan McLoughlin
Created: 2.23.2023
"""
# %%
import numpy as np
from numpy.typing import ArrayLike
import random
import ReinforcementLearning
import NeuralNet as nn

class TicTacToe:
# %%
# A function to return an n x n matrix to represent the game board
    def init_game_matrix(self, dimensions: int) -> np.ndarray:
        return np.zeros((dimensions, dimensions))

# %%

# A function to print the game board to the console
    def print_game_matrix(self, game_matrix: ArrayLike):
        for row in game_matrix:
            temp = ""
            for item in row:
                if (item == 1.0):
                    temp = temp + "x" # x's are 1.0
                elif (item == -1.0):
                    temp = temp + "o" # o's are -1.0
                else:
                    temp = temp + "-" # empty spaces are 0.0
                temp = temp + " "
            print(temp)

    # %%

    # A function to check if a the game is in an end state
    def check_end_state(self, game_matrix: ArrayLike):
        n = np.shape(game_matrix)[0] / 1
        draw = 0.0
        x_win = 1.0
        o_win = 2.0
        no_end = -1.0
        # Check diags for win state
        if (np.sum(np.diag(game_matrix)) == n or np.sum(np.diag(np.flipud(game_matrix))) == n):
            return x_win
        if (np.sum(np.diag(game_matrix)) == -n or np.sum(np.diag(np.flipud(game_matrix))) == -n):
            return o_win
        # Check rows for win state
        for row in game_matrix:
            if (np.sum(row) == n):
                return x_win
            if (np.sum(row) == -n):
                return o_win
        # Check columns for win state
        for row in game_matrix.T:
            if (np.sum(row) == n):
                return x_win
            if (np.sum(row) == -n):
                return o_win
        # Check if draw present
        for row in game_matrix:
            for element in row:
                if (element == 0):
                     # Assume if element is zero (no x or o), no end state reached
                    return no_end
        return draw # If nothing else is returned, a draw is present

    # %%

    # A function to make tic-tac-toe moves with no strategy (random input)
    def random_move(self, game_matrix: ArrayLike, play_as_x: bool) -> np.ndarray:
        if play_as_x:
            mark = 1.0
        else:
            mark = -1.0
        while True:
            row_index = random.randint(0, 2)
            column_index = random.randint(0, 2)
            if (game_matrix[row_index][column_index] != 0.0):
                continue
            else:
                game_matrix[row_index][column_index] = mark
                return game_matrix

    # %%

    # A function to make moves with a little strategy (if a winning move is present, make it)
    def random_move_winning(self, game_matrix: ArrayLike, play_as_x: bool) -> np.ndarray:
        if play_as_x:
            mark = 1.0
        else:
            mark = -1.0
        while True:
            for row in game_matrix:
                if (np.sum(row) == mark * 2):
                    with np.nditer(row, op_flags=['readwrite']) as it:
                        for x in it:
                            if (x[...] == 0.0):
                                x[...] = mark
                                return game_matrix
            for row in game_matrix.T:
                if (np.sum(row) == mark * 2):
                    with np.nditer(row, op_flags=['readwrite']) as it:
                        for x in it:
                            if (x[...] == 0.0):
                                x[...] = mark
                                return game_matrix
            if (np.sum(np.diag(game_matrix)) == mark * 2):
                for index in range(0, game_matrix.shape[0]):
                    if (game_matrix[index][index] == 0.0):
                        game_matrix[index][index] = mark
                        return game_matrix
            if (np.sum(np.diag(np.flipud(game_matrix))) == mark * 2):
                for index in range(0, game_matrix.shape[0]):
                    if (np.flipud(game_matrix)[index][index] == 0.0):
                        game_matrix = np.flipud(game_matrix)
                        game_matrix[index][index] = mark
                        return np.flipud(game_matrix)
            row_index = random.randint(0, 2)
            column_index = random.randint(0, 2)
            if (game_matrix[row_index][column_index] != 0.0):
                continue
            else:
                game_matrix[row_index][column_index] = mark
                return game_matrix

    # %%

    # A function to run a full game
    def tic_tac_toe(self, dimensions: int, display_game_board: bool):
        game_matrix = self.init_game_matrix(dimensions)
        if (display_game_board):
            print("Initial Board:")
            self.print_game_matrix(game_matrix)
        game_state = -1.0 # The condition used to specify an unfinished game
        turn = 0
        while (game_state == -1.0):
            if (turn % 2 == 0):
                x_turn = True
            else:
                x_turn = False
            game_matrix = self.random_move_winning(game_matrix, x_turn)
            if (display_game_board):
                print("Turn #{turn_num}:".format(turn_num = turn + 1))
                self.print_game_matrix(game_matrix)
            game_state = self.check_end_state(game_matrix)
            turn += 1
        if (display_game_board):
            print("Game Result:")
            print("Draw.") if (game_state == 0.0) else print("X wins!") if (game_state == 1.0) else print("O wins!")
        return game_state

# %%

    def two_d_game_matrix_to_vector(self, game_matrix):
        """ Return a vector containing the elements of the game matrix """
        return np.array([game_matrix.flatten()]).T

# %%

    def vector_to_2d_game_matrix(self, game_vector):
        """ Inverse of the 2d_game_matrix_to_vector() method """
        vector_root = int(np.sqrt(np.size(game_vector)))
        return game_vector.reshape((vector_root, vector_root))

# %%

    def tic_tac_toe_NN(self, dimensions=3):
        # Create the game matrix
        game_matrix = self.init_game_matrix(dimensions)
        # Set the game state to unfinished
        game_state = -1.0
        # Initialize the number of turns to 0
        turn = 0
        while (game_state == -1.0): # While the game is unfinished,
            if (turn % 2 == 0): # if it is player 1's turn
                # TODO: Fix this, it doesnt work - NN_player needs another
                # input, and currently the function has no way to get updated
                # output from the model every loop
                game_matrix = self.NN_player(game_matrix)
                if (game_matrix == None):
                    return -2.0
            else:
                game_matrix = self.random_move_winning(game_matrix, False)
            game_state = self.check_end_state(game_matrix)
            turn += 1
        return game_state

    def NN_player(self, game_matrix, network_output, isPlayer1):
        """ If the network outputs a legal move, returns the game matrix.
        If the output is not a legal move, returns none.
        """
        if (isPlayer1):
            mark = 1.0
        else:
            mark = -1.0
        game_matrix = self.two_d_game_matrix_to_vector(game_matrix)
        if (game_matrix[network_output][0] != 0):
            return None
        else:
            game_matrix[network_output][0] = mark
            return self.vector_to_2d_game_matrix(game_matrix)


















