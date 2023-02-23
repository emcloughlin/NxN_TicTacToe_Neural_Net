"""
TicTacToe_Solver.py
Author: Evan McLoughlin
Created: 2.23.2023
"""
# %%
import numpy as np
from numpy.typing import ArrayLike
import random

# %%

# A function to return an n x n matrix to represent the game board
def init_game_matrix(dimensions: int) -> np.ndarray:
    return np.zeros((dimensions, dimensions))

# %%

# A function to print the game board to the console
def print_game_matrix(game_matrix: ArrayLike):
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
def check_end_state(game_matrix: ArrayLike):
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

# A function to play tic-tac-toe with no strategy (random input)
def ttt_random_move(game_matrix: ArrayLike, play_as_x: bool) -> np.ndarray:
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

# A function to play tic-tac-toe
def tic_tac_toe(dimensions: int, display_game_board: bool):
    game_matrix = init_game_matrix(dimensions)
    if (display_game_board):
        print("Initial Board:")
        print_game_matrix(game_matrix)
    game_state = -1.0 # The condition used to specify an unfinished game
    turn = 0
    while (game_state == -1.0):
        if (turn % 2 == 0):
            x_turn = True
        else:
            x_turn = False
        game_matrix = ttt_random_move(game_matrix, x_turn)
        if (display_game_board):
            print("Turn #{turn_num}:".format(turn_num = turn + 1))
            print_game_matrix(game_matrix)
        game_state = check_end_state(game_matrix)
        turn += 1
    print("Game Result:")
    print("Draw.") if (game_state == 0.0) else print("X wins!") if (game_state == 1.0) else print("O wins!")

# %%























# %%
