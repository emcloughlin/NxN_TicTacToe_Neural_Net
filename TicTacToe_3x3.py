# %%
import numpy as np
# %%


# %%

# A function to return an n x n matrix to represent the game board
def init_game_matrix(dimensions = int):
    return np.zeros((3,3))

# %%

# A function to print the game board to the console
def print_game_matrix(game_matrix):

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
def check_end_state(game_matrix):

    # TODO: I finished coding this and then had to send my laptop in for
    # repairs, so this actually needs to be tested lol

    n = np.shape(game_matrix)[0] / 1 # Check the shape of the matrix to determine win number (and convert to float for simplicity)
    draw = 0.0
    x_win = 1.0
    o_win = 2.0
    no_end = -1.0

    # Check diags for win state
    if (np.sum(np.diag(game_matrix)) == n or np.sum(np.diag(game_matrix.T)) == n):
        return x_win
    if (np.sum(np.diag(game_matrix)) == -n or np.sum(np.diag(game_matrix.T)) == -n):
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
    return draw # If nothing else is rerturned, a draw is present







