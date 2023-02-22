# %%
import numpy as np
# %%


# %%

# Define a function to return an n x n matrix to represent the game board
def init_game_matrix(dimensions = int):
    return np.zeros((3,3))

# %%

# Define a function to print the game board to the console
def print_game_matrix(game_matrix):

    for row in game_matrix:
        temp = ""
        for item in row:
            if (item > 0.5):
                temp = temp + "x"
            elif (item < 0.5):
                temp = temp + "o"
            else:
                temp = temp + "-"
            temp = temp + " "
        print(temp)
