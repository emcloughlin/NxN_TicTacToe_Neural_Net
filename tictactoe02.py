'''
2/28/23 

'''

import json
import random
import numpy as np
import tensorflow as tf

# Define the file name to use for the dataset
file_name = 'dataset.json'

# Define the size of the dataset to generate
data_size = 1000

# Define the symbols for each player
player_symbols = ['X', 'O']

# Define the empty board symbol
empty_symbol = '.'

# Define the symbol mapping
symbol_mapping = {empty_symbol: 0, player_symbols[0]: -1, player_symbols[1]: 1}

# Define the winning patterns
winning_patterns = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

# Define the get_winner function
def get_winner(board, symbol_mapping, winning_patterns):
    for pattern in winning_patterns:
        symbols = [board[i] for i in pattern]
        if len(set(symbols)) == 1 and symbols[0] != symbol_mapping[empty_symbol]:
            return symbol_mapping[symbols[0]]

# Generate the dataset
dataset = []
for i in range(data_size):
    # Create an empty board
    board = [empty_symbol] * 9

    # Randomly select the starting player
    player = random.choice(player_symbols)

    # Play until the game is over
    while True:
        # Find all the empty cells
        empty_cells = [i for i, symbol in enumerate(board) if symbol == empty_symbol]

        # Check if the game is over
        if not empty_cells or get_winner(board, symbol_mapping, winning_patterns):
            break

        # Select a random empty cell
        cell_index = random.choice(empty_cells)

        # Set the cell to the player's symbol
        board[cell_index] = player

        # Switch to the other player
        player = player_symbols[(player_symbols.index(player) + 1) % 2]

    # Convert the board to the input list
    input_list = [symbol_mapping[symbol] for symbol in board]

    # Convert the winner to the target list
    winner = get_winner(board, symbol_mapping, winning_patterns)
    target_list = [1 if winner == symbol_mapping[symbol] else 0 for symbol in board]

    # Add the data point to the dataset
    dataset.append((input_list, target_list))

# Shuffle the dataset
random.shuffle(dataset)

# Split the dataset into training and testing sets
split_index = int(0.8 * len(dataset))
train_data = dataset[:split_index]
test_data = dataset[split_index:]

# Convert the dataset to strings and store it in a JSON file
train_data_str = [json.dumps(data) for data in train_data]
test_data_str = [json.dumps(data) for data in test_data]
with open(file_name, 'w') as f:
    json.dump({'train': train_data_str, 'test': test_data_str}, f)

# Load the dataset from the JSON file and convert it back to Python objects
with open(file_name, 'r') as f:
    data = json.load(f)
train_data_str, test_data_str = data['train'], data['test']
train_data = [json.loads(data_str) for data_str in train_data_str]
test_data = [json.loads(data_str) for data_str in test_data_str]




'''
==============================================================================
Train the neural network.
==============================================================================
'''


# Load data from JSON file
with open('dataset.json', 'r') as f:
    data = json.loads(f.read())

# Convert string inputs and targets to arrays
for d in data:
    d['input_list'] = np.array(d['input_list'])
    d['target_list'] = np.array(d['target_list'])


# Split data into input and target arrays
input_data = np.array([d['input_list'] for d in data])
target_data = np.array([d['target_list'] for d in data])

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(9, input_shape=(9,), activation='relu'),
    tf.keras.layers.Dense(9, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model on data
model.fit(input_data, target_data, epochs=50, batch_size=32, validation_split=0.2)

# Save the model to a file
model.save('tictactoe_model.h5')

























