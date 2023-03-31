# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:27:22 2023

@author: Andrew Fowler
"""

import numpy as np
import pandas as pd

def initializeBoard():
    return(np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]))

def check_for_winner(x):
    gameOver = False
    winner = None

    if 3 in np.dot(x, np.array([1,1,1])):
        gameOver = True
        winner = 1

    if -3 in np.dot(x, np.array([1,1,1])):
         gameOver = True
         winner = -1

    #Checks for vertical win
    if 3 in np.dot(x.transpose(), np.array([1,1,1])):
        gameOver = True
        winner = 1

    if -3 in np.dot(x.transpose(), np.array([1,1,1])):
         gameOver = True
         winner = -1

    #Checks for diagonal win
    if (np.dot(x.diagonal(), [1,1,1]) == 3) or (np.dot(np.fliplr(x).diagonal(), [1,1,1]) == 3):
        gameOver = True
        winner = 1

    if (np.dot(x.diagonal(), [1,1,1]) == -3) or (np.dot(np.fliplr(x).diagonal(), [1,1,1]) == -3):
        gameOver = True
        winner = -1

    return(gameOver, winner)

def check_for_winning_move(x):
    hasWinningMove = False
    winningMove = None

    for i in np.arange(3):
        if x[i].dot([1,1,1]) == 2.01:
            hasWinningMove = True
            squareToPlay = int(np.where(x[i] != 1)[0])
            winningMove = np.array([i, squareToPlay])
            return(hasWinningMove, winningMove)

    for i in np.arange(3):
        if x.T[i].dot([1,1,1]) == 2.01:
            hasWinningMove = True
            squareToPlay = int(np.where(x.T[i] != 1)[0])
            winningMove = np.array([squareToPlay, i])
            return(hasWinningMove, winningMove)

    for i in np.arange(2):
        if x.diagonal().dot([1,1,1]) == 2.01:
            hasWinningMove = True
            squareToPlay = int(np.where(x.diagonal() != 1)[0])
            winningMove = np.array([squareToPlay, squareToPlay])
            return(hasWinningMove, winningMove)
        if np.fliplr(x).diagonal().dot([1,1,1]) == 2.01:
            hasWinningMove = True
            squareToPlay = int(np.where(np.fliplr(x).diagonal() != 1)[0])
            winningMove = np.array([squareToPlay, 2 - squareToPlay])
            return(hasWinningMove, winningMove)

    return(hasWinningMove, winningMove)

def check_for_opponent_winning_move(x):

    hasWinningMove = False
    savingMove = None

    for i in np.arange(3):
        if x[i].dot([1,1,1]) == -1.99:
            hasWinningMove = True
            squareToPlay = int(np.where(x[i] != -1)[0])
            savingMove = np.array([i, squareToPlay])
            return(hasWinningMove, savingMove)

    for i in np.arange(3):
        if x.T[i].dot([1,1,1]) == -1.99:
            hasWinningMove = True
            squareToPlay = int(np.where(x.T[i] != -1)[0])
            savingMove = np.array([squareToPlay, i])
            return(hasWinningMove, savingMove)

    for i in np.arange(2):
        if x.diagonal().dot([1,1,1]) == -1.99:
            hasWinningMove = True
            squareToPlay = int(np.where(x.diagonal() != -1)[0])
            savingMove = np.array([squareToPlay, squareToPlay])
            return(hasWinningMove, savingMove)

        if np.fliplr(x).diagonal().dot([1,1,1]) == -1.99:
            hasWinningMove = True
            squareToPlay = int(np.where(np.fliplr(x).diagonal() != -1)[0])
            savingMove = np.array([squareToPlay, 2 - squareToPlay])
            return(hasWinningMove, savingMove)

    return(hasWinningMove, savingMove)

def check_for_fork(x):

    hasFork = False
    forkMove = None

    orientation1 = x
    orientation2 = np.fliplr(x.T)
    orientation3 = np.flip(x)
    orientation4 = np.fliplr(x).T

    orientations = np.array([orientation1, orientation2, orientation3, orientation4])

    for i in np.arange(4):
        if orientations[i][0,1] == orientations[i][1,0] == 1:
            if orientations[i][1,1] == orientations[i][1,2] == orientations[i][2,1] != -1:
                hasFork = True
                forkMove = np.array([1,1])
                break
            if orientations[i][0,0] == orientations[i][0,2] == orientations[i][2,0] != -1:
                hasFork = True
                forkMove = np.array([0,0])
                break

        if orientations[i][2,0] == orientations[i][0,2] == 1:
            if orientations[i][0,0] == orientations[i][1,0] == orientations[0,1] != -1:
                hasFork = True
                forkMove = np.array([0,0])
                break

        if orientations[i][0,0] == orientations[i][1,2] == 1:
            if orientations[i][1,0] == orientations[i][1,1] == orientations[i][2,2] != -1:
                hasFork = True
                forkMove = np.array([1,1])
                break

        if orientations[i][0,1] == orientations[i][2,0] == 1:
            if orientations[i][1,1] == orientations[i][2,1] == orientations[i][2,2] != -1:
                hasFork = True
                forkMove = np.array([2,1])
                break

    if hasFork == True:
        if i == 0:
            return(hasFork, forkMove)
        if i == 1:
            if forkMove.all() == np.array([0,0]).all():
                forkMove = np.array([2,0])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([2,1]).all():
                forkMove == np.array([2,2])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([1,1]).all():
                return(hasFork, forkMove)
        if i == 2:
            forkMove = np.flip(forkMove)
            return(hasFork, forkMove)
        if i == 3:
            if forkMove.all() == np.array([0,0]).all():
                forkMove = np.array([0,2])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([2,1]).all():
                forkMove == np.array([1,0])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([1,1]).all():
                return(hasFork, forkMove)

    return(hasFork, forkMove)

def check_for_opponent_fork(x):
    hasFork = False
    forkMove = None

    orientation1 = x
    orientation2 = np.fliplr(x.T)
    orientation3 = np.flip(x)
    orientation4 = np.fliplr(x).T

    orientations = np.array([orientation1, orientation2, orientation3, orientation4])

    for i in np.arange(4):
        if orientations[i][0,1] == orientations[i][1,0] == -1:
            if orientations[i][1,1] == orientations[i][1,2] == orientations[i][2,1] != 1:
                hasFork = True
                forkMove = np.array([1,1])
                break
            if orientations[i][0,0] == orientations[i][0,2] == orientations[i][2,0] != 1:
                hasFork = True
                forkMove = np.array([0,0])
                break

        if orientations[i][2,0] == orientations[i][0,2] == -1:
            if orientations[i][0,0] == orientations[i][1,0] == orientations[0,1] != 1:
                hasFork = True
                forkMove = np.array([0,0])
                break

        if orientations[i][0,0] == orientations[i][1,2] == -1:
            if orientations[i][1,0] == orientations[i][1,1] == orientations[i][2,2] != 1:
                hasFork = True
                forkMove = np.array([1,1])
                break

        if orientations[i][0,1] == orientations[i][2,0] == -1:
            if orientations[i][1,1] == orientations[i][2,1] == orientations[i][2,2] != 1:
                hasFork = True
                forkMove = np.array([2,1])
                break

    if hasFork == True:
        if i == 0:
            return(hasFork, forkMove)
        if i == 1:
            if forkMove.all() == np.array([0,0]).all():
                forkMove = np.array([2,0])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([2,1]).all():
                forkMove == np.array([2,2])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([1,1]).all():
                return(hasFork, forkMove)
        if i == 2:
            forkMove = np.flip(forkMove)
            return(hasFork, forkMove)
        if i == 3:
            if forkMove.all() == np.array([0,0]).all():
                forkMove = np.array([0,2])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([2,1]).all():
                forkMove == np.array([1,0])
                return(hasFork, forkMove)
            if forkMove.all() == np.array([1,1]).all():
                return(hasFork, forkMove)

    return(hasFork, forkMove)

def make_random_move(x):

    newMove = np.random.randint(0,3,2)

    while x[newMove[0], newMove[1]] != 0.01:
        newMove = np.random.randint(0,3,2)

    return(newMove)

def nTicTacToe(n, player = 0): #player = 0 for X and 1 for O

    allData = pd.DataFrame()

    for i in np.arange(n):

        #No winner at start
        gameOver = False

        #Fresh board - Blank space = 0.01 instead of 0 so the dot product doesn't equal 0 at the start
        gameSim = initializeBoard()

        positionLog = pd.DataFrame()
        positionLog = pd.concat([positionLog, pd.DataFrame({"Position": [gameSim.flatten().copy()]})], ignore_index = True)
        positionToCompare = pd.DataFrame()

        moveNum = 0

        while(gameOver == False):

            #Checks for draw
            if (.01 in gameSim) == False:
                winner = 0.25
                break

            if moveNum % 2 == 0:
                hasWinningMove, winningMove = check_for_winning_move(gameSim)

                if hasWinningMove == True:
                    gameSim[winningMove[0], winningMove[1]] = 1

                if hasWinningMove == False:
                    opponentCanWin, savingMove = check_for_opponent_winning_move(gameSim)
                    if opponentCanWin == True:
                        gameSim[savingMove[0], savingMove[1]] = 1

                    if opponentCanWin == False:
                        hasFork, forkMove = check_for_fork(gameSim)
                        if hasFork == True:
                            gameSim[forkMove[0], forkMove[1]] = 1

                        if hasFork == False:
                            opponentHasFork, savingMove = check_for_opponent_fork(gameSim)
                            if opponentHasFork == True:
                                gameSim[savingMove[0], savingMove[1]] = 1

                            if opponentHasFork == False:
                                randomMove = make_random_move(gameSim)
                                gameSim[randomMove[0], randomMove[1]] = 1

            if moveNum % 2 == 1:
                randomMove = make_random_move(gameSim)
                gameSim[randomMove[0], randomMove[1]] = -1

            moveNum += 1

            #Saves snapshot of board
            if (moveNum % 2) == player:
                positionLog = pd.concat([positionLog, pd.DataFrame({"Position": [gameSim.flatten().copy()]})], ignore_index = True)

            if (moveNum % 2) != player:
                positionToCompare = pd.concat([positionToCompare, pd.DataFrame({"Position": [gameSim.flatten().copy()]})], ignore_index = True)

            gameOver, winner = check_for_winner(gameSim)

            if winner == -1:
                positionLog.Position = positionLog.Position.drop(len(positionLog.Position) - 1)
                positionLog = positionLog.dropna()

        if winner == 1:
            outcomeCol = pd.DataFrame()
            for j in np.arange(len(positionLog.Position)):
                outcomeCol = pd.concat([outcomeCol, pd.DataFrame({"Outcome": [positionLog.Position[j] == positionToCompare.Position[j]]})], ignore_index = True)

            for k in np.arange(len(outcomeCol.Outcome)):
                outcomeCol.Outcome[k] = np.where(outcomeCol.Outcome[k] == True, 0, winner)

            positionLog["Outcome"] = outcomeCol

            allData = pd.concat([allData, positionLog], ignore_index = True)

    return(allData)


# Define the neural network architecture
n_input = 9
n_hidden = 36
n_output = 9

# Initialize the weights and biases for each layer
W1 = np.random.randn(n_input, n_hidden) * 0.01
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * 0.01
b2 = np.zeros((1, n_output))

def forward_propagation(X):
    # Compute the output of the first layer
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1) # ReLU activation function

    # Compute the output of the second layer
    Z2 = np.dot(A1, W2) + b2
    exp_Z2 = np.exp(Z2)
    A2 = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True) # Softmax activation function

    # Store the intermediate values in a cache for backpropagation
    cache = (X, A1, Z1, W1, b1, Z2, A2, W2, b2)

    return A2, cache

def cross_entropy_loss(Y, Y_hat):
    m = Y.shape[0] # Number of samples
    loss = -1/m * np.sum(Y * np.log(Y_hat))

    return loss

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

def backpropagation(Y, Y_hat, cache):
    X, A1, Z1, W1, b1, Z2, A2, W2, b2 = cache
    m = Y.shape[0] # Number of samples
    # Compute the derivative of the loss with respect to the output of the second layer
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(A1.T, dZ2)
    db2 = 1/m * np.sum(dZ2, axis=0, keepdims=True)

    # Compute the derivative of the loss with respect to the output of the first layer
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = relu_backward(dA1, Z1)
    dW1 = 1/m * np.dot(X.T, dZ1)
    db1 = 1/m * np.sum(dZ1, axis=0, keepdims=True)

    # Update the weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    return W1, b1, W2, b2

# Generate some sample data
trainData = nTicTacToe(1000)
X = np.stack(trainData.Position)
Y = np.stack(trainData.Outcome)

# Hyperparameters
learning_rate = 1
iterations = 3000

# Train the neural network
for i in range(iterations):
    # Forward propagation
    Y_hat, cache = forward_propagation(X)

    # Compute the loss
    loss = cross_entropy_loss(Y, Y_hat)

    # Backpropagation
    W1, b1, W2, b2 = backpropagation(Y, Y_hat, cache)

    # Print the loss every 10 epochs
    if i % 20 == 0:
       print(f"Iteration {i}: Loss = {loss}")



def playAIMove(board):
    board = board.flatten()
    AIMove, cache = forward_propagation(board)
    moveToPlay = AIMove.argmax()
    illegalMove = False
    if board[moveToPlay] != 0.01:
        illegalMove = True
    while illegalMove == True:
        AIMove[0,AIMove.argmax()] = 0
        moveToPlay = AIMove.argmax()
        if board[moveToPlay] == 0.01:
            illegalMove = False
    board[moveToPlay] = 1
    board = board.reshape(3,3)
    print(board)
    return(board)































