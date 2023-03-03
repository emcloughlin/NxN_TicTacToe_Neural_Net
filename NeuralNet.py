"""
NeuralNet.py
Author: Evan McLoughlin
Created: 2.23.2023
"""
# %%

import numpy as np
from numpy.typing import ArrayLike
# %%

class NeuralNet:

    def __init__(self, network_descriptor: list, rng_seed=1,
                 network_weights=None):
        """ Initialize the neural network """
        np.random.seed(rng_seed)
        self.network_descriptor = network_descriptor
        if (network_weights == None):
            self.network_weights = self.initialize_network_weights()
        else:
            self.network_weights = network_weights

    def normalization(self, x: ArrayLike, derivative=False) -> np.ndarray:
        """ Returns result of normalizing the input using either a sigmoid
            or its derivative

            Parameters:
            x -- input to be normalized
            derivative -- if true, use the sigmoid's derivative instead
        """
        if (derivative):
            return np.exp(-x) / ((np.exp(-x) + 1)**2)
        return 1 / (1 + np.exp(-x))

    def random_weights_array(self, array_shape):
        """ Returns an array of randomized weights """
        return 2 * np.random.random(array_shape) - 1

    def initialize_network_weights(self):
        """ Returns a list of weight arrays for each step in the network

            Parameters:
            network_descriptor -- a list n elements corresponding to a network
                                  consisting of n layers where the size of the
                                  nth element is the number of nodes in the
                                  nth layer
        """
        network_weight_list = []
        for index in range(len(self.network_descriptor) - 1):
            weight_shape = (self.network_descriptor[index + 1],
                            self.network_descriptor[index])
            network_weight_list.append(self.random_weights_array(weight_shape))
        return network_weight_list

    def singularize_output(self, network_vector: ArrayLike) -> int:
        """ Returns the index of the vectorized game board to mark """
        index = 0
        i = 0
        biggest_element = 0
        for element in network_vector.flatten():
            if (element > biggest_element):
                biggest_element = element
                index = i
            i += 1
        return index

    def network_function(self, input_vector: ArrayLike):
        """ Returns the result of applying the model to an input vector """
        network_vector = input_vector
        for element in self.network_weights:
            network_vector = self.normalization(np.dot(element, network_vector))
        return self.singularize_output(network_vector)






