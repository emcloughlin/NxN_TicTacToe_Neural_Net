"""
NeuralNet_Class.py
Author: Evan McLoughlin
Created: 2.23.2023

Author's Note: A lot of this is based off of the sample code provided
"""
# %%

import numpy as np
from numpy.typing import ArrayLike
# %%

class NeuralNet:

    def __init__(self, network_descriptor: list, rng_seed=1):
        np.random.seed(rng_seed)
        self.network_descriptor = network_descriptor
        self.network_weights = self.initialize_network_weights()

    # A function to normalize output; for now, just a sigmoid
    def normalization(self, x: ArrayLike, derivative=False) -> np.ndarray:
        if (derivative):
            return np.exp(-x) / ((np.exp(-x) + 1)**2)
        return 1 / (1 + np.exp(-x))

    # A function to return randomized weights for the neural net
    def random_weights_array(self, array_shape):
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

    def network_function(self, input_vector: ArrayLike):
        """ Returns the result of applying the model to an input vector """
        network_vector = input_vector
        for element in self.network_weights:
            network_vector = self.normalization(np.dot(element, network_vector))
        return network_vector

    def update_weights








