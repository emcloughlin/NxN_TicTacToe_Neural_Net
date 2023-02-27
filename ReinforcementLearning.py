"""
ReinforcementLearning.py
Author: Evan McLoughlin
Created: 2.26.2023
"""
# %%

import numpy as np
import NeuralNet as nn
import random
# %%

class ReinforcementLearning:

    def __init__(self, rng_seed=1, pop_size=100, in_nodes=9, out_nodes=9,
                 init_hidden_layer_min_max=(1, 4), init_node_max=13):
       random.seed(rng_seed)
       np.random.seed(rng_seed)
       self.pop_size = pop_size + 1  # Buffer for inclusivity
       self.init_hidden_layer_min_max = init_hidden_layer_min_max
       self.init_node_max = init_node_max
       self.in_nodes = in_nodes
       self.out_nodes = out_nodes

    def create_gen_1(self):
        self.network_list = []
        self.network_descriptor_list = []
        for i in range(self.pop_size):
            # Set the number of input nodes in each network descriptor
            network_descriptor = [self.in_nodes]
            for j in range(np.random.randint(self.init_hidden_layer_min_max[0],
                                             self.init_hidden_layer_min_max[1])):
                # Set the number of hidden layer nodes
                network_descriptor.append(np.random.randint(low=3, high=self.init_node_max))
            # Set the number of output nodes
            network_descriptor.append(self.out_nodes)
            # Populate the network_list and descriptor_list
            self.network_descriptor_list.append(network_descriptor)
            self.network_list.append(nn.NeuralNet(network_descriptor, i))


















