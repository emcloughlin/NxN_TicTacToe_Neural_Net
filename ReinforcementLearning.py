"""
ReinforcementLearning.py
Author: Evan McLoughlin
Created: 2.26.2023
"""
# %%

import numpy as np
import NeuralNet as nn
import random
from math import floor
from copy import deepcopy

# %%

class ReinforcementLearning:

    def __init__(self,
                 rng_seed=2, pop_size=500, tests_per_model=200,
                 in_nodes=9, out_nodes=9,
                 init_hidden_layer_min_max=(1, 2), init_node_max=9,
                 mutation_rate=0.5, top_score_percentage=0.05):
       random.seed(rng_seed)
       np.random.seed(rng_seed)
       self.pop_size = pop_size
       self.init_hidden_layer_min_max = init_hidden_layer_min_max
       self.init_node_max = init_node_max + 1 # Buffer for inclusivity
       self.in_nodes = in_nodes
       self.out_nodes = out_nodes
       self.tests_per_model = tests_per_model
       self.mutation_rate = mutation_rate
       self.top_score_percentage = top_score_percentage

    def create_gen_1(self):
        self.network_list = []
        self.network_descriptor_list = []
        for i in range(self.pop_size):
            # Set the number of input nodes in each network descriptor
            network_descriptor = [self.in_nodes]
            for j in range(np.random.randint(self.init_hidden_layer_min_max[0],
                                             self.init_hidden_layer_min_max[1])):
                # Set the number of hidden layer nodes
                network_descriptor.append(np.random.randint(low=9,
                    high=self.init_node_max))
            # Set the number of output nodes
            network_descriptor.append(self.out_nodes)
            # Populate the network_list and descriptor_list
            self.network_descriptor_list.append(network_descriptor)
            self.network_list.append(nn.NeuralNet(network_descriptor, i))

    def determine_best_networks(self, test_results):
        """ Returns a list of the top 5% success rate in a generation

            Parameters:
            test_results -- an array containing test results whose pattern
                            matches that of network_list
        """
        top_performers_list = []
        top_performers_index_array = test_results.argsort()
        test_results.sort()
        network_array = np.array(self.network_list)
        self.best_scores = []
        for index in range(floor(self.pop_size * self.top_score_percentage)):
            top_performers_list.append(
                    network_array[top_performers_index_array][-(index + 1)])
            if (index < 5):
                self.best_scores.append(test_results[-(index + 1)])
        return top_performers_list

    def mutate_network(self, old_network):
        """ Returns a network with weights that based off of an old model,
            but with random alterations

            Parameters:
            old_network -- the network to base the new weights off of
        """
        new_network = deepcopy(old_network)
        num_weight_sets = np.shape(new_network.network_weights)[0]
        for index0 in range(num_weight_sets):
            weight_shape = np.shape(new_network.network_weights[index0])
            flattened_weights = new_network.network_weights[index0].flatten()
            num_weights_this_set = np.size(flattened_weights)
            num_elements_to_mutate = floor(num_weights_this_set
                                           * self.mutation_rate)
            mutated_index_list = []
            edit_count = 0
            while(edit_count < num_elements_to_mutate):
                index_to_mutate = random.randrange(np.size(flattened_weights))
                if (index_to_mutate in mutated_index_list):
                    pass
                else:
                    mutation_scale = random.gauss(0, 0.25)
                    flattened_weights[index_to_mutate] *= mutation_scale
                    mutated_index_list.append(index_to_mutate)
                    edit_count += 1
            new_network.network_weights[index0] = flattened_weights.reshape(
                    weight_shape)
        return new_network

    def evolve_networks(self, test_results):
        """ Updates the network list based on the test results

            Keywords:
            test_results -- see self.determine_best_networks
        """
        top_performers_list = self.determine_best_networks(test_results)
        print("Top 5% of Scores:\n{temp}".format(temp=self.best_scores))
        children_per_network = round(len(top_performers_list)
                                     / self.top_score_percentage)
        new_network_list = []
        for network in top_performers_list:
            for i in range(children_per_network):
                new_network_list.append(self.mutate_network(network))
        for network in self.network_list:
            del network
        self.network_list = new_network_list

