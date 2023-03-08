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
                 rng_seed=2, pop_size=1000, tests_per_model=150,
                 in_nodes=9, out_nodes=9,
                 init_hidden_layer_min_max=(1, 2), init_node_max=9,
                 init_mutation_rate=0.91, top_score_percentage=0.005,
                 init_mutation_deviation = 0.75):
       random.seed(rng_seed)
       np.random.seed(rng_seed)
       self.pop_size = pop_size
       self.init_hidden_layer_min_max = init_hidden_layer_min_max
       self.init_node_max = init_node_max + 1 # Buffer for inclusivity
       self.in_nodes = in_nodes
       self.out_nodes = out_nodes
       self.tests_per_model = tests_per_model
       self.mutation_rate = init_mutation_rate
       self.top_score_percentage = top_score_percentage
       self.best_scores_avg = np.array([])
       self.mutation_scale_deviation = init_mutation_deviation
       self.Kp = 1
       self.Ki = 0.0001
       self.Kd = 0.0001
       self.accumulated_error = np.array([])
       self.best_score = 0

# %%

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

# %%

    def determine_best_networks(self, test_results):
        """ Returns a list of the top 5% success rate in a generation

            Parameters:
            test_results -- an array containing test results whose pattern
                            matches that of network_list
        """
        self.top_performers_list = []
        top_performers_index_array = test_results.argsort()
        test_results.sort()
        network_array = np.array(self.network_list)
        self.best_scores = []
        for index in range(floor(self.pop_size * self.top_score_percentage)):
            self.top_performers_list.append(
                    network_array[top_performers_index_array][-(index + 1)])
            self.best_scores.append(test_results[-(index + 1)])
        self.update_best_scores_avg()

# %%

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
                    mutation_scale = random.gauss(
                        1, self.mutation_scale_deviation)
                    flattened_weights[index_to_mutate] *= mutation_scale
                    mutated_index_list.append(index_to_mutate)
                    edit_count += 1
            new_network.network_weights[index0] = flattened_weights.reshape(
                    weight_shape)
        return new_network

# %%

    def evolve_networks(self, test_results):
        """ Updates the network list based on the test results

            Keywords:
            test_results -- see self.determine_best_networks
        """
        self.determine_best_networks(test_results)
        children_per_network = round(self.pop_size
                                     / len(self.top_performers_list))
        print(f"Top 5 Scores:\t{self.best_scores[0:5]}")
        print(f"High Score Avg:\t {self.best_scores_avg[-1]:.3f}")
        print(f"High Score Variance:\t{self.get_best_scores_variance():.3f}")
        print("")
        new_network_list = []
        for network in self.top_performers_list:
            for i in range(children_per_network):
                new_network_list.append(self.mutate_network(network))
        for network in self.network_list:
            del network
        self.network_list = new_network_list

# %%

    def breed_networks(self, network1, network2):
        """ Returns a neural network with weights inherited from two parent
            networks.
        """
        network_weight_array_shape = np.shape(network1.network_weights)
        new_network_weights = np.zeros_like(network1.network_weights)
        for i in range(network_weight_array_shape[0]):
            for j in range(network_weight_array_shape[1]):
                for k in range(network_weight_array_shape[2]):
                    if self.coinflip() == 0:
                        new_network_weights[i][j][k] = network1.network_weights[i][j][k]
                    else:
                        new_network_weights[i][j][k] = network2.network_weights[i][j][k]
        return nn.NeuralNet([9, 9, 9], network_weights=new_network_weights)

# %%

    def evolve_networks_with_genetics(self, test_results):
        """ Updates the network list based on the test results

            Keywords:
            test_results -- see self.determine_best_networks
        """
        self.determine_best_networks(test_results)
        print(f"Top 5 Scores:\t{self.best_scores[0:5]}")
        print(f"High Score Avg:\t {self.best_scores_avg[-1]:.3f}")
        print(f"High Score Variance:\t{self.get_best_scores_variance():.3f}")
        print("")
        new_network_list = []
        child_network_list = []
        for i in range(len(self.top_performers_list)):
            for network in self.top_performers_list[i:]:
                child_network_list.append(
                    self.breed_networks(self.top_performers_list[i], network))
        children_per_network = round(self.pop_size
                                     / len(child_network_list))
        for network in child_network_list:
            for i in range(children_per_network):
                new_network_list.append(self.mutate_network(network))
        for network in self.network_list:
            del network
        self.network_list = new_network_list

# %%

    def coinflip(self):
        """ Returns either a 0 or 1, randomly"""
        return random.randrange(2)

# %%

    def save_best_network_overall(self):
        """ Saves a copy of the highest scoring network from the
            entirety of testing
        """
        if self.best_scores[0] > self.best_score:
            self.best_score = self.best_scores[0]
            self.best_network = self.top_performers_list[0]


# %%

    def get_score_delta(self):
        """ Returns the change in the average of self.best_scores"""
        return self.best_scores_avg[-1] - self.best_scores_avg[-2]

# %%

    def update_best_scores_avg(self):
        """ Updates self.best_scores_avg """
        self.best_scores_avg = np.append(self.best_scores_avg,
                                         np.mean(self.best_scores))

# %%

    def update_mutation_scale_deviation(self):
        """ Updates self.mutation_scale_deviation """
        error = self.error()
        self.mutation_scale_deviation = self.Kp * error

# %%

    def error(self):
        """ Returns the error """
        return 1 - self.best_scores_avg[-1]

# %%

    def set_controller_coeffs(self):
        """ Update the controller coeffs via user input"""
        print("Current vals -- Kp={a}, Ki={b}".format(
            a=self.Kp, b=self.Ki))
        self.Kp = float(input("New Kp:  "))
        self.Ki = float(input("New Ki:  "))
        #self.Kd = float(input("New Kd:  "))

# %%

    def get_best_scores_variance(self):
        """ Returns the variance of the set of best_scores """
        return np.var(self.best_scores)

