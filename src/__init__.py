from .KuramotoNeuralNetwork import run_function_pulsed, solve_K_ode_free, solve_K_ode_nudged, compute_kuramoto_gradients, determine_K_binary_distance, main_K_training_preamble
# solve forward and solve backward should be named differently, though not the same as SL functions

from .basics import read_and_plot, clear_csv, save_array_to_file, append_to_csv, save_parameters, load_parameters, record_states, write_separator

from .StuartLandauNeuralNet import solve_SL_ode_free, solve_SL_ode_nudged, sum_and_divide_array, main_SL_training_preamble, shuffle_and_batch, determine_SL_binary_distance

from .WeightsModule import create_random_connections, create_square_lattice_connections, create_weight_update_mask, create_kuramoto_symmetric_weights

from .XORProblem import XOR_problem_SL_determine_accuracy, XOR_problem_SL_map_features_and_labels, XOR_problem_K_map_features_and_labels, XOR_problem_K_determine_accuracy

from .DoubleXORProblem import double_XOR_SL_map_features_and_labels, double_XOR_SL_determine_accuracy, double_XOR_K_map_features_and_labels, double_XOR_K_determine_accuracy

"""
This module consists of methods used for running and training Stuart-Landau oscillatory neural networks. Input data has to be passed via bias term (u) and output is defined as amplitudes (or densities, which are the same).

There are also methods for array manipulation used for parameter initialization and parameter updates.

Method for mapping features and labels (map_features_and_labels(init_amplitudes, outputn) which gives out relative amplitude, features and labels) and assigning the dataset has to be  defined in main file, as it depends on problem that is being solved. One can use one of given example methods and copy it under appropreate name.
"""


# Submodules (classes) are not currently in use, below is an example for possible future use
class StuartLandauNeuralNet:
    def __init__(self):
        self.solve_SL_ode_free = solve_SL_ode_free

stuartLandauNeuralNet = StuartLandauNeuralNet()
