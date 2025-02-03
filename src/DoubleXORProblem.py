import jax.numpy as jnp
import itertools

# Here double XOR dataset generating methods and accuracy estimate methods are defined

# 0=-1 and 1=1 for practical purposes
def XOR_gate(a, b):
    if a != b:
        return 1
    else:
        return -1

def double_XOR_gate(arr):
    arr1 = arr[0:2]
    arr2 = arr[2:4]
    el1 = XOR_gate(arr1[0], arr1[1])
    el2 = XOR_gate(arr2[0], arr2[1])
    return [el1, el2]

def generate_binary_dataset(length, func):
    """
    Generate a dataset of binary arrays of given length and their corresponding outputs.

    Args:
        length (int): Length of binary input arrays.
        func (callable): Function mapping binary arrays to binary arrays.

    Returns:
        tuple: (inputs, outputs) - both are JAX arrays.
    """
    input_list = jnp.array([jnp.array(seq) for seq in itertools.product([-1, 1], repeat=length)])
    output_list = jnp.array([func(seq) for seq in input_list])
    return input_list, output_list

# Stuart-Landau dataset methods
def double_XOR_SL_map_features_and_labels(feature_multiplier, feature_constant, label_multiplier, inference_parameters, outputn):
    # feature_constant as a number that mitigates internal symmetries is crucial (one can also just use 0 and 1 but it is less plyable)
    if type(inference_parameters) == int:
        amplitude_relative = inference_parameters
    else:
        amplitude_relative = inference_parameters[outputn[0]]

    features, labels = generate_binary_dataset(4, double_XOR_gate)
    features = features * feature_multiplier + feature_constant
    labels = labels * label_multiplier + amplitude_relative
    return amplitude_relative, features, labels

def double_XOR_SL_determine_accuracy(amplitude, label, outputn, amplitude_relative):
    return (
        (amplitude[outputn[0]] > amplitude_relative) == (label[0] > amplitude_relative) and
        (amplitude[outputn[1]] > amplitude_relative) == (label[1] > amplitude_relative)
    )

# Kuramoto dataset methods
def double_XOR_K_map_features_and_labels():
    features, labels = generate_binary_dataset(4, double_XOR_gate)
    return False, features * jnp.pi * 0.5, labels * jnp.pi * 0.5

def double_XOR_K_determine_accuracy(phases, label, outputn):
    # accuracy based on distance
    accuracy_measure = [1-jnp.cos(phases[outputn[0]]-label[0]), 1-jnp.cos(phases[outputn[1]]-label[1])]

    if accuracy_measure[0] < 1 and accuracy_measure[1] < 1:  # Both cosines are closer 
        return True
    else:  # Both cosines are further
        return False

"""
# Kuramoto dataset methods
# check if generating works correctly
# Generate dataset
length = 4  # Change as needed
X, Y = generate_binary_dataset(length, double_XOR_gate)
f, l = double_XOR_SL_dataset(100, 30, 1/3, 0, 0)
print(f)
print(l)
"""