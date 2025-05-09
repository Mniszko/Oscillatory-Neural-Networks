import jax.numpy as jnp
from .StuartLandauNeuralNet import determine_SL_binary_distance
# Studart-Landau dataset methods

# old accuracy determining method
'''
def XOR_problem_SL_determine_accuracy(amplitude, label, outputn, amplitude_relative):
    """
    Function assumes that outputn has one element matching the XOR problem
    """
    if amplitude[outputn[0]] > amplitude_relative and label > amplitude_relative:
        return True
    elif amplitude[outputn[0]] < amplitude_relative and label < amplitude_relative:
        return True
    else:
        return False
'''

# new accuracy determining method
def XOR_problem_SL_determine_accuracy(amplitude, phase, label_amplitude, label_phase, outputn, all_labels_amplitude, all_labels_phase, cost_mix_type):
    """
    Determines accuracy (1 if correct prediction has lower loss, 0 otherwise)
    """
    # Get the alternative label (the incorrect one)
    incorrect_amplitude = all_labels_amplitude[1] if jnp.array_equal(label_amplitude, all_labels_amplitude[0]) else all_labels_amplitude[0]
    incorrect_phase = all_labels_phase[1] if jnp.array_equal(label_phase, all_labels_phase[0]) else all_labels_phase[0]

    # Calculate losses
    correct_loss = determine_SL_binary_distance(amplitude[outputn], phase[outputn], label_amplitude, label_phase, outputn, cost_mix_type)
    incorrect_loss = determine_SL_binary_distance(amplitude[outputn], phase[outputn], incorrect_amplitude, incorrect_phase, outputn, cost_mix_type)

    # Return 1 if correct loss is smaller than incorrect loss, 0 otherwise
    return (correct_loss < incorrect_loss).astype(int)

def XOR_problem_SL_map_features_and_labels(feature_multiplier, feature_constant, label_multiplier, amplitudes, outputn):
    """
    Function assumes that outputn has one element matching the XOR problem
    """
    if type(amplitudes) == int:
        amplitude_relative = amplitudes
    else:
        amplitude_relative = amplitudes[outputn[0]]

    features = jnp.array([
        [0,0],
        [1,0],
        [0,1],
        [1,1]
    ])*feature_multiplier+feature_constant
    labels_amplitude = jnp.array([[-1],[1],[1],[-1]]) * label_multiplier + amplitude_relative
    labels_phase = jnp.array([[-jnp.pi],[jnp.pi],[jnp.pi],[-jnp.pi]])/2 # the same as in Kuramoto
    return amplitude_relative, features, labels_amplitude, labels_phase

# Kuramoto dataset methods
def XOR_problem_K_map_features_and_labels():
    features = jnp.array([
            [-jnp.pi,-jnp.pi],
            [jnp.pi,-jnp.pi],
            [-jnp.pi,jnp.pi],
            [jnp.pi,jnp.pi]
        ])/2
    labels = jnp.array([[-jnp.pi],[jnp.pi],[jnp.pi],[-jnp.pi]])/2
    return features, labels

def XOR_problem_K_determine_accuracy(phases, label, outputn):
    # accuracy based on distance
    accuracy_measure = 1-jnp.cos(phases[outputn[0]]-label[0])

    if accuracy_measure < 1:
        return True
    else:
        return False
