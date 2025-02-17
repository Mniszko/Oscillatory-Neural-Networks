import jax.numpy as jnp

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
def XOR_problem_SL_determine_accuracy(amplitude, label, outputn, amplitude_relative):
    return (
        (amplitude[outputn[0]] > amplitude_relative) == (label[0] > amplitude_relative)
    )

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
    labels = jnp.array([[-1],[1],[1],[-1]]) * label_multiplier + amplitude_relative
    return amplitude_relative, features, labels

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
