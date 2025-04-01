import jax
import jax.numpy as jnp
from jax.scipy.linalg import eigh
from .WeightsModule import create_weight_update_mask

'''
def create_symmetric_weights(N, loc, scale, inputn, rng_key):
    """
    Create a symmetric weight matrix with random values, ensuring weights between neurons in inputn are zero.

    :param N: int, number of neurons
    :param loc: float, mean of the normal distribution
    :param scale: float, standard deviation of the normal distribution
    :param inputn: list or array, indices of neurons whose weights must be zero
    :param rng_key: jax.random.PRNGKey, random key for reproducibility
    :return: jnp.array, symmetric weight matrix
    """
    # Generate random numbers using jax.random
    random_values = jax.random.normal(rng_key, shape=(N, N)) * scale + loc
    # Create a lower triangular matrix
    lower_triangular = jnp.tril(random_values, k=-1)
    # Reflect it to make it symmetric
    symmetric_matrix = lower_triangular + lower_triangular.T
    # Set the diagonal to zero
    symmetric_matrix = symmetric_matrix.at[jnp.diag_indices(N)].set(0)
    # Set weights between neurons in inputn to zero
    for i in inputn:
        symmetric_matrix = symmetric_matrix.at[i, inputn].set(0)
        symmetric_matrix = symmetric_matrix.at[inputn, i].set(0)
    return symmetric_matrix
'''

def create_symmetric_weights(N, loc, scale, inputn, rng_key, options=None):
    """
    Create a symmetric weight matrix with random values, ensuring weights between neurons in inputn are zero.
    Additional options control the eigenvalues and signs of non-diagonal elements.

    :param N: int, number of neurons
    :param loc: float, mean of the normal distribution
    :param scale: float, standard deviation of the normal distribution
    :param inputn: list or array, indices of neurons whose weights must be zero
    :param rng_key: jax.random.PRNGKey, random key for reproducibility
    :param options: dict, additional options to control matrix properties
                   - 'eigenvalues': 'one_positive', 'all_positive', 'all_negative'
                   - 'nondiagonal': 'positive', 'negative'
    :return: jnp.array, symmetric weight matrix
    """
    # Generate random numbers using jax.random
    random_values = jax.random.normal(rng_key, shape=(N, N)) * scale + loc
    # Create a lower triangular matrix
    lower_triangular = jnp.tril(random_values, k=-1)
    # Reflect it to make it symmetric
    symmetric_matrix = lower_triangular + lower_triangular.T
    # Set the diagonal to zero
    symmetric_matrix = symmetric_matrix.at[jnp.diag_indices(N)].set(0)

    # Apply additional options if provided
    if options is not None:
        if 'diagonal' in options:
            diagonal_option = options['diagonal']
            for i in inputn:
                symmetric_matrix = symmetric_matrix.at[i, inputn].set(0)
                symmetric_matrix = symmetric_matrix.at[inputn, i].set(0)
            if diagonal_option == "zero":
                pass
            elif diagonal_option =="nonzero":
                symmetric_matrix += jnp.eye(N)*1
                symmetric_matrix += jnp.eye(N)*1
                pass

        if 'eigenvalues' in options:
            eigenvalues_option = options['eigenvalues']
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eigh(symmetric_matrix)
            sorted_eigenvalues = jnp.sort(eigenvalues)
            if eigenvalues_option == 'one_positive':
                # Shift diagonal to ensure only one positive eigenvalue
                shift = -sorted_eigenvalues[-2]
                symmetric_matrix = symmetric_matrix.at[jnp.diag_indices(N)].add(shift)
                eigenvalues, eigenvectors = eigh(symmetric_matrix)
            elif eigenvalues_option == 'all_positive':
                # Shift diagonal to ensure all eigenvalues are positive
                shift = -jnp.min(eigenvalues) + 1e-2
                symmetric_matrix = symmetric_matrix.at[jnp.diag_indices(N)].add(shift)
                eigenvalues, eigenvectors = eigh(symmetric_matrix)
            elif eigenvalues_option == 'all_negative':
                # Shift diagonal to ensure all eigenvalues are negative
                shift = -jnp.max(eigenvalues) - 1e-5
                symmetric_matrix = symmetric_matrix.at[jnp.diag_indices(N)].add(shift)
                eigenvalues, eigenvectors = eigh(symmetric_matrix)

        if 'nondiagonal' in options:
            nondiagonal_option = options['nondiagonal']
            if nondiagonal_option == 'positive':
                # Ensure all non-diagonal elements are positive
                symmetric_matrix = jnp.where(symmetric_matrix < 0, 0, symmetric_matrix)
            elif nondiagonal_option == 'negative':
                # Ensure all non-diagonal elements are negative
                symmetric_matrix = jnp.where(symmetric_matrix > 0, 0, symmetric_matrix)

    return symmetric_matrix

# Function to initialize neurons and connections
"""
def initialize_neurons(N, inputn):
    neurons = jnp.arange(0, N, 1)
    connections_neuronwise = jnp.array([
        [element for element in neurons if element != neuron]
        for neuron in neurons
    ])
    return neurons, connections_neuronwise
"""

def initialize_neurons(N, inputn):
    neurons = jnp.arange(0, N, 1)
    connections_neuronwise = jnp.array([
        [element for element in neurons] # previously added 'if element != neuron'
        for neuron in neurons
    ])
    return neurons, connections_neuronwise
# Function to initialize weights and fields
def initialize_weights_and_SL_fields(N, inputn, connections_neuronwise, rng_key, weight_option):
    weights_real_matrix = create_symmetric_weights(N, 0., 1., inputn, rng_key, options=weight_option)
    weights_imaginary_matrix = create_symmetric_weights(N, 0., 1., inputn, rng_key, options=weight_option)
    weight_update_mask = create_weight_update_mask(N, inputn)
    weights_real = weights_real_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
    weights_imaginary = weights_imaginary_matrix[connections_neuronwise, jnp.arange(N)[:, None]]

    pField = jnp.zeros(N)
    alternating_array = jnp.array([(-1) ** i for i in range(N)])
    uField = jax.random.uniform(rng_key, shape=(N,), minval=20, maxval=40) * alternating_array

    return weights_real, weights_real_matrix, weights_imaginary, weights_imaginary_matrix, weight_update_mask, pField, uField


def initialize_weights_and_K_fields(N, inputn, connections_neuronwise, rng_key):
    weights_matrix = create_symmetric_weights(N, 0., 1., inputn, rng_key)
    weight_update_mask = create_weight_update_mask(N, inputn)
    weights = weights_matrix[connections_neuronwise, jnp.arange(N)[:, None]]

    biases = jax.random.uniform(rng_key, shape=(N,), minval=-0.5, maxval=0.5)
    bias_phases = jax.random.uniform(rng_key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)
    return weights, weights_matrix, weight_update_mask, biases, bias_phases

# Function to initialize simulation parameters
def initialize_simulation_params(N, outputn, batch_size, random_init_times, beta_value=1e-4):
    beta = jnp.zeros(N).at[outputn].set(beta_value)
    inv_nudge_step = 1 / beta[outputn[0]]
    inv_batch_size = 1 / batch_size
    inv_random_init_times = 1 / random_init_times
    return beta, inv_nudge_step, inv_batch_size, inv_random_init_times

# Function to map features and labels
def initialize_SL_states_and_features(feature_multiplier, feature_constant, label_multiplier, init_amplitudes, init_phases, uField, inputn, outputn, map_features_and_labels):
    amplitude_relative, features, labels = map_features_and_labels(feature_multiplier, feature_constant, label_multiplier, init_amplitudes, outputn)
    uField = uField.at[jnp.array(inputn)].set([features[0][0], features[1][1]])
    return amplitude_relative, features, labels
