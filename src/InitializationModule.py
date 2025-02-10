import jax
import jax.numpy as jnp
from .WeightsModule import create_weight_update_mask

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

# Function to initialize neurons and connections
def initialize_neurons(N, inputn):
    neurons = jnp.arange(0, N, 1)
    connections_neuronwise = jnp.array([
        [element for element in neurons if element != neuron]
        for neuron in neurons
    ])
    return neurons, connections_neuronwise

# Function to initialize weights and fields
def initialize_weights_and_SL_fields(N, inputn, connections_neuronwise, rng_key):
    weights_real_matrix = create_symmetric_weights(N, 0., 1., inputn, rng_key)
    weights_imaginary_matrix = create_symmetric_weights(N, 0., 1., inputn, rng_key)
    weight_update_mask = create_weight_update_mask(N, inputn)
    weights_real = weights_real_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
    weights_imaginary = weights_imaginary_matrix[connections_neuronwise, jnp.arange(N)[:, None]]

    pField = jnp.zeros(N)
    uField = jax.random.uniform(rng_key, shape=(N,), minval=-30, maxval=30)
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
