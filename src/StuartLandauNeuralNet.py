import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit, random
from .InitializationModule import create_symmetric_weights, initialize_neurons, initialize_weights_and_SL_fields, initialize_simulation_params, initialize_SL_states_and_features
from .WeightsModule import create_random_connections, create_square_lattice_connections, create_weight_update_mask

# in this code I'm to use input with two methods simultaneously - via external force parameter (u_n) and via specific weight stabilization.
@jit
def network_evolution(state, t, W, alpha, omega, pField, uField, coupled_neuron, input_mask):
    """
    Computes the derivatives of system parameters (amplitudes and phases)

    :param state: tuple, (amplitudes, phases) where:
                  amplitudes: jnp.array of amplitudes (rho_1, rho_2, ..., rho_n)
                  phases: jnp.array of phases (theta_1, theta_2, ..., theta_n)
    :param t: float, current time
    :param W: jnp.array coupling matrix
    :param alpha: float, constant parameter
    :param omega: jnp.array, vector of frequencies
    :param pField: jnp.array, vector of pump biases
    :param uField: jnp.array, vector of inputs as forces
    :param coupled_neuron: jnp.array, indices of coupled oscillators where row is a neuron and elements are its couplings. Warning! if i is coupled to j, j has to be coupled back to i
    :param input_mask: jnp.array, vector of binary elements where 0 corresponds to indeces with artificially stabilized evolution and 1 to those with standard dynamical evolution
    :return: tuple, (dden_dt, dphase_dt), derivatives of amplitudes and phases
    """
    amplitudes, phases = state

    # Gather coupled neuron values
    coupled_amplitudes = amplitudes[coupled_neuron]
    coupled_phases = phases[coupled_neuron]

    # Compute amplitude derivatives (dden_dt)
    amplitude_coupling = jnp.sum(W * coupled_amplitudes * jnp.cos(coupled_phases - phases[:, None]), axis=1)
    dden_dt = -alpha * amplitudes**3 + amplitudes * pField + amplitude_coupling + uField * jnp.cos(phases)

    # Compute phase derivatives (dphase_dt)
    phase_coupling = jnp.sum(W * coupled_amplitudes / amplitudes[:, None] * jnp.sin(coupled_phases - phases[:, None]), axis=1)
    dphase_dt = omega + phase_coupling - uField / amplitudes * jnp.sin(phases)

    return dden_dt * input_mask, dphase_dt * input_mask

@jit
def network_evolution_nudge(state, t, W, alpha, omega, pField, uField, coupled_neuron, input_mask, beta, target):
    """
    Computes the derivatives of system parameters (amplitudes and phases) after nudge of dynamics

    :param state: tuple, (amplitudes, phases) where:
                  amplitudes: jnp.array of amplitudes (rho_1, rho_2, ..., rho_n)
                  phases: jnp.array of phases (theta_1, theta_2, ..., theta_n)
    :param t: float, current time
    :param W: jnp.array coupling matrix
    :param alpha: float, constant parameter
    :param omega: jnp.array, vector of frequencies
    :param pField: jnp.array, vector of pump biases
    :param uField: jnp.array, vector of inputs as forces
    :param coupled_neuron: jnp.array, indices of coupled oscillators where row is a neuron and elements are its couplings. Warning! if i is coupled to j, j has to be coupled back to i
    :param input_mask: jnp.array, vector of binary elements where 0 corresponds to indeces with artificially stabilized evolution and 1 to those with standard dynamical evolution
    :param beta: jnp.array, vector of real elements, nonzero on indeces corresponding to output neurons
    :param target: jnp.array, vector of real elements, nonzero where label mapping is introduced (on output neuron indeces)
    :return: tuple, (dden_dt, dphase_dt), derivatives of amplitudes and phases
    """
    amplitudes, phases = state

    # Gather coupled neuron values
    coupled_amplitudes = amplitudes[coupled_neuron]
    coupled_phases = phases[coupled_neuron]

    # Compute amplitude derivatives (dden_dt)
    amplitude_coupling = jnp.sum(W * coupled_amplitudes * jnp.cos(coupled_phases - phases[:, None]), axis=1)
    dden_dt = -alpha * amplitudes**3 + amplitudes * pField + amplitude_coupling + uField * jnp.cos(phases)
    dden_dt += - beta*(amplitudes-target)

    # Compute phase derivatives (dphase_dt)
    phase_coupling = jnp.sum(W * coupled_amplitudes / amplitudes[:, None] * jnp.sin(coupled_phases - phases[:, None]), axis=1)
    dphase_dt = omega + phase_coupling - uField / amplitudes * jnp.sin(phases)

    return dden_dt * input_mask, dphase_dt * input_mask

@jit
def solve_SL_ode_free(state, times, weights, alpha, omega, pField, uField, coupled_neuron, input_mask):
    return odeint(
        network_evolution,
        state,
        times,
        weights,
        alpha,
        omega,
        pField,
        uField,
        coupled_neuron,
        input_mask
    )

@jit
def solve_SL_ode_nudged(state, times, weights, alpha, omega, pField, uField, coupled_neuron, input_mask, beta, target):
    return odeint(
        network_evolution_nudge,
        state,
        times,
        weights,
        alpha,
        omega,
        pField,
        uField,
        coupled_neuron,
        input_mask,
        beta,
        target
    )


@jit
def sum_and_divide_array(array, divisor):
    return jnp.sum(jnp.array(array))/divisor

def determine_SL_binary_distance(amplitude, label, outputn):
    return jnp.abs(jnp.sum(amplitude[outputn] - label))

def main_SL_training_preamble(N, T, dt, omega, alpha, batch_size, random_init_times, inputn, outputn, rng_key, feature_multiplier, feature_constant, label_multiplier, map_features_and_labels):
    """
    Function returning object of all parameters
    N - integer larger than number of inputs and number of outputs
    T - real value
    dt - real value
    omega - real value
    alpha - real value
    batch_size - integer smaller than number of feature an label elements
    random_init_times - integer
    inputn - array of integers smaller than N and different from one another
    outputn - integer smaller than N and different from those in inputn
    map_features_and_labels - function mapping features and labels for a given problem to amplitudes
    """
    neurons, connections_neuronwise = initialize_neurons(N, inputn)
    weights, weights_matrix, weight_update_mask, pField, uField = initialize_weights_and_SL_fields(N, inputn, connections_neuronwise, rng_key)
    beta, inv_nudge_step, inv_batch_size, inv_random_init_times = initialize_simulation_params(
        N, outputn, batch_size, random_init_times
    )

    times = jnp.arange(0, T + dt, dt)
    init_amplitudes = jax.random.uniform(rng_key, shape=(N,), minval=-1, maxval=1)/2 + 2
    init_phases = jax.random.uniform(rng_key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)/200

    # input mask for choosing naurons that do not evolve (here unused)
    input_mask = jnp.ones(N)


    states = solve_SL_ode_free((init_amplitudes, init_phases), times, weights, alpha, omega, pField, uField, connections_neuronwise, input_mask)

    init_amplitudes = states[0][-1]
    init_phases = states[1][-1]
    amplitude_relative, features, labels = initialize_SL_states_and_features(feature_multiplier, feature_constant, label_multiplier, init_amplitudes, init_phases, uField, inputn, outputn, map_features_and_labels)

    return {
        'neurons': neurons,
        'connections_neuronwise': connections_neuronwise,
        'weights': weights,
        'weights_matrix': weights_matrix,
        'weight_update_mask': weight_update_mask,
        'pField': pField,
        'uField': uField,
        'beta': beta,
        'inv_nudge_step': inv_nudge_step,
        'inv_batch_size': inv_batch_size,
        'inv_random_init_times': inv_random_init_times,
        'times': times,
        'init_amplitudes': init_amplitudes,
        'init_phases': init_phases,
        'input_mask': input_mask,
        'amplitude_relative': amplitude_relative,
        'features': features,
        'labels': labels
    }

def shuffle_and_batch(array1, array2, batch_size, key):
    """
    Shuffle two arrays simultaneously using JAX random key and batch them.

    Args:
        array1 (list or jnp.ndarray): The first array.
        array2 (list or jnp.ndarray): The second array (same length as array1).
        batch_size (int): Size of each batch.
        key (jax.random.PRNGKey): A random key for reproducibility.

    Returns:
        list: A list of batches where each batch is a list of lists [[a1, b1], [a2, b2], ...].
    """
    assert len(array1) == len(array2), "Arrays must have the same length."
    assert len(array1) % batch_size == 0, "Array length must be divisible by batch_size."

    array1 = jnp.array(array1)
    array2 = jnp.array(array2)

    # Generate a permutation of indices
    perm = random.permutation(key, len(array1))

    # Shuffle arrays using the permuted indices
    shuffled_array1 = array1[perm]
    shuffled_array2 = array2[perm]

    # Convert back to Python lists and batch
    shuffled_array1 = list(shuffled_array1)
    shuffled_array2 = list(shuffled_array2)

    batches = [
        [[shuffled_array1[i + j], shuffled_array2[i + j]] for j in range(batch_size)]
        for i in range(0, len(array1), batch_size)
    ]

    return batches
