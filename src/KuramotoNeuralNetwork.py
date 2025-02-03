import numpy as np
from scipy.integrate import odeint
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
jax.config.update("jax_enable_x64", True)
from .InitializationModule import initialize_simulation_params, initialize_weights_and_K_fields, initialize_neurons

def run_function_pulsed(function, arguments, phases, feature, time_array, slice_start, slice_stop, time_step, num_slices):
    """
    Method for running kuramoto evolution with given number of single step pulses changing parameters to given values before further evolution.
    """
    slice_start_num = int(slice_start/time_step)
    slice_stop_num = int(slice_stop/time_step)

    inputn = arguments[4]
    slicing_range = time_array[slice_start_num:slice_stop_num]
    step = len(slicing_range)//num_slices
    slice_before = time_array[0:slice_start_num]
    slice_after = time_array[slice_stop_num:-1]

    solution_full = []

    if len(slice_before) > 1:
        solution = odeint(function, phases, slice_before, args=arguments, full_output=0)
        phases = solution[-1]
        solution_full.append(solution)
    
    for i in range(0, len(slicing_range), step):
        sliced_time = slicing_range[i:i+step]
        solution = odeint(function, phases, sliced_time, args=arguments, full_output=0)
        phases = solution[-1]
        solution_full.append(solution)
        phases[inputn[0]] = feature[0]
        phases[inputn[1]] = feature[1]
        print("impulse at time:", (len(slice_before)+i+step)*time_step)

    if len(slice_after) > 1:
        solution = odeint(function, phases, slice_after, args=arguments, full_output=0)
        phases = solution[-1]
        solution_full.append(solution)
    
    return np.vstack(solution_full), np.array(solution[-1][-1])

@jax.jit
def kuramoto_oscillators(theta, t, K, h, psi, coupled_theta, input_mask):
    """
    Computes the derivative of theta for the Kuramoto oscillators (JAX version).
    
    :param theta: jnp.array, current phase angles
    :param t: float, current time (not used in this function)
    :param K: jnp.array, coupling matrix
    :param h: jnp.array, external driving strengths
    :param psi: jnp.array, external driving phases
    :param coupled_theta: jnp.array, indices of coupled oscillators
    :param input_mask: list, mask where 0 indicates input neurons and 1 indicates free neurons
    :return: jnp.array, derivatives of theta
    """
    # Extract the coupled values
    coupled_values = theta[coupled_theta]

    # Compute the sin(theta_i - coupled_theta)
    sin_diffs = jnp.sin(theta[:, None] - coupled_values)  # Vectorized difference
    sin_external = jnp.sin(theta - psi)

    # Compute dtheta_dt using vectorized operations
    dtheta_dt = -jnp.sum(sin_diffs * K, axis=1) - h * sin_external 

    return dtheta_dt * input_mask

@jax.jit
def kuramoto_oscillators_nudge(theta, t, K, h, psi, coupled_theta, input_mask, beta, target):
    """
    Computes the derivative of theta for the Kuramoto oscillators with a nudge term (JAX version).
    
    :param theta: jnp.array, current phase angles
    :param t: float, current time (not used in this function)
    :param K: jnp.array, coupling matrix
    :param h: jnp.array, external driving strengths
    :param psi: jnp.array, external driving phases
    :param coupled_theta: jnp.array, indices of coupled oscillators
    :param input_mask: jnp.array, mask where 0 indicates input neurons and 1 indicates free neurons
    :param beta: jnp.array, nudge strengths
    :param target: jnp.array, target phase angles
    :return: jnp.array, derivatives of theta
    """
    # Extract the coupled values
    coupled_values = theta[coupled_theta]

    # Compute the sin(theta_i - coupled_theta) differences
    sin_diffs = jnp.sin(theta[:, None] - coupled_values)  # Vectorized difference
    sin_external = jnp.sin(theta - psi)

    # Nudge term
    nudge_term = beta * jnp.sin(theta - target) / (jnp.cos(theta - target) + 1 + 1e-8)

    # Compute dtheta_dt using vectorized operations
    dtheta_dt = -(
        jnp.sum(sin_diffs * K, axis=1)
        + h * sin_external
        + nudge_term
    ) 

    return dtheta_dt * input_mask
    
@jax.jit
def solve_K_ode_free(phases, times, weights, biases, bias_phases, connections_neuronwise, input_mask):
    return odeint(
        kuramoto_oscillators,
        phases,
        times,
        weights,
        biases,
        bias_phases,
        connections_neuronwise,
        input_mask
    )
@jax.jit
def solve_K_ode_nudged(phases, times, weights, biases, bias_phases, connections_neuronwise, input_mask, beta, target):
    return odeint(
        kuramoto_oscillators_nudge,
        phases,
        times,
        weights,
        biases,
        bias_phases,
        connections_neuronwise,
        input_mask,
        beta,
        target
    )


@jax.jit
def compute_kuramoto_gradients(phases, weights, biases, bias_phases):
    N = phases.shape[0]

    # Compute gradient_weights_forward using broadcasting
    delta_phases = phases[:, None] - phases[None, :]
    gradient_weights_forward = -jnp.cos(delta_phases)

    # Compute gradient_biases_forward and gradient_bias_phases_forward
    gradient_biases_forward = -jnp.cos(phases - bias_phases)
    gradient_bias_phases_forward = -biases * jnp.sin(phases - bias_phases)

    return gradient_weights_forward, gradient_biases_forward, gradient_bias_phases_forward

def determine_K_binary_distance(phases, outputn, label):
    return 1-jnp.cos(phases[outputn]-label)

def main_K_training_preamble(N, T, dt, batch_size, random_init_times, inputn, outputn, rng_key, map_features_and_labels):

    neurons, connections_neuronwise = initialize_neurons(N, inputn)
    beta, inv_nudge_step, inv_batch_size, inv_random_init_times = initialize_simulation_params(
        N, outputn, batch_size, random_init_times
    )
    features, labels= map_features_and_labels()
    times = jnp.arange(0, T+dt, dt)

    # input mask for choosing naurons that do not evolve
    input_mask = jnp.array([0 if neuron in inputn else 1 for neuron in neurons])    

    weights, weights_matrix, weight_update_mask, biases, bias_phases = initialize_weights_and_K_fields(N, inputn, connections_neuronwise, rng_key)
    
    init_phases = jax.random.uniform(rng_key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)

    return {
        'neurons': neurons,
        'connections_neuronwise': connections_neuronwise,
        'weights': weights,
        'weights_matrix': weights_matrix,
        'weight_update_mask': weight_update_mask,
        'biases': biases,
        'bias_phases': bias_phases,
        'beta': beta,
        'inv_nudge_step': inv_nudge_step,
        'inv_batch_size': inv_batch_size,
        'inv_random_init_times': inv_random_init_times,
        'times': times,
        'init_phases': init_phases,
        'input_mask': input_mask,
        'features': features,
        'labels': labels
    }