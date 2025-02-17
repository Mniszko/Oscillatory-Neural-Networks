import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import argparse
import time
from src import save_array_to_file, solve_SL_ode_free, solve_SL_ode_nudged, sum_and_divide_array, determine_SL_binary_distance, main_SL_training_preamble as main_training_preamble, XOR_problem_SL_determine_accuracy, XOR_problem_SL_map_features_and_labels, shuffle_and_batch, double_XOR_SL_map_features_and_labels, double_XOR_SL_determine_accuracy, record_states, write_separator

jax.config.update("jax_enable_x64", True)

determine_distance = determine_SL_binary_distance

solve_ode_free = solve_SL_ode_free
solve_ode_nudged = solve_SL_ode_nudged

determine_accuracy = XOR_problem_SL_determine_accuracy
map_features_and_labels = XOR_problem_SL_map_features_and_labels

"""
determine_accuracy = double_XOR_SL_determine_accuracy
map_features_and_labels = double_XOR_SL_map_features_and_labels
"""

def training_function(name, N, do_save, num_of_epochs, learning_rate, weight_type, normalize, feature_multiplier, feature_constant, label_multiplier):
    #compiled here because it needs static N
    @jax.jit
    def calculate_energy_gradient(amplitudes, phases, weight_type):
        # Allocate gradient_weights and gradient_biases
        if weight_type == 'r':
            gradient_weights_real = jnp.zeros((N, N))
            gradient_weights_imaginary = False
        if weight_type == 'i':
            gradient_weights_imaginary = jnp.zeros((N, N))
            gradient_weights_real = False
        if weight_type == 'c':
            gradient_weights_real = jnp.zeros((N, N))
            gradient_weights_imaginary = jnp.zeros((N, N))
        gradient_biases = amplitudes * jnp.cos(phases)

        # Vectorized calculation of gradient_weights
        i_indices, j_indices = jnp.triu_indices(N, k=1)

        if weight_type == 'r':
            values_real = -amplitudes[i_indices] * amplitudes[j_indices] * jnp.cos(phases[i_indices] - phases[j_indices])
            gradient_weights_real = gradient_weights_real.at[i_indices, j_indices].set(values_real)
            gradient_weights_real = gradient_weights_real.at[j_indices, i_indices].set(values_real)  # Symmetric assignment

        if weight_type == 'i':
            values_imaginary = amplitudes[i_indices] * amplitudes[j_indices] * jnp.sin(phases[i_indices] - phases[j_indices])
            gradient_weights_imaginary = gradient_weights_imaginary.at[i_indices, j_indices].set(values_imaginary)
            gradient_weights_imaginary = gradient_weights_imaginary.at[j_indices, i_indices].set(values_imaginary)  # Symmetric assignment

        if weight_type == 'c':
            values_real = -amplitudes[i_indices] * amplitudes[j_indices] * jnp.cos(phases[i_indices] - phases[j_indices])
            gradient_weights_real = gradient_weights_real.at[i_indices, j_indices].set(values_real)
            gradient_weights_real = gradient_weights_real.at[j_indices, i_indices].set(values_real)  # Symmetric assignment
            values_imaginary = amplitudes[i_indices] * amplitudes[j_indices] * jnp.sin(phases[i_indices] - phases[j_indices])
            gradient_weights_imaginary = gradient_weights_imaginary.at[i_indices, j_indices].set(values_imaginary)
            gradient_weights_imaginary = gradient_weights_imaginary.at[j_indices, i_indices].set(values_imaginary)  # Symmetric assignment

        return gradient_weights_real, gradient_weights_imaginary, gradient_biases

    rng_key = jax.random.PRNGKey(round(time.time()*1e7))
    if N<3:
        print('Not enough neurons for minimal inference with 2i 1o net')
        return 1
    if weight_type not in ['r','i','c']:
        print('Wrong weight type entered')
        return 1

    T = 200
    dt = 0.01
    omega = jnp.zeros(N)
    alpha = 1.
    batch_size = 4
    random_init_times = 1
    inputn = [0,1]
    outputn = [2]
    inputn = jnp.array(inputn)
    outputn = jnp.array(outputn)

    preamble = main_training_preamble(N, T, dt, omega, alpha, batch_size, random_init_times, inputn, outputn, rng_key, feature_multiplier, feature_constant, label_multiplier, weight_type, map_features_and_labels)
    neurons = preamble['neurons']
    connections_neuronwise = preamble['connections_neuronwise']
    weights_real = preamble['weights_real']
    weights_real_matrix = preamble['weights_real_matrix']
    weights_imaginary = preamble['weights_imaginary']
    weights_imaginary_matrix = preamble['weights_imaginary_matrix']
    weight_update_mask = preamble['weight_update_mask']
    pField = preamble['pField']
    uField = preamble['uField']
    beta = preamble['beta']
    inv_nudge_step = preamble['inv_nudge_step']
    inv_batch_size = preamble['inv_batch_size']
    inv_random_init_times = preamble['inv_random_init_times']
    times = preamble['times']
    init_amplitudes = preamble['init_amplitudes']
    init_phases = preamble['init_phases']
    input_mask = preamble['input_mask']
    amplitude_relative = preamble['amplitude_relative']
    features = preamble['features']
    labels = preamble['labels']

    distances = []
    accuracies = []

    print(f"\tAmplitude relative: \t{amplitude_relative}")

    # training the network
    for epoch in range(num_of_epochs):

        time0 = time.time()
        weight_real_gradient = jnp.zeros((N,N))
        weight_imaginary_gradient = jnp.zeros((N,N))
        bias_gradient = jnp.zeros(N)
        bias_phase_gradient = jnp.zeros(N)

        distance_temp = []
        accuracies_temp = []

        if (epoch+1)%10 == 0 or epoch==0:
            print(f"epoch number {epoch+1}")
        
        batches = shuffle_and_batch(features, labels, batch_size, rng_key)
        for batch in batches:
            for feature, label in batch:

                target = jnp.zeros(N)
                target = target.at[outputn].set(label)

                uField = uField.at[jnp.array(inputn)].set([feature[0], feature[1]])

                states = solve_ode_free((init_amplitudes, init_phases), times, weights_real, weights_imaginary, alpha, omega, pField, uField, connections_neuronwise, input_mask)
                amplitudes = states[0][-1]
                phases = states[1][-1]
                if do_save=="y" or do_save=="yes":
                    record_states(name, amplitudes, phases) #records stable states across each batch

                # removes nonstable solutions by restarting the thing
                if any(x > 1e-5 for x in (states[0][-10] - states[0][-1])):
                    if not T==400:
                        T=400 # first we try to make simulation time longer, if that doesn't work the parameters are discarted
                    else:
                        print(f"\tNonstable final state encountered! Restarting from epoch {epoch}")
                        return 1

                gradient_weights_real_forward, gradient_weights_imaginary_forward, gradient_biases_forward = calculate_energy_gradient(amplitudes, phases)


                # appending to training data arrays
                distance_temp.append(determine_distance(amplitudes, label, outputn))
                accuracies_temp.append(determine_accuracy(amplitudes, label, outputn, amplitude_relative))

                """
                # debugging
                if (epoch+1)%10 == 0 or epoch==0:
                    print(f"output vs label: {amplitudes[outputn]} ---- {label}")
                """

                states = solve_ode_nudged((init_amplitudes, init_phases), times, weights_real, weights_imaginary, alpha, omega, pField, uField, connections_neuronwise, input_mask, beta, target)
                amplitudes = states[0][-1]
                phases = states[1][-1]

                gradient_weights_real_backward, gradient_weights_imaginary_backward, gradient_biases_backward = calculate_energy_gradient(amplitudes, phases)

                if weight_type == 'r':
                    weight_real_gradient += gradient_weights_real_backward - gradient_weights_real_forward
                elif weight_type == 'i':
                    weight_imaginary_gradient += gradient_weights_imaginary_backward - gradient_weights_imaginary_forward
                elif weight_type == 'c':
                    weight_real_gradient += gradient_weights_real_backward - gradient_weights_real_forward
                    weight_imaginary_gradient += gradient_weights_imaginary_backward - gradient_weights_imaginary_forward
                bias_gradient += gradient_biases_backward - gradient_biases_forward

            # Parameters are updated once per batch loop
            if weight_type == 'r':
                weight_real_gradient *= inv_batch_size*inv_random_init_times*inv_nudge_step
            elif weight_type == 'i':
                weight_imaginary_gradient *= inv_batch_size*inv_random_init_times*inv_nudge_step
            elif weight_type == 'c':
                weight_real_gradient *= inv_batch_size*inv_random_init_times*inv_nudge_step
                weight_imaginary_gradient *= inv_batch_size*inv_random_init_times*inv_nudge_step
            bias_gradient *= inv_batch_size*inv_random_init_times*inv_nudge_step
            
            if normalize: #np.linalg.norm(weight_gradient,ord=1) > 1* N*N:
                #print(f"gradient absolute size exceeded expected value\nL(∇W) = {np.linalg.norm(weight_gradient, ord=1)}")
                #print("NORMALIZED!")
                if weight_type == 'r':
                    weight_real_gradient /= jnp.linalg.norm(weight_real_gradient,ord=1)
                elif weight_type == 'i':
                    weight_imaginary_gradient /= jnp.linalg.norm(weight_imaginary_gradient,ord=1)
                elif weight_type == 'c':
                    weight_real_gradient /= jnp.linalg.norm(weight_real_gradient,ord=1)
                    weight_imaginary_gradient /= jnp.linalg.norm(weight_imaginary_gradient,ord=1)
                bias_gradient /= jnp.linalg.norm(bias_gradient,ord=1)

            if weight_type == 'r':
                weights_real_matrix -= learning_rate * weight_real_gradient * weight_update_mask * 0.5 # 0.5 comes from hamiltonian formulation
                weights_real = weights_real_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
            elif weight_type == 'i':
                weights_imaginary_matrix -= learning_rate * weight_imaginary_gradient * weight_update_mask * 0.5
                weights_imaginary = weights_imaginary_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
            elif weight_type == 'c':
                weights_real_matrix -= learning_rate * weight_real_gradient * weight_update_mask * 0.5 # 0.5 comes from hamiltonian formulation
                weights_real = weights_real_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
                weights_imaginary_matrix -= learning_rate * weight_imaginary_gradient * weight_update_mask * 0.5
                weights_imaginary = weights_imaginary_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
            uField -= learning_rate * bias_gradient * 0.5

        distances.append(sum_and_divide_array(distance_temp, batch_size))
        accuracies.append(sum_and_divide_array(accuracies_temp, batch_size))


        time1 = time.time()
        if False:#time1-time0>10:
            print(f"Too much time taken for epoch {epoch}: {time1-time0} ------- threshold = {10}")
            return 1

        """
        # debugging
        if (epoch+1)%100 == 0 or epoch==0:
            print(f"Finished epoch number {epoch+1}")
            print(f"distances read {jnp.array(distance_temp).tolist()}")
            print(f"accuracies read {jnp.array(accuracies_temp).tolist()}")
            print(f"time taken for the epoch: {time1-time0}")
        """
    if do_save=="y" or do_save=="yes":
        write_separator(name) # writes separator in stable state recordings
        save_array_to_file(jnp.array(distances), name + ".txt")
        save_array_to_file(jnp.array(accuracies), name + "_acc.txt")

    # plotting
    elif do_save=="n" or do_save=="no":
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        axes[0].plot(distances, label="amplitude", c="r")
        axes[0].set_title("distances")
        axes[0].grid()
        axes[1].plot(accuracies, c="b")
        axes[1].set_title("accuracies")
        axes[1].grid()
        plt.savefig("XOR_SLNN_distances.png")
        plt.show()

    return 0

def main():
    parser = argparse.ArgumentParser(description="training")

    # Define the arguments
    parser.add_argument('name', type=str, help="Filenames of output data.")
    parser.add_argument('number', type=int, help="An integer number.")
    parser.add_argument('letter', type=str, help="A letter (string) save or not y or n.")
    parser.add_argument('num_of_epochs', type=int, help="Number of epochs.")
    parser.add_argument('learning_rate', type=float, help="Real valued rate of learning (commonly eta).")
    parser.add_argument('weight_type', type=str, help="Weights should be chosen to be trained as real, imaginary or complex values (understood as in SL model without imaginary unit by time derivative), acceptable values are c, i, r as in complex, imaginary, real")
    parser.add_argument('normalize', type=bool, help="Normalize gradients during gradient descent according to internal rules: 0 or 1 (false or true)")
    parser.add_argument('feature_multiplier', type=float, help="feature multiplier value")
    parser.add_argument('feature_constant', type=float, help="feature constant value")
    parser.add_argument('label_multiplier', type=float, help="label multiplier value")
    args = parser.parse_args()


    while True:
        answer = training_function(args.name, args.number, args.letter, args.num_of_epochs, args.learning_rate, args.weight_type, args.normalize, args.feature_multiplier, args.feature_constant, args.label_multiplier)
        if answer == 0:
            break

if __name__ == "__main__":
    #main_dynamics_test()
    main()
