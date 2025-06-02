import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import argparse
import time
import os, glob
from src import save_array_to_file, solve_SL_ode_free, solve_SL_ode_nudged, sum_and_divide_array, determine_SL_binary_distance, main_SL_training_preamble as main_training_preamble, XOR_problem_SL_determine_accuracy, XOR_problem_SL_map_features_and_labels, shuffle_and_batch, double_XOR_SL_map_features_and_labels, double_XOR_SL_determine_accuracy, record_all_states, write_separator, save_single_value, OptDigits_SL_determine_accuracy, OptDigits_SL_map_features_and_labels_with_initial_randomization, OptDigits_SL_map_features_and_labels, OptDigits_separate_training_and_test, testLoop, record_important_values, record_states

jax.config.update("jax_enable_x64", True)

determine_distance = determine_SL_binary_distance

solve_ode_free = solve_SL_ode_free
solve_ode_nudged = solve_SL_ode_nudged

determine_accuracy = OptDigits_SL_determine_accuracy
map_features_and_labels = OptDigits_SL_map_features_and_labels_with_initial_randomization

"""
determine_accuracy = double_XOR_SL_determine_accuracy
map_features_and_labels = double_XOR_SL_map_features_and_labels
"""

def training_function(name, N, do_save, num_of_epochs, learning_rate, weight_type, normalize, feature_multiplier, feature_constant, label_multiplier, weight_option, high_value, beta_val, lattice_connections):
    #compiled here because it needs static N
    @jax.jit
    def calculate_energy_gradient(amplitudes, phases):
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
        gradient_biases = -2 * amplitudes * jnp.cos(phases)

        # Vectorized calculation of gradient_weights
        i_indices, j_indices = jnp.triu_indices(N, k=1)

        if weight_type == 'r':
            values_real = -amplitudes[i_indices] * amplitudes[j_indices] * jnp.cos(phases[i_indices] - phases[j_indices])/2
            gradient_weights_real = gradient_weights_real.at[i_indices, j_indices].set(values_real)
            gradient_weights_real = gradient_weights_real.at[j_indices, i_indices].set(values_real)/2  # Symmetric assignment

        if weight_type == 'i':
            values_imaginary = amplitudes[i_indices] * amplitudes[j_indices] * jnp.sin(phases[i_indices] - phases[j_indices])/2
            gradient_weights_imaginary = gradient_weights_imaginary.at[i_indices, j_indices].set(values_imaginary)
            gradient_weights_imaginary = gradient_weights_imaginary.at[j_indices, i_indices].set(values_imaginary)  # Symmetric assignment

        if weight_type == 'c':
            values_real = -amplitudes[i_indices] * amplitudes[j_indices] * jnp.cos(phases[i_indices] - phases[j_indices])/2
            gradient_weights_real = gradient_weights_real.at[i_indices, j_indices].set(values_real)
            gradient_weights_real = gradient_weights_real.at[j_indices, i_indices].set(values_real)  # Symmetric assignment
            values_imaginary = amplitudes[i_indices] * amplitudes[j_indices] * jnp.sin(phases[i_indices] - phases[j_indices])/2
            gradient_weights_imaginary = gradient_weights_imaginary.at[i_indices, j_indices].set(values_imaginary)
            gradient_weights_imaginary = gradient_weights_imaginary.at[j_indices, i_indices].set(values_imaginary)  # Symmetric assignment

        return gradient_weights_real, gradient_weights_imaginary, gradient_biases

    rng_key = jax.random.PRNGKey(round(time.time()*1e7))
    if N<74:
        print('Not enough neurons for minimal inference with 64i 10o net')
        return 1
    if weight_type not in ['r','i','c']:
        print('Wrong weight type entered')
        return 1

    T = 200
    dt = 0.01
    omega = jnp.zeros(N)
    alpha = 1.
    batch_size = 10
    random_init_times = 1 # placeholder without puprose
    if N < 93:
        inputn = jnp.arange(0,64,1)
        outputn = jnp.arange(64,74,1)
    else:
        inputn = jnp.arange(10,74,1)
        outputn = jnp.arange(N-20,N-10,1)

    preamble = main_training_preamble(N, T, dt, omega, alpha, batch_size, random_init_times, inputn, outputn, rng_key, feature_multiplier, feature_constant, label_multiplier, weight_type, map_features_and_labels, weight_option, beta_val, -high_value, high_value, lattice_connections)
    neurons = preamble['neurons']
    connections_neuronwise = preamble['connections_neuronwise']
    weights_real = preamble['weights_real']
    weights_real_matrix = preamble['weights_real_matrix']
    weights_imaginary = preamble['weights_imaginary']
    weights_imaginary_matrix = preamble['weights_imaginary_matrix']
    weight_update_mask = jnp.ones((N, N)) * (1 - jnp.eye(N))
    pField = preamble['pField']
    uField = preamble['uField']
    beta = preamble['beta']
    inv_nudge_step = 1 / beta[outputn[0]]
    inv_nudge_step = preamble['inv_nudge_step']
    inv_batch_size = preamble['inv_batch_size']
    times = preamble['times']
    init_amplitudes = preamble['init_amplitudes']
    init_phases = preamble['init_phases']
    input_mask = preamble['input_mask']
    amplitude_relative = preamble['amplitude_relative']
    features = preamble['features']
    labels = preamble['labels']

    record_important_values(name, amplitude_relative, rng_key)

    if jnp.isnan(amplitude_relative):
        print('Amplitude relative equal to nan found. Restarting.')
        return 1
    print(f"\tAmplitude relative: \t{amplitude_relative}")
    

    features, labels, test_dataset_features, test_dataset_labels = OptDigits_separate_training_and_test(features, labels)

    # training the network
    for epoch in range(num_of_epochs):

        time0 = time.time()
        weight_real_gradient = jnp.zeros((N,N))
        weight_imaginary_gradient = jnp.zeros((N,N))
        bias_gradient = jnp.zeros(N)
        bias_phase_gradient = jnp.zeros(N)

        distance_temp = []
        accuracies_temp = []

        if (epoch+1)%1 == 0 or epoch==0:
            print(f"epoch number {epoch+1}")
        
        batches = shuffle_and_batch(features, labels, batch_size, rng_key)
        batch_time_0 = time.time()
        for batch_number, batch in enumerate(batches):
            print(f"\tbatch being processed: {batch_number} out of {len(batches)}")
            for feature, label in batch:

                target = jnp.zeros(N)
                target = target.at[outputn].set(label)

                uField = uField.at[inputn].set(feature)

                states = solve_ode_free((init_amplitudes, init_phases), times, weights_real, weights_imaginary, alpha, omega, pField, uField, connections_neuronwise, input_mask)
                amplitudes = states[0][-1]
                phases = states[1][-1]
                # removes nonstable solutions by restarting the thing
                if any(x > 1e-5 for x in (states[0][-10] - states[0][-1])):
                    if not T==400:
                        T=400 # first we try to make simulation time longer, if that doesn't work the parameters are discarted
                    else:
                        print(f"\tNonstable final state encountered! Restarting at epoch {epoch}")
                        return 1

                gradient_weights_real_forward, gradient_weights_imaginary_forward, gradient_biases_forward = calculate_energy_gradient(amplitudes, phases)


                # appending to training data arrays
                distance_temp.append(determine_distance(amplitudes, label, outputn))
                accuracies_temp.append(determine_accuracy(amplitudes, label, outputn, amplitude_relative, label_multiplier))

                if jnp.isnan(distance_temp[-1]):
                    print(f'distance found to be equal NaN, restarting at epoch {epoch}')
                    return 1

                # saving distances and accuracies once per inference
                distance = (sum_and_divide_array(distance_temp, batch_size))
                accuracy = (sum_and_divide_array(accuracies_temp, batch_size))
                label_as_index = jnp.argmax(label)
                save_single_value(distance, name + "training.txt")
                save_single_value(accuracy, name + "training_acc.txt")
                save_single_value(label_as_index, name + "training_label.txt")
                record_states(name + "training", amplitudes[outputn], phases[outputn])


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
                weight_real_gradient *= inv_batch_size*inv_nudge_step
            elif weight_type == 'i':
                weight_imaginary_gradient *= inv_batch_size*inv_nudge_step
            elif weight_type == 'c':
                weight_real_gradient *= inv_batch_size*inv_nudge_step
                weight_imaginary_gradient *= inv_batch_size*inv_nudge_step
            bias_gradient *= inv_batch_size*inv_nudge_step
            
            if normalize: #np.linalg.norm(weight_gradient,ord=1) > 1* N*N:
                #print(f"gradient absolute size exceeded expected value\nL(∇W) = {np.linalg.norm(weight_gradient, ord=1)}")
                #print("NORMALIZED!")
                if weight_type == 'r':
                    weight_real_gradient /= jnp.linalg.norm(weight_real_gradient,ord=2)
                elif weight_type == 'i':
                    weight_imaginary_gradient /= jnp.linalg.norm(weight_imaginary_gradient,ord=2)
                elif weight_type == 'c':
                    weight_real_gradient /= jnp.linalg.norm(weight_real_gradient,ord=2)
                    weight_imaginary_gradient /= jnp.linalg.norm(weight_imaginary_gradient,ord=2)
                bias_gradient /= jnp.linalg.norm(bias_gradient,ord=2)

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
        batch_time_1 = time.time()
        print(f"\tTime taken for batch: {batch_time_1-batch_time_0}")

        # Write (newline) separator once per epoch
        save_single_value('\n', name + "training.txt")
        save_single_value('\n', name + "training_acc.txt")
        save_single_value('\n', name + "training_label.txt")
        save_single_value('-'*10, name + "training_amplitudes.txt")

        # running inference on test dataset
        test_try = TestLoop(solve_ode_free, state, times, weights_real, weights_imaginary, alpha, omega, pField, uField, coupled_neuron, input_mask, name, outputn, test_dataset_features, test_dataset_labels)
 
        if test_try == 1:
            print("Error encountered during test inference!")
            return 1

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
        #save_array_to_file(jnp.array(distances), name + ".txt")
        #save_array_to_file(jnp.array(accuracies), name + "_acc.txt")
        aaaaa = 1

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
    parser.add_argument('beta_val', type=float, help="value of beta parameter")
    parser.add_argument('lattice_connections', type=str, help="type t for triangular or a for all-to-all")
    args = parser.parse_args()

    weight_option = {
        'diagonal':'nonzero',
        'eigenvalues':'all_positive',
        'nondiagonal':'rand'
    }

    high_value = 0.2/float(args.number)

    iterator = 0
    while True:
        # removes redundant files (all of them correspond to only one iteration of the code)
        name = args.name
        name += f"_version_{iterator}"
        iterator += 1
        answer = training_function(name, args.number, args.letter, args.num_of_epochs, args.learning_rate, args.weight_type, args.normalize, args.feature_multiplier, args.feature_constant, args.label_multiplier, weight_option, high_value, args.beta_val, args.lattice_connections)
        if answer == 0:
            break

if __name__ == "__main__":
    #main_dynamics_test()
    main()
