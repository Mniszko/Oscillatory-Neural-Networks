import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import argparse
import time
from src import solve_K_ode_free, solve_K_ode_nudged, sum_and_divide_array, compute_kuramoto_gradients as compute_gradients, create_kuramoto_symmetric_weights as create_symmetric_weights, XOR_problem_K_map_features_and_labels,  XOR_problem_K_determine_accuracy, double_XOR_K_map_features_and_labels, double_XOR_K_determine_accuracy, determine_K_binary_distance, main_K_training_preamble as main_training_preamble, shuffle_and_batch

jax.config.update("jax_enable_x64", True)

determine_distance = determine_K_binary_distance

solve_ode_free = solve_K_ode_free
solve_ode_nudged = solve_K_ode_nudged

determine_accuracy = XOR_problem_K_determine_accuracy
map_features_and_labels = XOR_problem_K_map_features_and_labels

"""
determine_accuracy = double_XOR_K_determine_accuracy
map_features_and_labels = double_XOR_K_map_features_and_labels
"""

def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="A program to demonstrate command-line arguments.")
    
    # Define the arguments
    parser.add_argument('name', type=str, help="Filenames of output data.")
    parser.add_argument('number', type=int, help="An integer number.")
    parser.add_argument('letter', type=str, help="A letter (string).")
    parser.add_argument('num_of_epochs', type=int, help="Number of epochs.")
    parser.add_argument('learning_rate', type=float, help="Learning rate.")
    parser.add_argument('normalize', type=bool, help="Normalize gradients during gradient descent according to internal rules: 0 or 1 (false or true)")

    # Parse the arguments
    args = parser.parse_args()

    name = args.name
    N = args.number
    do_save = args.letter
    num_of_epochs = args.num_of_epochs
    learning_rate = args.learning_rate
    normalize = args.normalize
    rng_key = jax.random.PRNGKey(round(time.time()*1e7))
    
    if N<3:
        return 1

    outputn = [2]
    inputn = [0, 1]
    inputn = jnp.array(inputn)
    outputn = jnp.array(outputn)

    distances = []
    accuracies = []

    T = 100
    dt = 0.01
    random_init_times = 1
    batch_size = 4

    preamble = main_training_preamble(N, T, dt, batch_size, random_init_times, inputn, outputn, rng_key, map_features_and_labels)
    neurons = preamble['neurons']
    connections_neuronwise = preamble['connections_neuronwise']
    weights = preamble['weights']
    weights_matrix = preamble['weights_matrix']
    weight_update_mask = preamble['weight_update_mask']
    biases = preamble['biases']
    bias_phases = preamble['bias_phases']
    beta = preamble['beta']
    inv_nudge_step = preamble['inv_nudge_step']
    inv_batch_size = preamble['inv_batch_size']
    inv_random_init_times = preamble['inv_random_init_times']
    times = preamble['times']
    init_phases = preamble['init_phases']
    input_mask = preamble['input_mask']
    features = preamble['features']
    labels = preamble['labels']

    phases = jnp.copy(init_phases)

    # training the network
    for epoch in range(num_of_epochs):

        time0 = time.time()
        if epoch%100 == 0:
            print(f"Starting epoch number {epoch}")

        weight_gradient = jnp.zeros((N,N))
        bias_gradient = jnp.zeros(N)
        bias_phase_gradient = jnp.zeros(N)
        
        distance_temp = []
        accuracies_temp = []

        batches = shuffle_and_batch(features, labels, batch_size, rng_key)
        for batch in batches:
            for feature, label in batch:
            
                target = jnp.zeros(N)
                target = target.at[outputn].set(label)

                # inserting input values and random initialization
                phases = jax.random.uniform(rng_key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)
                for i,neuron in enumerate(inputn):
                    phases = phases.at[neuron].set(feature[i])

                # solve problem in first step and calculate energies
                thetas = solve_ode_free(phases, times, weights, biases, bias_phases, connections_neuronwise, input_mask)
                phases = thetas[-1]
                gradient_weights_forward, gradient_biases_forward, gradient_bias_phases_forward = compute_gradients(
                    phases, weights, biases, bias_phases
                )
                distance_temp.append(determine_distance(phases, outputn, label))
                accuracies_temp.append(determine_accuracy(phases, label, outputn))

                # calculating nudge of inference and its energy
                thetas_back = solve_ode_nudged(phases, times, weights, biases, bias_phases, connections_neuronwise, input_mask, beta, target)
                phases = thetas_back[-1]
                gradient_weights_backward, gradient_biases_backward, gradient_bias_phases_backward = compute_gradients(
                    phases, weights, biases, bias_phases
                )

                # calculating gradient
                weight_gradient += gradient_weights_backward - gradient_weights_forward
                bias_gradient += gradient_biases_backward - gradient_biases_forward
                bias_phase_gradient += gradient_bias_phases_backward - gradient_bias_phases_forward

            # parameter updates
            weight_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
            bias_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times
            bias_phase_gradient *= inv_nudge_step * inv_batch_size * inv_random_init_times

            # normalization (required for first few steps)
            if normalize:
                #print(f"gradient absolute size exceeded expected value\nL(∇W) = {np.linalg.norm(weight_gradient, ord=2)}")
                weight_gradient /= jnp.linalg.norm(weight_gradient,ord=2)
                bias_gradient /= jnp.linalg.norm(bias_gradient,ord=2)
                bias_phase_gradient /= jnp.linalg.norm(bias_phase_gradient,ord=2)

            weights_matrix -= learning_rate * weight_gradient * weight_update_mask
            weights = weights_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
            biases -= learning_rate * bias_gradient
            bias_phases -= learning_rate * bias_phase_gradient
        
        # calculating distance
        distances.append(sum_and_divide_array(distance_temp, batch_size))
        accuracies.append(sum_and_divide_array(accuracies_temp, batch_size))

        time1 = time.time()        
        if (epoch+1)%100 == 0 or epoch==0:
            print(f"Finished epoch number {epoch+1}")
            print(f"time taken for the epoch: {time1-time0}")
        
        """
        # debugging
        if (epoch+1)%100 == 0 or epoch==0:
            print(f"Finished epoch number {epoch+1}")
            print(f"distances read {jnp.array(distance_temp).tolist()}")
            print(f"accuracies read {jnp.array(accuracies_temp).tolist()}")
            print(f"time taken for the epoch: {time1-time0}")
        """
        
    if do_save=="y" or do_save=="yes":
        save_array_to_file(jnp.array(distances), name + ".txt")
        save_array_to_file(jnp.array(accuracies), name + "_acc.txt")

    # plotting
    elif do_save=="n" or do_save=="no":
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        axes[0].plot(distances, label="distance", c="r")
        axes[0].set_title("distances")
        axes[0].grid()
        axes[1].plot(accuracies, c="b")
        axes[1].set_title("accuracies")
        axes[1].grid()
        plt.savefig("XOR_SLNN_distances.png")
        plt.show()
    
    return 0

if __name__ == "__main__":
    main()