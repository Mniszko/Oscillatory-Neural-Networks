import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
import argparse
import time
from src import save_array_to_file, solve_SL_ode_free, solve_SL_ode_nudged, sum_and_divide_array, determine_SL_binary_distance, main_SL_training_preamble as main_training_preamble, XOR_problem_SL_determine_accuracy, XOR_problem_SL_map_features_and_labels, shuffle_and_batch, double_XOR_SL_map_features_and_labels, double_XOR_SL_determine_accuracy

jax.config.update("jax_enable_x64", True)

determine_distance = determine_SL_binary_distance

solve_ode_free = solve_SL_ode_free
solve_ode_nudged = solve_SL_ode_nudged

determine_accuracy = XOR_problem_SL_determine_accuracy
map_features_and_labels = XOR_problem_SL_map_features_and_labels

def main():
    parser = argparse.ArgumentParser(description="training")

    # Define the arguments
    parser.add_argument('number', type=int, help="Number of neurons.")
    parser.add_argument('T', type=float, help="evolution time")
    parser.add_argument('name', type=str, help="name of the output file without filetype")

    args = parser.parse_args()

    N = args.number
    T = args.T
    dt = 0.01
    omega = jnp.zeros(N)
    alpha = 1.
    batch_size = 4
    random_init_times = 1
    inputn = [0,1]
    outputn = [2]
    inputn = jnp.array(inputn)
    outputn = jnp.array(outputn)
    feature_multiplier, feature_constant, label_multiplier = 100., 30., 1./3.
    rng_key = jax.random.PRNGKey(round(time.time()*1e7))
    if N<3:
        return 1

    # preamble
    preamble = main_training_preamble(N, T, dt, omega, alpha, batch_size, random_init_times, inputn, outputn, rng_key, feature_multiplier, feature_constant, label_multiplier, map_features_and_labels)
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
    init_phases = preamble['init_phases'] * 0
    input_mask = preamble['input_mask']
    amplitude_relative = preamble['amplitude_relative']
    features = preamble['features']
    labels = preamble['labels']

    all_solutions = []

    time0 = time.time()
    for feature, label in zip(features, labels):

        uField = uField.at[inputn].set(feature)

        solution_full = solve_ode_free((init_amplitudes, init_phases), times, weights_real, weights_imaginary, alpha, omega, pField, uField, connections_neuronwise, input_mask)

        all_solutions.append(solution_full)

        print(f"output vs label:\n\t{solution_full[0][-1][outputn[0]]} ---- {label[0]}")
        print(f"input parameter vs features:\n\t{solution_full[0][-1][inputn[0]]} ---- {feature[0]}")
        print(f"\t{solution_full[0][-1][inputn[1]]} ---- {feature[1]}")

    colors = ["green" for _ in range(N)]
    for i in range(len(inputn)):
        colors[inputn[i]] = "red" 
    for i in range(len(outputn)):
        colors[outputn[i]] = "blue"

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 24))  
    for row_axes in axes:
        for ax in row_axes:
            ax.grid(True)

    for i, solution_full in enumerate(all_solutions):
        for j in range(N):
            # amplitudes
            axes[i, 0].plot(
                jnp.arange(0, len(solution_full[0]) * dt, dt),  
                solution_full[0][:, j],
                label=f'Oscillator {j+1}',
                color=colors[j]  
            # phases
            )
            axes[i, 1].plot(
                jnp.arange(0, len(solution_full[1]) * dt, dt),  
                jnp.sin(solution_full[1][:, j]),
                label=f'Oscillator {j+1}',
                color=colors[j]  
            )

    print(f"Time taken for the opperation: {time.time()-time0}")
    print(f"input mask used: {input_mask}")
    plt.tight_layout() 
    plt.savefig(f'{args.name}.png')
    plt.show()
    return

if __name__ == "__main__":
    main()