import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import colorsys
import os
import imageio
import re
import itertools
import pandas as pd
import csv
import time
import plotly.graph_objects as go
import sys
import argparse
plt.rcParams.update({'font.size': 18})
sys.path.append('./src')
from basics import save_array_to_file, network_evolution, network_evolution_nudge, calculateEnergyGradient
from StuartLandauNeuralNet import solve_ode_free, create_symmetric_weights, main_training_preamble, create_random_connections, create_square_lattice_connections, shuffle_and_batch


def main():
    parser = argparse.ArgumentParser(description="training")

    # Define the arguments
    parser.add_argument('number', type=int, help="An integer number.")
    parser.add_argument('T', type=float, help="max time")

    args = parser.parse_args()

    N = args.number
    T = args.T
    rng_key = jax.random.PRNGKey(round(time.time()*1e7))

    #N = int(input("Enter number of (fully connected) neurons (minimum stands at N = 5):\t"))
    if N<3:
        return 1

    # preamble
    neurons = jnp.arange(0,N,1)
    outputn = 2
    inputn = [0,1]
    connections_neuronwise = jnp.array([
    [element for element in neurons if element != neuron]
    for neuron in neurons
    ])
    weights_matrix = create_symmetric_weights(N, 0., 1.,  inputn, rng_key)
    weight_update_mask = jnp.ones_like(weights_matrix)
    for i in inputn:
        weight_update_mask = weight_update_mask.at[i, inputn].set(0)
        weight_update_mask = weight_update_mask.at[inputn, i].set(0)
    weights = weights_matrix[connections_neuronwise, jnp.arange(N)[:, None]]
    #pField = jnp.array([np.random.uniform(-1, 1)/2 for _ in neurons])+2
    pField = jnp.zeros(N)
    # here uField acts as bias
    uField = jax.random.uniform(rng_key, shape=(N,), minval=-3, maxval=3)
    alpha = 1.
    omega = jnp.zeros(N)

    beta = jnp.zeros(N)
    beta = beta.at[outputn].set(1e-6)

    batch_size = 4
    random_init_times = 1

    amplitude_relative = 1

    features = jnp.array([
        [-1,-1],
        [1,-1],
        [-1,1],
        [1,1]
    ])*20
    labels = jnp.array([-1,1,1,-1])/5 + amplitude_relative

    #input_mask = jnp.array([0 if neuron in inputn else 1 for neuron in neurons])
    input_mask = jnp.ones(N)

    all_solutions = []

    time0 = time.time()
    for feature, label in zip(features, labels):
    #feature = jnp.array([1,-1])*5
    #label = 1/2 + amplitude_relative

        uField = uField.at[jnp.array(inputn)].set([feature[0], feature[1]])
        # assigning xi values
        #weights = weights.at[inputn].at(2).set([feature[0], feature[1]])
        time_step = 0.01
        times = jnp.arange(0, T+time_step, time_step)

        init_amplitudes = jax.random.uniform(rng_key, shape=(N,), minval=-1, maxval=1)/2 + amplitude_relative
        init_phases = jax.random.uniform(rng_key, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)/100

        state = (jnp.copy(init_amplitudes), jnp.copy(init_phases))

        arguments = (weights, alpha, omega, pField, uField, connections_neuronwise, input_mask)

        solution_full = solve_ode_free(state, times, *arguments)
        all_solutions.append(solution_full)
        print(f"output vs label: {solution_full[0][-1][outputn]} ---- {label}")
        print(f"input vs features:\n\t{solution_full[0][-1][inputn[0]]} ---- {feature[0]}")
        print(f"\t{solution_full[0][-1][inputn[1]]} ---- {feature[1]}")

    colors = ["green" for _ in range(N)]  # Use a standard Python list
    colors[inputn[0]] = "red"  # Direct assignment for inputn[0]
    colors[inputn[1]] = "red"  # Direct assignment for inputn[1]
    colors[outputn] = "blue"   # Direct assignment for outputn

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 24))  # Adjust the figure size for 4 rows and 2 columns
    for row_axes in axes:
        for ax in row_axes:
            ax.grid(True)

    for i, solution_full in enumerate(all_solutions):
        for j in range(N):
            # Plot amplitudes in the first column
            axes[i, 0].plot(
                jnp.arange(0, len(solution_full[0]) * time_step, time_step),  # Use len(solution_full[0])
                solution_full[0][:, j],
                label=f'Oscillator {j+1}',
                color=colors[j]  # Use the appropriate color
            )
            # Plot phases in the second column
            axes[i, 1].plot(
                jnp.arange(0, len(solution_full[1]) * time_step, time_step),  # Use len(solution_full[1])
                jnp.sin(solution_full[1][:, j]),
                label=f'Oscillator {j+1}',
                color=colors[j]  # Use the appropriate color
            )

    print(f"Time taken for the opperation: {time.time()-time0}")
    print(f"input mask used: {input_mask}")
    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig('example_evolution.png')
    plt.show()
    return

if __name__ == "__main__":
    #main_dynamics_test()
    main()