# import numpy as np
# from scipy.integrate import odeint
import jax as jx
import jax.numpy as jnp
from jax.experimental.ode import odeint
import csv

def read_and_plot(filename, ax, x_range=(0, 100), y_bins=50, x_bins=100, y_max=5, plot_title="", multiplicator=1, plot_scale="lin"):
    """
    Reads arrays from a file, accumulates counts in 2D bins, and plots as a binplot on a given axis.
    The y-axis is limited to either `y_max` or the highest y value in the data, whichever is lower.

    Parameters:
    - filename (str): The name of the file to read data from.
    - ax (matplotlib.axes.Axes): The axis on which to plot the data.
    - x_range (tuple): The range of x-axis values for binning.
    - y_bins (int): The number of bins along the y-axis.
    - x_bins (int): The number of bins along the x-axis.
    - y_max (float): The maximum y value limit.
    - plot_title (str): The title for the plot.
    - multiplicator (float): A multiplier for the y values.
    - plot_scale (str): The scale for the y-axis ('lin' or 'log').
    """

    # Initialize lists to accumulate all x and y values across arrays
    all_x = []
    all_y = []

    if plot_scale not in ["log", "lin"]:
        print("---------------\n\tInvalid plot_scale variable parsed!\n---------------")
        return

    # Track the maximum length of y arrays for setting up x-axis
    length_y = 0

    with open(filename, "r") as file:
        for line in file:
            # Convert the line to a numpy array of y-values
            y_array = np.array([float(num) for num in line.strip().split()]) * multiplicator
            length_y = max(length_y, len(y_array))  # Update the length_y if this y_array is longer
            
            # Assuming x array is a sequence (0, 1, ..., len(y_array)-1) for each y_array
            x_array = np.arange(len(y_array))
            
            # Append these x and y values to the lists
            all_x.extend(x_array)
            all_y.extend(y_array)

    # Set up the x and y bins
    x_space = np.linspace(0, length_y, x_bins + 1)

    # Determine the effective maximum y limit
    actual_y_max = min(max(all_y), y_max)

    if plot_scale == "log":
        y_space = np.logspace(np.log10(min(all_y)), np.log10(actual_y_max), y_bins + 1)
        print(f"The effective y-axis maximum is set to: {actual_y_max}")
        ax.set_yscale('log')
        ax.set_ylabel("Value (log scale)")
    else:  # plot_scale == "lin"
        y_space = np.linspace(min(all_y), actual_y_max, y_bins + 1)
        ax.set_yscale('linear')
        ax.set_ylabel("Value")

    # Compute the 2D histogram of counts
    counts, xedges, yedges = np.histogram2d(
        all_x, all_y, bins=[x_space, y_space], range=[x_range, (min(all_y), actual_y_max)]
    )

    # Plot the histogram
    ax.hist2d(all_x, all_y, bins=(x_space, y_space), cmap="viridis")
    ax.set_title(plot_title)
    ax.set_xlabel("Epoch")

    # Add a colorbar (requires the figure to add it correctly)
    cbar = ax.figure.colorbar(ax.collections[0], ax=ax)
    #cbar.set_label()

    # Calculate and plot the mean line
    unique_x = np.unique(all_x)
    mean_y_values = [np.mean([y for x, y in zip(all_x, all_y) if x == ux]) for ux in unique_x]
    ax.plot(unique_x, mean_y_values, color='white', linewidth=2, label='Mean Value')


def clear_csv(filename):
    """
    Clears the contents of the CSV file.
    
    :param filename: str, name of the CSV file to clear
    """
    with open(filename, mode='w', newline='') as file:
        pass

def save_array_to_file(array, filename="savepoint.txt"):
    """
    Appends a numpy array as a new line in a specified file.
    """
    with open(filename, "a") as file:
        # Convert the array to a string representation and append a newline
        file.write(" ".join(map(str, array.flatten())) + "\n")

def append_to_csv(filename, data_array):
    """
    Appends a single array (row) to the CSV file.
    
    :param filename: str, name of the CSV file
    :param data_array: list, the array to be appended as a row in the CSV
    """
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_array)

def save_parameters(weights_matrix, weights, biases, bias_phases, filename):
    """
    Save weights, biases, and bias_phases to a .npz file.
    
    Parameters:
        weights (jnp.ndarray): The weights matrix.
        biases (jnp.ndarray): The biases array.
        bias_phases (jnp.ndarray): The bias phases array.
        filename (str): The filename to save the parameters.
    """
    jnp.savez(filename, weights_matrix=weights_matrix, weights=weights, biases=biases, bias_phases=bias_phases)

def load_parameters(filename):
    """
    Load weights, biases, and bias_phases from a .npz file.
    
    Parameters:
        filename (str): The filename to load the parameters from.
    
    Returns:
        tuple: A tuple containing the weights, biases, and bias_phases arrays.
    """
    data = jnp.load(filename+'.npz')
    weights_matrix = data['weights_matrix']
    weights = data['weights']
    biases = data['biases']
    bias_phases = data['bias_phases']
    return weights_matrix, weights, biases, bias_phases

# unused function; energy gradient function has to be defined inside main file as for recursive inference of proper mapping to work
def calculate_energy_gradient(state, gradientWeights, N, gradientBiases = False):
    phases = state[N:2*N]
    densities = state[0:N]
    for i in range(1,N):
        for j in range(0,i):
            gradientWeights[i][j] = -densities[i]*densities[j]*jnp.cos(phases[i]-phases[j])
            gradientWeights[j][i] = gradientWeights[i][j] 
    if type(gradientBiases) != bool:
        for i in range(1,N):
            gradientBiases[i] = -densities[i]*densities[i]
    return gradientWeights, gradientBiases
