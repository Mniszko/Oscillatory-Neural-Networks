import numpy as np
import matplotlib.pyplot as plt
import argparse


def check_nan_values(main_data, pha_data, amp_data):
    # Check for NaN values in main_data
    main_nan_cases = [i for i, case in enumerate(main_data) if np.isnan(case).any()]
    if main_nan_cases:
        print(f"Main Data: NaN values found in cases {main_nan_cases}")
    else:
        print("Main Data: No NaN values found in any case.")

    # Check for NaN values in pha_data
    pha_nan_cases = []
    for case_number, case in enumerate(pha_data):
        if any(np.isnan(row).any() for row in case):
            pha_nan_cases.append(case_number)
    if pha_nan_cases:
        print(f"Phase Data: NaN values found in cases {pha_nan_cases}")
    else:
        print("Phase Data: No NaN values found in any case.")

    # Check for NaN values in amp_data
    amp_nan_cases = []
    for case_number, case in enumerate(amp_data):
        if any(np.isnan(row).any() for row in case):
            amp_nan_cases.append(case_number)
    if amp_nan_cases:
        print(f"Amplitude Data: NaN values found in cases {amp_nan_cases}")
    else:
        print("Amplitude Data: No NaN values found in any case.")

def read_state_data(filename):
    with open(filename, 'r') as file:
        data = file.read().split('-' * 40 + '\n')
    # For each case, split into rows, convert to floats, transpose rows/columns, and filter elements
    return [
        [[row[i] for i in range(1, len(row), 4)] for row in zip(*[list(map(float, line.split(','))) for line in case.strip().split('\n') if line.strip()])]
        for case in data if case.strip()
    ]

"""
def read_state_data(filename):
    datasets = []  # List to store all datasets
    current_dataset = []  # Temporary storage for the current dataset
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Check for the separator line (40 dashes)
            if line == '-' * 40:
                if current_dataset:
                    # Transpose rows to columns and store the dataset
                    datasets.append(list(map(list, zip(*current_dataset))))
                    current_dataset = []  # Reset for the next dataset
            else:
                # Convert line into a list of values and store it
                current_dataset.append([float(value) for value in line.split(',')])
    
    # Ensure the last dataset is added if the file does not end with dashes
    if current_dataset:
        datasets.append(list(map(list, zip(*current_dataset))))
    
    return datasets
"""

def read_amp_data(data_name):
    return read_state_data(data_name+'_amp.txt')

def read_pha_data(data_name):
    return read_state_data(data_name+'_pha.txt')

def read_dis_data(filename):
    y_arrays = []
    with open(filename+'.txt', "r") as file:
        for line in file:
            # Convert the line to a numpy array of y-values
            y_array = np.array([float(num) for num in line.strip().split()])
            y_arrays.append(y_array)  # Append the y-array to the list
    return y_arrays

def read_acc_data(filename):
    y_arrays = []
    with open(filename+'_acc.txt', "r") as file:
        for line in file:
            # Convert the line to a numpy array of y-values
            y_array = np.array([float(num) for num in line.strip().split()])
            y_arrays.append(y_array)  # Append the y-array to the list
    return y_arrays

def shift_phases(phases, index=0):
    for j in range(len(phases)):
        for i in range(len(phases[j])):
            phases[j][i] -= phases[j][0]
    return phases

def adjust_rows(data, index=0):
    """
    Adjusts each row in the data by adding the first element of the row to all elements in that row.

    Parameters:
        data (list of list of list of float): The input data (output of read_state_data).

    Returns:
        list of list of list of float: The adjusted data.
    """
    adjusted_data = []
    for case in data:
        adjusted_case = []
        for row in case:
            first_element = row[index]  # Get the first element of the row
            adjusted_row = [x + first_element for x in row]  # Add it to all elements in the row
            adjusted_case.append(adjusted_row)
        adjusted_data.append(adjusted_case)
    return adjusted_data

def plot_single(data_x, data_y, ax, plot_scale='lin', y_max=None, plot_title="Line Plot", ylabel='Ylabel'):
    # Determine the y-axis limits
    actual_y_max = min(max(data_y), y_max) if y_max is not None else max(data_y)
    y_min = min(data_y)

    # Set the y-axis scale
    if plot_scale == 'log':
        ax.set_yscale('log')
        print(f"The effective y-axis maximum is set to: {actual_y_max}")
    elif plot_scale == 'lin':
        ax.set_yscale('linear')

    # Plot the line
    ax.plot(data_x, data_y, color='blue', linewidth=2, label='Data')

    # Set labels and title
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    ax.set_title(plot_title)
    return ax

def plot_multiple_lines(all_x, all_y, colors, ax, plot_scale='lin', y_max=None, plot_title="Multiple Lines", ylabel='Ylabel'):
    """
    Plots multiple lines on the same subplot with corresponding colors.

    Parameters:
        all_x (list of numpy arrays): Array of x-values for each line.
        all_y (list of numpy arrays): Array of y-values for each line.
        colors (list of str): Array of colors for each line.
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        plot_scale (str): Scale of the y-axis ('lin' or 'log').
        y_max (float): Maximum y-axis limit (optional).
        plot_title (str): Title of the plot.
    """
    # Validate inputs
    if len(all_x) != len(all_y) or len(all_y) != len(colors):
        raise ValueError("Lengths of all_x, all_y, and colors must be the same.")

    # Determine the global y-axis limits
    all_y_flat = np.concatenate(all_y)
    actual_y_max = min(max(all_y_flat), y_max) if y_max is not None else max(all_y_flat)
    y_min = min(all_y_flat)

    # Set the y-axis scale
    if plot_scale == 'log':
        ax.set_yscale('log')
        print(f"The effective y-axis maximum is set to: {actual_y_max}")
    elif plot_scale == 'lin':
        ax.set_yscale('linear')

    # Plot each line with its corresponding color
    for x, y, color in zip(all_x, all_y, colors):
        ax.plot(x, y, color=color, linewidth=2)

    # Set labels and title
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    ax.set_title(plot_title)


    return ax

def plot_dis_single(data_x, data_y, ax, y_max=None, plot_scale='lin'):
    return plot_single(data_x, data_y, ax, plot_scale, y_max=None, plot_title='Distance Plot', ylabel='Distance')

def plot_acc_single(data_x, data_y, ax, y_max=None):
    return plot_single(data_x, data_y, ax, 'lin', y_max=None, plot_title='Accuracy Plot', ylabel='Accuracy')

def plot_dis_multiple(all_x, all_y, ax, y_max=None, plot_scale='lin'):
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_x)))
    return plot_multiple_lines(all_x=all_x, all_y=all_y, colors=colors, ax=ax,plot_scale=plot_scale, y_max=y_max, plot_title='Distance Plot', ylabel='Distance')
def plot_acc_multiple(all_x, all_y, ax, y_max=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_x)))
    return plot_multiple_lines(all_x=all_x, all_y=all_y, colors=colors, ax=ax,plot_scale='lin', y_max=y_max, plot_title='Accuracy Plot', ylabel='Accuracy')

def plot_all_histo(data_x, data_y, ax, plot_scale='lin', x_bins=10, y_bins=10, x_range=None, y_max=None, plot_title="Histogram"):
    # Determine the y-axis limits
    actual_y_max = min(max(all_y), y_max) if y_max is not None else max(all_y)
    y_min = min(all_y)

    # Define the x-axis range if not provided
    if x_range is None:
        x_range = (min(all_x), max(all_x))

    # Define the x-axis bins
    x_space = np.linspace(x_range[0], x_range[1], x_bins + 1)

    # Define the y-axis bins based on the scale
    if plot_scale == 'log':
        y_space = np.logspace(np.log10(y_min), np.log10(actual_y_max), y_bins + 1)
        print(f"The effective y-axis maximum is set to: {actual_y_max}")
        ax.set_yscale('log')
    elif plot_scale == 'lin':
        y_space = np.linspace(y_min, actual_y_max, y_bins + 1)
        ax.set_yscale('linear')

    ax.set_ylabel("Distance")

    # Compute the 2D histogram of counts
    counts, xedges, yedges = np.histogram2d(
        all_x, all_y, bins=[x_space, y_space], range=[x_range, (y_min, actual_y_max)]
    )

    # Plot the histogram
    im = ax.hist2d(all_x, all_y, bins=(x_space, y_space), cmap="viridis", range=[x_range, (y_min, actual_y_max)])
    ax.set_title(plot_title)
    ax.set_xlabel("Epoch")

    # Add a colorbar (requires the figure to add it correctly)
    cbar = ax.figure.colorbar(im[3], ax=ax)
    cbar.set_label("Counts")

    # Calculate and plot the mean line
    unique_x = np.unique(all_x)
    mean_y_values = [np.mean(all_y[all_x == ux]) for ux in unique_x]
    ax.plot(unique_x, mean_y_values, color='white', linewidth=2, label='Mean Value')

    return ax

def plot_all_dis_histo(data_x, data_y, ax, x_bins=100, y_bins=100, x_range=None, y_max=None, plot_scale='lin'):
    return plot_all_histo(data_x, data_y, ax, plot_scale, x_bins, x_range, y_max, "Distance")

def plot_all_acc_histo(data_x, data_y, ax, x_bins=100, y_bins=100, x_range=None, y_max=None):
    return plot_all_histo(data_x, data_y, ax, 'lin', x_bins, x_range, y_max, "Accuracy")

def plot_dataset(x_data, data_y, ax):
    neurons = len(data_y)
    for neuron_y in data_y:
        ax.plot(x_data, np.array(neuron_y))
    return ax

def plot_pha_single(data_x, data_y, ax):
    #corrected_y = []
    #for neuron_y in data_y:
    #    corrected_y.append(np.array(neuron_y)-neuron_y[0])
    plot_dataset(data_x, np.cos(np.array(data_y)), ax)
    ax.set_label(r"$Cos(\theta)$")
    ax.set_title('r"$Cos(\theta)$"')
    return ax
def plot_amp_single(data_x, data_y, ax):
    plot_dataset(data_x, data_y, ax)
    ax.set_label(r"$\rho$")
    ax.set_title(r"$\rho$")
    return ax

def plot_pha_mean():
    return

def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('name', type=str, help="name of file without suffixes")
    parser.add_argument('index', type=int, help="index of given training")
    args = parser.parse_args()

    name = args.name
    index = args.index

    distances = read_dis_data(name)
    accuracies = read_acc_data(name)
    amplitudes = read_amp_data(name)
    phases = read_pha_data(name)
    #phases = adjust_rows(phases)


    check_nan_values(distances, phases, amplitudes)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    x_array = np.arange(len(distances[index]))
    # plot dis single działa dobrze
    plot_dis_single(x_array, distances[index], axs[0, 0], y_max=5, plot_scale='lin')
    plot_acc_single(x_array, accuracies[index], axs[1, 0], y_max=5)
    if amplitudes != []:
        plot_amp_single(x_array, amplitudes[index], axs[0, 1])
    else:
        pass
    if phases != []:
        plot_pha_single(x_array, phases[index], axs[1, 1])
    else:
        pass
    #plot_dis_multiple(np.array([x_array, x_array, x_array]), distances[0:3], axs[0, 0])
    #plot_acc_multiple(np.array([x_array, x_array, x_array]), accuracies[0:3], axs[1, 0])


    plt.tight_layout()
    plt.savefig(name + ' ' + str(index) + ".png")
    #plt.show()
    return

if __name__ == '__main__':
    main()