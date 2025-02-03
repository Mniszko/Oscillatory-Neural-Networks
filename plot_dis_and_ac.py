import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
import colorsys
import csv
import argparse

from src import read_and_plot

def main():
    parser = argparse.ArgumentParser(description="Training")

    # Define the arguments
    parser.add_argument('data_name', type=str, help="name of '.txt' data without suffix (accuracies should have '_acc' added to filename)")
    parser.add_argument('plot_dis', type=str, choices=['y', 'n'], help="Plot distances? ('y' or 'n').")
    parser.add_argument('dis_scale', type=str, choices=['log', 'lin'], help="Distance scale ('log' or 'lin').")
    parser.add_argument('plot_acc', type=str, choices=['y', 'n'], help="Plot accuracies? ('y' or 'n').")
    parser.add_argument('plot_name', type=str, help="Name to save plot as (must end with '.svg').")

    args = parser.parse_args()

     # Determine number of subplots needed
    num_plots = (args.plot_dis == 'y') + (args.plot_acc == 'y')

    if num_plots == 0:
        print("No plots will be created as for both plot_dis and plot_acc, 'n' is parsed.")
        return

    # Create subplots: 1 or 2 horizontally
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))  # Adjust figure size
    if num_plots == 1:
        axs = [axs]  # Ensure axs is always a list for consistency

    plot_idx = 0

    # Plot distances if required
    if args.plot_dis == 'y':
        print("Creating distance plot...")
        try:
            read_and_plot(args.data_name+".txt", axs[plot_idx], x_range=(0, 100), y_bins=60, x_bins=60,
                          plot_title="Distances", multiplicator=1, plot_scale=args.dis_scale)
            plot_idx += 1
        except Exception as e:
            print(f"Error while creating distance plot: {e}")

    # Plot accuracies if required
    if args.plot_acc == 'y':
        print("Creating accuracy plot...")
        try:
            read_and_plot(args.data_name+"_acc.txt", axs[plot_idx], x_range=(0, 100), y_bins=10, x_bins=60,
                          plot_title="Accuracies", multiplicator=1, plot_scale="lin")
        except Exception as e:
            print(f"Error while creating accuracy plot: {e}")

    # Save the plot
    try:
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(args.plot_name)
        print(f"Plot saved as {args.plot_name}")
    except Exception as e:
        print(f"Error while saving plot: {e}")
    plt.show()

if __name__ == "__main__":
    main()
