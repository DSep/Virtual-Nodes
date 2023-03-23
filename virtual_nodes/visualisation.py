import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import make_interp_spline, BSpline


def visualise_graph(g: torch.tensor, filename=None):
    '''
    Args:

    (+) g: (torch.tensor): (N, N) unweighted adjacency matrix.
    '''
    g = g.numpy()
    G = nx.DiGraph(g)
    nx.draw_circular(G, with_labels=True, font_weight='bold')
    # nx.draw(G, with_labels=True, font_weight='bold')

    if filename:
        plt.savefig(filename)
    plt.close()


def plot_smooth_curve(tensor, title=''):
    N = tensor.shape[0]
    x = torch.arange(1, N+1)
    y = tensor

    # Generate a smoother x-axis for the curve
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # Create a spline function to generate a smooth curve
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)

    # Plot the smooth curve
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Values')
    plt.title(title)
    plt.show()


def plot_smooth_curves(tensor, x=None, xlabel='', ylabel='', filename=None):
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(1)

    N, D = tensor.shape

    if x is None:
        x = torch.arange(1, N+1)
    elif x.shape[0] != D:
        raise ValueError("The length of x must match the first dimension of the input tensor")

    # Generate a smoother x-axis for the curve
    # x_smooth = np.linspace(x.min(), x.max(), 300)

    for i in range(N):
        y = tensor[i, :]

        # Create a spline function to generate a smooth curve
        # spline = make_interp_spline(x, y, k=3)
        # y_smooth = spline(x_smooth)

        # Plot the smooth curve
        plt.plot(x, y, label=f'Label {i+1}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('')
    plt.legend()
    plt.show()

    if filename:
        plt.savefig(filename)
    
    plt.close()


def plot_heatmap(tensor, xlabel, ylabel, filename=None):
    # Convert the PyTorch tensor to a NumPy array
    np_tensor = tensor.numpy()
    
    # Set the color map for the heatmap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create the heatmap using Seaborn
    ax = sns.heatmap(np_tensor, cmap=cmap, annot=True, fmt=".2f", vmin=-1, vmax=1)
    
    # Set the axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Save the plot to the file if a filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    else:
        # Show the plot
        plt.show()

    plt.close()

def visualise_neighbourhood_label_dist(dist: torch.tensor,
                                       dist_prime: torch.tensor = None,
                                       mu_dist: torch.tensor = None, 
                                       filename = None,
                                       show = False):
    '''
    Plot normalised bincount (bar chart) distribution over the neighbourhood
    label distribution represented in `dist`. If `dist` or `mu_dist` are
    provided, they will be plotted comparatively against `dist`.
    '''

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the bar plot using dist
    K = len(dist)
    ax.bar(range(K), dist, label="Original Distribution")

    # Plot dist_prime if provided
    if dist_prime is not None:
        ax.plot(range(K), dist_prime, label="New Distribution", color="red", linewidth=2)

    # Plot mu_dist if provided
    if mu_dist is not None:
        ax.plot(range(K), mu_dist, label="Mean Distribution", color="green", linewidth=2)

    # Add legend and labels
    ax.legend()
    ax.set_xlabel("Label")
    ax.set_ylabel("Probability")

    # Save figure to file if filename is provided
    if filename is not None:
        plt.savefig(filename)

    # Show the plot
    if show:
        plt.show()

    plt.close()
