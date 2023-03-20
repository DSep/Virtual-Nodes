import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def visualise_graph(g: torch.tensor, filename=None):
    '''
    Args:

    (+) g: (torch.tensor): (N, N) unweighted adjacency matrix.
    '''
    g = g.numpy()
    G = nx.DiGraph(g)
    nx.draw_circular(G, with_labels=True, font_weight='bold')

    if filename:
        plt.savefig(filename)
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

    