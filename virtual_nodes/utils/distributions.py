import torch
import networkx as nx
import numpy as np


def compute_avg_node_degree(g):
    '''
    Computes the mean node in degree and standard deviation.

    Args:

    (+) g (torch.tensor): (N, N) unweighted adjacency matrix.

    Returns:

    (+) mean_in_degree (float): The mean node degree.
    (+) std_in_degree (float): The standard deviation of the node degrees.
    '''
    G = nx.DiGraph(g.numpy())

    # Calculate the degree of each node
    degrees = [degree for node, degree in G.in_degree]

    # Calculate the mean and std of the degrees
    mean_in_degree = np.mean(degrees)
    std_in_degree = np.std(degrees)

    return mean_in_degree, std_in_degree


def calculate_neighbour_distribution(edge_index, x):
    ''''''
    pass