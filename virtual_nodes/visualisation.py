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


