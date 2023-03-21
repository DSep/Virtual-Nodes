import torch

from virtual_nodes.augment import compute_neighbourhood_feature_label_distribution, add_vnodes, compute_differences, unmask_graph, add_undirected_vnodes_to_graph
from virtual_nodes.visualisation import visualise_graph


# Define a simple graph
simple_g = torch.tensor([[0, 1, 1, 0, 0],
                         [1, 0, 0, 1, 1],
                         [1, 0, 0, 0, 1],
                         [0, 1, 0, 0, 1],
                         [0, 1, 1, 1, 0]]) # .to(torch.float)

simple_features = torch.tensor([[0, 0],
                                [1, 1],
                                [1, 1],
                                [0, 0],
                                [0, 0]]).to(torch.float)

labels = torch.tensor([0, 1, 1, 0, 0])
num_labels = torch.unique(labels).max().item() + 1

train_mask = torch.tensor([1, 0, 1, 0, 1])
# Unit tests for process.py


def testcompute_neighbourhood_feature_label_distribution():
    # Test add_vnodes
    label_neigh_dist, label_feat_mu, label_feat_std = compute_neighbourhood_feature_label_distribution(simple_g, simple_features, labels, num_labels)

    # assert something about the result like shapes and values
    # Probability distributions should sum to 1
    print("label_neigh_dist", label_neigh_dist)
    print("label_feat_mu", label_feat_mu)
    print("label_feat_std", label_feat_std)
    assert(torch.allclose(torch.sum(label_neigh_dist[0], dim=0), torch.ones(num_labels)))


def test_add_vnodes():
    num_new_nodes = 2
    new_edges = torch.tensor([[5, 0], [5, 2], [5, 4], [6, 1], [6, 3]])
    new_features = torch.tensor([[2, 2], [3, 3]]).to(torch.float)
    new_labels = torch.tensor([1, 0])

    new_g, new_features, new_labels = add_vnodes(simple_g, simple_features, labels, num_new_nodes, new_edges, new_features, new_labels)
    # print(f'New Graph: {new_g}')
    # print(f'New Features: {new_features}')
    # print(f'New Labels: {new_labels}')

    # assert something about the result like shapes and values
    assert(new_g.shape == (7, 7))
    assert(new_features.shape == (7, 2))
    assert(new_labels.shape == (7,))


def test_diff_from_average_label():
    # Get average label info
    label_neigh_dist, label_feat_mu, label_feat_std = compute_neighbourhood_feature_label_distribution(simple_g, simple_features, labels, num_labels)

    # assert something about the result like shapes and values
    # Probability distributions should sum to 1
    print("label_neigh_dist", label_neigh_dist)
    print("label_feat_mu", label_feat_mu)
    print("label_feat_std", label_feat_std)

    # Get differences of all nodes relative to average label
    node_neigh_delta, node_feat_delta = compute_differences(simple_g, simple_features, labels, num_labels, label_neigh_dist, label_feat_mu, label_feat_std)
    print("node_neigh_delta", node_neigh_delta)
    print("node_feat_delta", node_feat_delta)
    assert(node_neigh_delta.shape == (simple_g.shape[0], num_labels))
    assert(node_feat_delta.shape == (simple_g.shape[0], simple_features.shape[1]))


def test_adj_zeros_ones(adj: torch.Tensor):
    print("Graph adjacency:", adj[0:10, 0:10])
    # Verify that adjacency is 0s and 1s
    vals = [0.0, 1.0]
    assert(torch.is_tensor(adj))
    assert(torch.all(sum(adj!=i for i in vals).bool()))


def test_unmask_graph_identity():
    # A graph where the entire masked graph is masked for training
    # without any virtual nodes should be exactly the same as the
    # original graph.
    # g, features, labels, g_prime, features_prime, labels_prime, train_mask
    train_mask = torch.ones(simple_g.shape[0]).bool()
    g_restored, features_restored, labels_restored = unmask_graph(simple_g,
                                                                  simple_features,
                                                                  labels,
                                                                  simple_g,
                                                                  simple_features,
                                                                  labels,
                                                                  train_mask)
    
    # Shape should be the same
    assert(g_restored.shape == simple_g.shape)
    assert(features_restored.shape == simple_features.shape)
    assert(labels_restored.shape == labels.shape)

    # Values should be the same
    assert(torch.allclose(g_restored, simple_g))
    assert(torch.allclose(features_restored, simple_features))
    assert(torch.allclose(labels_restored, labels))
    

def test_unmask_graph_add_vnodes_entire_masking():
    # A graph where the entire masked graph is masked for training
    # with virtual nodes should be the same as the original graph
    # with the virtual nodes added.
    train_mask = torch.ones(simple_g.shape[0]).bool()
    num_new_nodes = 2
    new_edges = torch.tensor([[5, 0], [5, 2], [5, 4], [6, 1], [6, 3]])
    new_features = torch.tensor([[2, 2], [3, 3]]).to(torch.float)
    new_labels = torch.tensor([1, 0])
    g_prime, features_prime, labels_prime = add_vnodes(simple_g, simple_features, labels, num_new_nodes, new_edges, new_features, new_labels)
    
    g_restored, features_restored, labels_restored = unmask_graph(simple_g,
                                                                  simple_features,
                                                                  labels,
                                                                  g_prime,
                                                                  features_prime,
                                                                  labels_prime,
                                                                  train_mask)
    
    # Shape should go to (N + 2, N + 2)
    assert(g_restored.shape == (simple_g.shape[0] + num_new_nodes, simple_g.shape[0] + num_new_nodes))
    assert(features_restored.shape == (simple_features.shape[0] + num_new_nodes, simple_features.shape[1]))
    assert(labels_restored.shape == (labels.shape[0] + num_new_nodes,))

    # The new nodes, features and rows should just be appended to the original graph
    assert(torch.allclose(g_restored[0:simple_g.shape[0], 0:simple_g.shape[0]], simple_g))
    assert(torch.allclose(features_restored[0:simple_features.shape[0], :], simple_features))
    assert(torch.allclose(labels_restored[0:labels.shape[0]], labels))
    assert(torch.allclose(g_restored[simple_g.shape[0], :], torch.tensor([1, 0, 1, 0, 1, 0, 0])))
    assert(torch.allclose(g_restored[simple_g.shape[0] + 1, :], torch.tensor([0, 1, 0, 1, 0, 0, 0])))
    assert(torch.allclose(features_restored[simple_features.shape[0], :], torch.tensor([2, 2], dtype=torch.float)))
    assert(torch.allclose(features_restored[simple_features.shape[0] + 1, :], torch.tensor([3, 3], dtype=torch.float)))
    assert(torch.allclose(labels_restored[simple_features.shape[0]], torch.tensor([1])))
    assert(torch.allclose(labels_restored[simple_features.shape[0] + 1], torch.tensor([0])))


def test_unmask_graph_add_vnodes_partial_masking(directed: bool = True):
    train_mask = torch.tensor([True, False, True, False, True])
    num_new_nodes = 2
    new_edges = torch.tensor([[3, 0], [3, 1], [3, 2], [4, 1], [4, 2]])
    new_features = torch.tensor([[2, 2], [3, 3]]).to(torch.float)
    new_labels = torch.tensor([1, 0])

    visualise_graph(simple_g, 'initial_g.png')

    masked_simple_g = simple_g[train_mask,:][:, train_mask]
    masked_simple_features = simple_features[train_mask, :]
    masked_labels = labels[train_mask]
    visualise_graph(masked_simple_g, 'masked_g.png')
    
    # NOTE: You can change the behaviour of the test here
    if directed:
        g_prime, features_prime, labels_prime = add_vnodes(masked_simple_g,
                                                        masked_simple_features,
                                                        masked_labels,
                                                        num_new_nodes,
                                                        new_edges,
                                                        new_features,
                                                        new_labels)
        
        visualise_graph(g_prime, 'masked_g_with_vnodes.png')
    else:
        g_prime, features_prime, labels_prime = add_undirected_vnodes_to_graph(masked_simple_g,
                                                                               masked_simple_features,
                                                                               masked_labels,
                                                                               new_features,
                                                                               new_edges,
                                                                               new_labels,
                                                                               num_new_nodes)

        visualise_graph(g_prime, 'masked_g_with_undirected_vnodes.png')

    g_restored, features_restored, labels_restored = unmask_graph(simple_g,
                                                                  simple_features,
                                                                  labels,
                                                                  g_prime,
                                                                  features_prime,
                                                                  labels_prime,
                                                                  train_mask,
                                                                  directed=directed)
    
    restored_str = 'g_restored_directed.png' if directed else 'g_restored_undirected.png'
    visualise_graph(g_restored, restored_str)
    
    # Shape should go to (N + 2, N + 2)
    assert(g_restored.shape == (simple_g.shape[0] + num_new_nodes, simple_g.shape[0] + num_new_nodes))
    assert(features_restored.shape == (simple_features.shape[0] + num_new_nodes, simple_features.shape[1]))
    assert(labels_restored.shape == (labels.shape[0] + num_new_nodes,))

    # Here's the complicated part.
    assert(torch.allclose(g_restored[0:3, 0:3], simple_g[0:3, 0:3]))
    assert(torch.allclose(features_restored[0:3, :], simple_features[0:3, :]))
    assert(torch.allclose(labels_restored[0:3], labels[0:3]))

    assert(torch.allclose(g_restored[5, :], torch.tensor([1, 0, 1, 0, 1, 0, 0])))
    assert(torch.allclose(g_restored[6, :], torch.tensor([0, 0, 1, 0, 1, 0, 0])))
    assert(torch.allclose(features_restored[5, :], torch.tensor([2, 2], dtype=torch.float)))
    assert(torch.allclose(features_restored[6, :], torch.tensor([3, 3], dtype=torch.float)))
    assert(torch.allclose(labels_restored[5], torch.tensor([1])))
    assert(torch.allclose(labels_restored[6], torch.tensor([0])))

def test_compute_difference():
    # TODO
    assert False

if __name__ == '__main__':
    # testcompute_neighbourhood_feature_label_distribution()
    # test_unmask_graph_identity()
    # test_unmask_graph_add_vnodes_entire_masking()
    # test_unmask_graph_add_vnodes_partial_masking(directed=True)
    test_unmask_graph_add_vnodes_partial_masking(directed=False)
