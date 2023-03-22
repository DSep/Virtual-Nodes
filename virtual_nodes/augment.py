import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

def compute_neighbourhood_feature_label_distribution(g, features, labels, num_labels):
    '''
    Compute the distribution of the features and labels of the neighbours of each node in the graph.
    Label distribution is a K x K matrix where each row represents the probability distribution over neighbourhood labels.
    Feature mu is a K x F matrix where each row represents the ___________________ over neighbourhood features.
    Assumes:
        - labels are integers from 0 to K-1, where K is the number of classes.
    '''
    num_nodes = g.shape[0]
    num_features = features.shape[1]
    unique_labels = torch.unique(labels)

    # K x K matrix where each row represents the probability distribution over neighbourhood labels
    label_neigh_dist = torch.zeros((num_labels, num_labels))
    # K x F matrix where each row represents the ___________________ over neighbourhood features
    label_feat_mu = torch.zeros((num_labels, num_features))

    print(f'Computing label dist and feature mu for {num_nodes} nodes')
    for node in range(num_nodes):
        neighbour_mask = g[node, :]
        neighbour_indices = torch.argwhere(neighbour_mask)  # K x 1
        neighbour_indices = neighbour_indices.squeeze(dim=-1)  # collapse to K

        neighbour_labels = labels[neighbour_indices]  # K?
        label_counts = torch.bincount(neighbour_labels, minlength=num_labels)
        label_neigh_dist[labels[node]] += label_counts

        # num_neighbours x F
        neighbour_features = features[neighbour_indices, :]
        # Get the sum feature vector of each neighbour # NOTE This is the naive appraoch to combining neighbour info
        # add a collapesed 1 x F
        label_feat_mu[labels[node]] += torch.sum(neighbour_features, dim=0)

    # Normalize each row (label) by the total occurances of a label – becomes probability distribution
    # https://pytorch.org/docs/stable/generated/torch.where.html
    totals_for_each = torch.sum(label_neigh_dist, dim=1)
    totals_for_each = torch.unsqueeze(totals_for_each, dim=1)
    # label_neigh_dist / totals_for_each
    label_neigh_dist = torch.div(label_neigh_dist, totals_for_each)
    # Compute mean feature vector for each label
    label_feat_mu = torch.div(label_feat_mu, totals_for_each)

    # Compute standard deviation feature vector over all neighbours of nodes with a given label l
    # NOTE: This is done separately to save memory and avoid storing all the feature vectors for ALL labels at once
    label_feat_std = torch.zeros((num_labels, num_features))
    print(f'Processing unique labels:', unique_labels)
    for l in unique_labels:
        print(f'Computing std for label {l}')
        # labels for each node
        label_mask = labels == l
        # list of node ids with label
        label_indices = torch.argwhere(label_mask)
        label_indices = label_indices.squeeze(dim=-1)
        # go through each of the nodes in label_indices neighbours and add their features to a tensor
        # reduced_adj = g[label_indices, :] # len(label_indices) x F
        neigh_features = torch.tensor([])
        for node_id in label_indices:
            neigh_mask = g[node_id, :]
            neigh_indices = torch.argwhere(neigh_mask)
            neigh_indices = neigh_indices.squeeze(dim=-1)
            neigh_features = torch.cat(
                (neigh_features, features[neigh_indices, :]), dim=0)

        # take standard devation over tensor (dim=1)
        label_feat_std[l] = torch.std(neigh_features, dim=0)
    return label_neigh_dist, label_feat_mu, label_feat_std


def compute_khop_neighbourhood_distributions(g,
                                             features, 
                                             labels, 
                                             k, 
                                             num_nodes, 
                                             num_features, 
                                             num_labels,
                                             agg='concat'):
    '''
    Computes k-hop neighbourhood distributions over neighbourhood
    features and labels for each node in the graph.

    Args:

    (+) g (torch.tensor): Assumes raw adjacency matrix `g` (N x N) as a torch.tensor.
    (+) features (torch.tensor): is a (N x F) torch.tensor where each row represents the features of a node.
    (+) labels (torch.tensor): is a (N x 1) torch.tensor where each row represents the label of a node.
    (+) agg (str): 'concat' or 'sum' to determine whether to concatenate or sum the k-hop neighbourhood distributions.

    Returns:

    (+) neigh_label_dist (torch.tensor): a (N x D) tensor representing a probability distribution
                                         over the k-hop neighbourhood labels for each node. If 
                                         `agg` is 'concat', then D = K * k, where K is the number
                                         of classes and k is the number of hops. If `agg` is 'sum',
                                         then D = K.
    
    (+) neigh_feat_mu (torch.tensor): a (N x D) tensor representing the mean of the k-hop neighbourhood
                                      features for each node. If `agg` is 'concat', then D = F * k,
                                      where F is the number of features and k is the number of hops. If
                                      `agg` is 'sum', then D = F.
    '''

    if agg == 'concat':
        neigh_label_dist = torch.zeros((num_nodes, num_labels * k))
        neigh_feat_mu = torch.zeros((num_nodes, num_features * k))
    elif agg == 'sum':
        neigh_label_dist = torch.zeros((num_nodes, num_labels))
        neigh_feat_mu = torch.zeros((num_nodes, num_features))

    g_k = g
    for i in range(k):
        # Compute the k-hop matrix
        if i > 0:
            g_k = g_k @ g
        
        for node_idx in range(num_nodes):
            # Count the number of paths between the current node and its neighbours
            # neighbour_paths = torch.where(g_k[node_idx, :])
            neighbour_paths = g_k

            # Create torch.tensor that aggregates features of shape (F)
            neigh_feats_agg = torch.zeros((num_features))
            neigh_label_agg = torch.zeros((num_labels))

            # Get the aggregated neighbour features and labels, scaled by the number of paths
            # connecting the current node and its neighbours.
            for neighbour_idx in range(num_nodes):

                neigh_feats_agg += neighbour_paths[node_idx][neighbour_idx] * features[neighbour_idx, :]
                neigh_label_agg[labels[neighbour_idx]] += neighbour_paths[node_idx][neighbour_idx]

            if agg == 'concat':
                # Concatenate the k-hop neighbourhood distributions for each node
                neigh_label_dist[node_idx, i*num_labels:(i+1)*num_labels] = neigh_label_agg / torch.sum(neigh_label_agg)
                neigh_feat_mu[node_idx, i*num_features:(i+1)*num_features] = neigh_feats_agg / torch.sum(neighbour_paths[node_idx])
                
                print('Warning: Using concat aggregation results in a standard deviation of 0.')

            elif agg == 'sum':
                # Sum the k-hop neighbourhood distributions for each node
                neigh_label_dist[node_idx, :] += neigh_label_agg / torch.sum(neigh_label_agg, dim=1)
                neigh_feat_mu[node_idx, :] += neigh_feats_agg / torch.sum(neigh_feats_agg)
        
    if agg == 'sum':
        neigh_label_dist /= k
        neigh_feat_mu /= k

    return neigh_label_dist, neigh_feat_mu


def _binarize_tensor(g):
    g_dense = g.to_dense()
    return torch.where(g_dense > 0, torch.ones_like(g_dense), torch.zeros_like(g_dense))


def _binarize_sparse_tensor(sparse_tensor):
    # Get the shape of the input sparse tensor
    shape = sparse_tensor.shape

    # Get the indices and values of the nonzero elements in the input sparse tensor
    indices = sparse_tensor.coalesce().indices()
    values = torch.ones(indices.shape[1])

    # Create a new sparse tensor with the same shape as the input sparse tensor, but with ones in the indices
    binary_tensor = torch.sparse.FloatTensor(indices, values, shape)

    return binary_tensor


def add_vnodes(
    masked_g,
    masked_features,
    masked_labels,
    num_new_nodes,
    new_edges,
    new_features,
    new_labels
):
    '''
    Creates a new graph (represented with an adjacency matrix, feature matrix and label vector) that includes
    the new virtual nodes whose features are given as an (N', F) torch.tensor where N' denotes the number of 
    virtual nodes to introduce, whose features are given in `new_features`.

    Args:

    (+) g (torch.tensor): Assumes raw adjacency matrix `g` (N x N) as a torch.tensor.
    (+) features (torch.tensor): is a (N x F) torch.tensor where each row represents the features of a node.
    ...
    (+) num_new_nodes (int): Self-explanatory.
    (+) new_edges (torch.tensor): (N' x 2) tensor where each row represents a directed edge from the source
                                  to the destination node.
    (+) new_features (torch.tensor): (N' x F) tensor representing the features of the new nodes added.
    (+) new_labels (torch.tensor): (N') tensor representing the labels of the new nodes added.

    Returns: 
    
    1. A new adjacency matrix as a torch.tensor (N + N', N + N') containing the additional virtual nodes
       where N' denotes the number of virtual nodes to introduce.
    2. A new feature matrix as a torch.tensor (N + N', F) where F denotes the feature dimension.
    3. A new label vector as a torch.tensor (N + N') 
    '''
    g_prime = F.pad(masked_g, (0, num_new_nodes), mode='constant', value=0) # (N x (N + N'))
    g_prime = torch.cat((g_prime, torch.zeros((num_new_nodes, g_prime.shape[1]), dtype=torch.long)), dim=0) # (N + N' x (N + N'))
    g_prime[new_edges[:, 0], new_edges[:, 1]] = 1 # Add the new edges to the graph
    
    # NOTE Implemented as function add_undirected_vnodes_to_graph below
    # if not directed:
    #     g_prime[new_edges[:, 1], new_edges[:, 0]] = 1 # Add the new edges to the graph

    features_prime = torch.cat((masked_features, new_features), dim=0) # (N + N' x F)
    labels_prime = torch.cat((masked_labels, new_labels), dim=0) # (N + N')

    return g_prime, features_prime, labels_prime


def add_undirected_vnodes_to_graph(g,
                                   features,
                                   labels, 
                                   new_features, 
                                   new_edges, 
                                   new_labels, 
                                   num_new_nodes):
    '''
    Args:

    (+) g (torch.tensor): Assumes raw adjacency matrix `g` (N x N) as a torch.tensor.
    (+) features (torch.tensor): is a (N x F) torch.tensor where each row represents the features of a node.
    ...
    (+) num_new_nodes (int): Self-explanatory.
    (+) new_edges (torch.tensor): (N' x 2) tensor where each row represents a directed edge from the source
                                    to the destination node.
    (+) new_features (torch.tensor): (N' x F) tensor representing the features of the new nodes added.
    (+) new_labels (torch.tensor): (N') tensor representing the labels of the new nodes added.

    Returns:

    1. A new adjacency matrix as a torch.tensor (N + N', N + N') containing the additional virtual nodes
         where N' denotes the number of virtual nodes to introduce.
    2. A new feature matrix as a torch.tensor (N + N', F) where F denotes the feature dimension.
    3. A new label vector as a torch.tensor (N + N').
    '''
    G = nx.from_numpy_array(g.numpy(), create_using=nx.DiGraph)

    for i, (features, label) in enumerate(zip(features, labels)):
        G.nodes[i]['features'] = features.numpy()
        G.nodes[i]['label'] = label.item()

    # Create the virtual nodes in the graph.
    for i in range(num_new_nodes):
        G.add_node(i + g.shape[0])
        G.nodes[i + g.shape[0]]['features'] = new_features[i].numpy()
        G.nodes[i + g.shape[0]]['label'] = new_labels[i].item()
    
    # Create bidirectional edges between the new virtual nodes and the target nodes.
    for i in range(new_edges.shape[0]):
        G.add_edge(new_edges[i, 0].item(), new_edges[i, 1].item())
        G.add_edge(new_edges[i, 1].item(), new_edges[i, 0].item())
    
    # Convert the graph back to a tensor.
    g_prime = torch.tensor(nx.to_numpy_array(G))
    features_prime = torch.tensor([G.nodes[i]['features'] for i in range(g_prime.shape[0])])
    labels_prime = torch.tensor([G.nodes[i]['label'] for i in range(g_prime.shape[0])])
    
    # Assert that the shape of `g_prime` is what we expect
    assert g_prime.shape[0] == g_prime.shape[1], "Augmented graph is not square."
    assert g_prime.shape[0] == features_prime.shape[0], "Augmented graph and features are not the same size."
    assert g_prime.shape[0] == labels_prime.shape[0], "Augmented graph and labels are not the same size."
    assert g_prime.shape[0] == g.shape[0] + num_new_nodes, "Augmented graph is not the correct size."

    assert torch.sum(g_prime) == torch.sum(g) + new_edges.shape[0] * 2, "Wrong number of edges"

    return g_prime, features_prime, labels_prime


def compute_differences(
    g,
    features,
    labels,
    # num_features,
    num_labels,
    # degree_vec,
    label_neigh_dist,
    label_feat_mu,
    label_feat_std,
):
    '''
    Args:

    (+) label_neigh_dist (torch.tensor): (K x K) tensor where K denotes the number of unique labels.
    (+) label_feat_mu (torch.tensor): (K x F) tensor where F denotes the feature dimension.
    (+) label_feat_std (torch.tensor): (K x F) tensor where F denotes the feature dimension.
    
    Compute the difference between a nodes neighbourhood and the neighbourhood of nodes with the same label.
    Takes a given graph with adjacency g, features and labels, and the label neighbourhood distribution and label feature distribution
    Returns the label distance and feature distance between each node and the average neighbourhood of nodes for its label.
    '''
    num_nodes = g.shape[0]
    num_features = features.shape[1]
    
    # Store all nodes' label distance and feature distance with mean
    node_neigh_delta = torch.zeros((num_nodes, num_labels)) # N x K
    node_feat_delta = torch.zeros((num_nodes, num_features)) # N x F
    print(f'Computing label dist and feature mu for {num_nodes} nodes')
    # Compute the neighbourhood label and feature distribution of each individual node
    for node in range(num_nodes):
        neighbour_mask = g[node, :]
        neighbour_indices = torch.argwhere(neighbour_mask)  # K x 1
        neighbour_indices = neighbour_indices.squeeze(dim=-1)  # collapse to K
        
        neighbour_labels = labels[neighbour_indices]  # ??
        label_counts = torch.bincount(neighbour_labels, minlength=num_labels)
        label_dist = label_counts / torch.sum(label_counts) # equivalent to label_counts / torch.sum(neighbour_mask) 

        neighbour_features = features[neighbour_indices, :] # num_neighbours x F
        neighbour_features = torch.mean(neighbour_features, dim=0) # F
        
        # Compare that with the average neighbourhood label and feature distribution of nodes with the same label
        average_features = label_feat_mu[labels[node]]
        average_label = label_neigh_dist[labels[node]]
        
        # Compute the difference between all of these distributions directly (not via KL divergence)
        node_neigh_delta[node] = average_label - label_dist
        node_feat_delta[node] = average_features - neighbour_features

    return node_neigh_delta, node_feat_delta


def convert_to_torch_distributions(label_neigh_dist, label_feat_mu, label_feat_std):
    # Construct label distribution objects from each of label_neigh_dist
    label_neigh_dist_objs = []
    for l in label_neigh_dist:
        label_neigh_dist_objs.append(torch.distributions.categorical.Categorical(l))

    # Construct feature distribution objects from each pair from label_feat_mu and label_feat_std
    feat_neigh_dist_objs = []  # list of torch distribution objects
    for mu, std in zip(label_feat_mu, label_feat_std):
        feat_neigh_dist_objs.append(torch.distributions.multivariate_normal.MultivariateNormal(mu, std))
    return label_neigh_dist_objs, feat_neigh_dist_objs

def compute_divergences(
    g,
    features,
    labels,
    num_features,
    num_labels,
    degree_vec,
    label_neigh_dist_objs,
    feat_neigh_dist_objs,
):
    '''
    Given a graph G (as a unweighted adjacency matrix) and a node classification task, for each class k of node in the graph
    generate a distribution over the neighbourhood labels of neighbouring nodes and a distribution over the features
    of each of the classes.
    Then for each node, compute its divergence with the distributions over the neighbourhood labels and features.
    '''
    num_nodes = g.shape[0]

    # Introduce the virtual nodes that connect all nodes with similar distributions of neighbour labels

    # NAIVE 1: Create a list of divergences, between each node's neighbour dists and the average label neighbour dist calculated
    degree_vec = torch.tensor(degree_vec)
    neigh_label_divergences = torch.zeros(num_nodes)
    neigh_feat_divergences = torch.zeros(num_nodes)
    for i, l in enumerate(labels):
        neighbour_mask = g[i, :]
        neighbour_indices = torch.argwhere(neighbour_mask)  # K x 1
        neighbour_indices = neighbour_indices.squeeze(dim=-1)  # collapse to K
        neighbour_labels = labels[neighbour_indices]  # K
        label_counts = torch.bincount(neighbour_labels, minlength=num_labels)

        # Label divergence
        neigh_label_divergences[i] = torch.distributions.kl.kl_divergence(
                label_neigh_dist_objs[l],
                torch.distributions.categorical.Categorical(label_counts / degree_vec[i]))

        # Feature divergence
        neighbour_features = features[neighbour_indices, :]
        neigh_feat_divergences[i] = torch.distributions.kl.kl_divergence(
                feat_neigh_dist_objs[l],
                torch.distributions.multivariate_normal.MultivariateNormal(
                    torch.mean(neighbour_features, dim=0),
                    torch.std(neighbour_features, dim=0)))

    return neigh_label_divergences, neigh_feat_divergences
    # # NAIVE 1: sub-approach 1: Add virtual nodes to nodes with divergence > epsilon
    # # eps = ...

    # # NAIVE 1: sub-approach 2: Add virtual nodes to p proportion of nodes with highest divergence
    # p = 0.1
    # num_vnodes = int(p * num_nodes)

    # kl_sorted_label_indices = torch.argsort(neigh_label_divergences)[:num_vnodes]
    # kl_sorted_feat_indices = torch.argsort(neigh_feat_divergences)[:num_vnodes]
    # return kl_sorted_label_indices, kl_sorted_feat_indices


# TODO
def mask_and_op(g, features, labels, train_mask):
    '''
    Masks the graph and performs the node-level operation op on the masked graph,
    then converts back to the original graph size and returns.

    Args:

    (+) g (torch.tensor): Adjacency matrix of the full graph (unmasked) (N, N).
    (+) features (torch.tensor): Node features of the full graph (unmasked) (N, F).
    (+) labels (torch.tensor): Node labels of the full graph (unmasked) (N,).
    (+) train_mask (torch.tensor): Mask indicating which nodes we know the labels of
        (N,).
    '''
    pass

def update_masks(train_mask, val_mask, test_mask, num_new_nodes):
    '''
    Updates the masks to account for the new nodes added to the graph.

    Args:

    (+) train_mask (torch.tensor): Mask indicating which nodes we know the labels of
        (N,).
    (+) val_mask (torch.tensor): Mask indicating which nodes we know the labels of
        (N,).
    (+) test_mask (torch.tensor): Mask indicating which nodes we know the labels of
        (N,).
    (+) num_new_nodes (int): Number of new nodes added to the graph.

    Returns:
    Updated masks (train_mask, val_mask, test_mask), size (N + num_new_nodes,).
    '''
    # Extend the length of train_mask by num_new_nodes and set the last num_new_nodes to 1
    train_mask = torch.cat((train_mask, torch.zeros(num_new_nodes)), dim=0).bool()

    # Extend the length of val_mask by num_new_nodes and set the last num_new_nodes to 0
    val_mask = torch.cat((val_mask, torch.zeros(num_new_nodes)), dim=0).bool()

    # Extend the length of test_mask by num_new_nodes and set the last num_new_nodes to 0
    test_mask = torch.cat((test_mask, torch.zeros(num_new_nodes)), dim=0).bool()

    return train_mask, val_mask, test_mask


def unmask_graph(g, features, labels, g_prime, features_prime, labels_prime, train_mask, directed=True):
    '''
    Before computing which virtual nodes to add, we mask g, features and labels using
    train_mask. Then, we call add_virtual_nodes on the masked g, features and labels.
    This returns g_prime, features_prime and labels_prime, which are the augmented, 
    masked graph. This function unmasks g_prime, features_prime and labels_prime to
    contain the masked rows of g, features and labels and the newly added virtual
    nodes of g_prime, features_prime and labels_prime.
    '''
    masked_g = g[train_mask, :][:, train_mask]
    num_new_nodes = g_prime.shape[0] - masked_g.shape[0]
    
    g_restored = torch.zeros(g.shape[0] + num_new_nodes, g.shape[0] + num_new_nodes, dtype=torch.long)
    features_restored = torch.zeros(g.shape[0] + num_new_nodes, features.shape[1], dtype=torch.float)
    labels_restored = torch.zeros(g.shape[0] + num_new_nodes, dtype=torch.long)

    g_restored[:g.shape[0], :g.shape[0]] = g
    features_restored[:g.shape[0], :] = features
    labels_restored[:g.shape[0]] = labels

    # Compute the absoluate differences in positions for all nodes in g and masked_g. 
    differences = torch.zeros((g.shape[0])).long()
    acc = 0
    print(train_mask)
    for i in range(g.shape[0]):
        differences[i] = acc
        if train_mask[i] == 0:
            acc += 1
    masked_diffs = differences[train_mask]

    # Adjust vnodes connections according to the absolute differences
    # for i in range(g.shape[0], g_restored.shape[0]):
    # for i in range(g_prime.shape[0] - num_new_nodes, g_prime.shape[0]):
    for i in range(num_new_nodes):
        target_idxs = torch.where(g_prime[masked_g.shape[0] + i, :] == 1)[0] # (num_neighbours,)
        updated_idxs = target_idxs + masked_diffs[target_idxs] # (num_neighbours,)
        restored_vnode_idx = g.shape[0] + i
        g_restored[restored_vnode_idx, updated_idxs] = 1

        if not directed:
            g_restored[updated_idxs, restored_vnode_idx] = 1

    features_restored[g.shape[0]:, :] = features_prime[g_prime.shape[0] - num_new_nodes:, :]
    labels_restored[g.shape[0]:] = labels_prime[g_prime.shape[0] - num_new_nodes:]

    return g_restored, features_restored, labels_restored


def create_vnodes_naive_strategy_1(masked_g, masked_features, masked_labels, num_nodes_masked, num_labels, p, agg):
    """
    For all observable nodes, computes the neighbourhood mean and std deviations of the
    neighbourhood label distributions (NDs) for each class. Then for each node with an 
    observable label, computes the difference between the NDs of the node's label and 
    the ND of the node. Then adds virtual nodes to the graph for the p% of nodes with
    the highest difference. 

    NOTE: Naive strategy 1 is the choice of:
        - using neighbourhood label distributions
        - using absolute differences in bincounts
    """

    # Compute the distribution of the features and labels of the neighbours of each node in the graph.
    label_neigh_dist, label_feat_mu, label_feat_std = compute_neighbourhood_feature_label_distribution(
        masked_g, masked_features, masked_labels, num_labels)
    
    # label_neigh_dist_objs, feat_neigh_dist_objs = convert_to_torch_distributions(label_neigh_dist, label_feat_mu, label_feat_std)
    node_neigh_delta, node_feat_delta = compute_differences(masked_g,
                                                            masked_features,
                                                            masked_labels,
                                                            num_labels,
                                                            label_neigh_dist,
                                                            label_feat_mu,
                                                            label_feat_std)
    # Determine new set of nodes and edges
    num_new_nodes = int(p * num_nodes_masked)
    
    if agg == 'sum':
        raise NotImplementedError("Sum aggregation is not currently implemented.")
    
    # Get the indices of num_new_nodes lowest elements of node_neigh_delta    
    target_node_indices_flat = torch.argsort(node_neigh_delta.view(-1))[:num_new_nodes]
    
    # Given the target_node_indices, get the corresponding row and column indices
    target_row_indices = torch.div(target_node_indices_flat, node_neigh_delta.shape[1], rounding_mode='floor')
    target_col_indices = target_node_indices_flat % node_neigh_delta.shape[1]
    
    # Turn target_row_indices and target_col_indices into a num_new_nodes x 2 tensor
    target_indices = torch.stack((target_row_indices, target_col_indices), dim=1) # elements are node_index, neighbour_label_index
    
    # Apply a single correction to each node  # TODO apply multiple operations per problem node
    # Create a new virtual node that connects to each node. Select a label based on the most different neighbour label distribution element index.
    new_nodes = torch.tensor(np.arange(num_nodes_masked, num_nodes_masked + len(target_indices)))
    new_edges = torch.cat((new_nodes.unsqueeze(1), target_indices[:, 0].unsqueeze(1)), dim=1)
    
    new_labels = target_indices[:, 1]
    new_features = torch.tensor(label_feat_mu[new_labels, :])
    return new_nodes.shape[0], new_edges, new_features, new_labels


def augment_graph(
    g,
    features,
    labels, 
    train_mask,
    val_mask,
    test_mask,
    num_features, 
    num_labels,
    p,
    agg,
    undirected=True,
):
    '''
    Creates virtual nodes according to a chosen strategy.
    Adds them to the graph, as well as the masks.

    Args:

    (+) g (torch.tensor): Adjacency matrix of the full graph (unmasked) (N, N).
    (+) features (torch.tensor): Node features of the full graph (unmasked) (N, F).
    (+) labels (torch.tensor): Node labels of the full graph (unmasked) (N,).
    (+) train_mask (torch.tensor): Mask indicating which nodes we know the labels of (N,).
    (+) p (float): Proportion of nodes to add virtual nodes to.
    (+) undirected (bool): Whether the edges connecting virtual to target nodes should be
                           directed or not.

    Returns:

    (+) g_prime (torch.tensor): Adjacency matrix of the augmented graph (unmasked) (N + N', N + N').
    (+) features_prime (torch.tensor): Node features of the augmented graph (unmasked) (N + N', F).
    (+) labels_prime (torch.tensor): Node labels of the augmented graph (unmasked) (N + N',).
    '''
    # g = _binarize_tensor(g) # Transform (sparse) weighted adj to (dense) unweighted adj
    
    # Mask the graph and features to only include the nodes with known labels.
    masked_g = g[train_mask, :][:, train_mask] # N_train x N_train
    assert masked_g.shape[0] == masked_g.shape[1], "Masked graph is not square."
    assert masked_g.shape[0] == torch.sum(train_mask), "Masked graph is not the same size as the train mask."

    masked_features = features[train_mask, :]
    masked_labels = labels[train_mask]
    
    num_new_nodes, new_edges, new_features, new_labels = create_vnodes_naive_strategy_1(masked_g, masked_features, masked_labels, masked_g.shape[0], num_labels, p, agg)

    if not undirected:
        g_prime, features_prime, labels_prime = add_vnodes(masked_g, masked_features, masked_labels, num_new_nodes, new_edges, new_features, new_labels)
    else:
        g_prime, features_prime, labels_prime = add_undirected_vnodes_to_graph(masked_g, 
                                                                               masked_features,
                                                                               masked_labels,
                                                                               new_features,
                                                                               new_edges,
                                                                               new_labels,
                                                                               num_new_nodes)

    print(f'Num New Nodes: {num_new_nodes}.')
    print(f'Actual Num New Nodes: {g_prime.shape[0] - masked_g.shape[0]}')
    print(f'G Prime Shape: {g_prime.shape}')
    print(f'G Shape: {g.shape}')

    # Unmask the graph
    g, features, labels = unmask_graph(g, features, labels, g_prime, features_prime, labels_prime, train_mask)

    print(f'G Augmented: {g.shape}')

    # Update masks to be the same size as the new graph
    train_mask, val_mask, test_mask = update_masks(train_mask, val_mask, test_mask, num_new_nodes)
    return g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, num_new_nodes

def naive_strategy_2():
    pass
