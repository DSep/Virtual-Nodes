import torch.nn as nn
import torch
from Heterophily_and_oversmoothing.model import GCNConvolution, GAT
#-------------------------------------------------------------------------------------------HETEROGCN------------------------------------------------------------------------------------

class HeteroGCN(nn.Module):
    '''
    Novel Contribution!
    
    TODO: Complete this.
    '''
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, in_feat, g, features, target_nodes, g_prime):
        # Use GAT to generate node embeddings for the nodes that we want to predict. These
        # are given by `target_nodes` and the original raw adjacency and feature matrices
        # are given by `g` and `features`.

        self.target_nodes = target_nodes
        self.g_prime = g_prime # New graph with the directed edges to the target nodes.

        nemb = nfeat # For simplicity of understanding – the output of GAT should be interpeted
                     # as the node embeddings for the target nodes.

        self.emb_layer = nn.Linear(in_feat, nfeat)
        self.emb_gnn = GAT(nfeat=nfeat, nhid=nhid, nlayers=1, nclass=nemb, dropout=0.6, alpha=0.2, nheads=1)

        num_vnodes = target_nodes.shape[0]

        super(HeteroGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvolution(nfeat, nhid, first_layer=True, num_vnodes=num_vnodes))
        for _ in range(nlayers-2):
            self.convs.append(GCNConvolution(nhid, nhid))
        self.convs.append(GCNConvolution(nhid, nclass))
        self.dropout = dropout

    def forward(self, x, adj):
        emb_x = self.emb_layer(x) # BOW -> EMB_DIM
        emb_x = self.emb_gnn(emb_x, adj) # EMB_DIM -> EMB_DIM
        vnode_x = emb_x[self.target_nodes, :]

        x = torch.cat((emb_x, vnode_x), dim=0)
        
        # for gc in self.convs[:-1]:
        #     vnode_x = F.relu(gc(x, self.g_prime))
        #     vnode_x = F.dropout(vnode_x, self.dropout, training=self.training)

        vnode_x = self.convs[-1](vnode_x, adj)

        # TODO: This method is incomplete.
        return None
