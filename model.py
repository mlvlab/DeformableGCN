import torch
import torch.nn as nn
import torch.nn.functional as F
from block import DeformableGConvBlock


class DeformableGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 n_classes,
                 n_blocks,
                 n_neighbor,
                 n_hops,
                 n_kernels,
                 phi_dim,
                 features,
                 activation,
                 dropout):
        super(DeformableGCN, self).__init__()
        self.g = g
        self.blocks = nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.feat_lin = nn.Linear(in_dim, hidden_dim)
        self.n_neighbor = n_neighbor
        self.n_hops = n_hops
        self.n_blocks = n_blocks
        self.activation = activation
                
        for _ in range(n_blocks-1):
            self.blocks.append(DeformableGConvBlock(hidden_dim, hidden_dim, in_dim, phi_dim, n_kernels, n_neighbor, n_hops, self.g, features, feat_drop=dropout, activation=activation))
        
        self.blocks.append(DeformableGConvBlock(hidden_dim, n_classes, in_dim, phi_dim, n_kernels, n_neighbor, n_hops, self.g, features, feat_drop=dropout))
        nn.init.xavier_normal_(self.feat_lin.weight, gain=1.414)


    def forward(self, features):
        h = features
        h = self.activation(self.feat_lin(h))
        
        l_sep_sum = 0
        l_focus_sum = 0
        for i in range(self.n_blocks):
            h, l_sep, l_focus = self.blocks[i](h)
            l_sep_sum += l_sep/self.n_blocks
            l_focus_sum += l_focus/self.n_blocks

        return F.log_softmax(h, 1), l_sep_sum, l_focus_sum