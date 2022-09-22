import torch as th
from torch import nn 
from torch.nn import functional as F


from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

import dgl


class DeformableGConvBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 raw_dim,
                 phi_dim,
                 n_kernels,
                 n_neighbor,
                 n_hops,
                 g,
                 features,                 
                 feat_drop=0.5,
                 activation=None
                 ):
        super(DeformableGConvBlock, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_dim)  
        self._out_dim = out_dim
        self.n_neighbor = n_neighbor
        self.n_kernels = n_kernels
        self.hops = n_hops
        self.phi_dim = phi_dim
        

        self.feat_drop = nn.Dropout(feat_drop)
        self.n_hops = n_hops+2      # graph_ori + 0hop + # hops
        
        self.deformablegconv = nn.ModuleList([])
        for _ in range(self.n_hops):
            self.deformablegconv.append(DeformableGConv(raw_dim, self._in_src_feats, out_dim, n_kernels, phi_dim, feat_drop))
        
        self.z = nn.Parameter(th.ones(out_dim, 1))

        self.gen_egraph(g, features)

        self.activation = activation

    def gen_egraph(self, graph, features):
        e_graph_list = []
        e = features
        e = th.tensor(e).to(features.device)
        e_tensor = e.unsqueeze(0)
        for k in range(self.hops + 1):
            if k>0:
                temp_graph = graph.remove_self_loop()
                temp_graph.ndata['e'] = e
                msg_fn = fn.copy_src('e', 'm')
                temp_graph.update_all(msg_fn, fn.mean('m', 'e'))
                e = temp_graph.dstdata['e']
                e_tensor = th.cat((e_tensor, e.unsqueeze(0)),0)
                
            e_graph = dgl.knn_graph(e, self.n_neighbor)
            e_graph = e_graph.to(features.device)
            e_graph.ndata['e'] = e
            e_graph_list.append(e_graph)
        e_graph = graph
        e = features
        e_graph = e_graph.to(features.device)
        e_graph_list.append(e_graph)
        e_tensor = th.cat([e_tensor, e.unsqueeze(0)], 0)

        self.e_graph_list = e_graph_list
        self.e_tensor = e_tensor
        
    def forward(self, h):
        h = self.feat_drop(h)

        l_sep_list, l_focus_list = [], []
        for l in range(self.n_hops):
            h_l, l_sep, l_focus = self.deformablegconv[l](h, self.e_tensor[l], self.e_graph_list[l])
            if l == 0:
                tilde_h = h_l.unsqueeze(0)
            else:
                tilde_h = th.cat([tilde_h, h_l.unsqueeze(0)], dim=0)
            
            l_sep_list.append(l_sep)
            l_focus_list.append(l_focus)

        l_sep, l_focus = sum(l_sep_list)/self.n_hops, sum(l_focus_list)/self.n_hops

        z = self.z.unsqueeze(0).repeat(tilde_h.size(0), 1, 1)
        score = th.matmul(tilde_h, z)
        score = F.softmax(score, 0)
        tilde_h = (score * tilde_h).sum(0)


        if self.activation:
            tilde_h = h + tilde_h
            tilde_h = self.activation(tilde_h)


        return tilde_h, l_sep, l_focus


class DeformableGConv(nn.Module):
    def __init__(self,
                 raw_dim,
                 hidden_dim,
                 out_dim,
                 n_kernels,
                 phi_dim,
                 feat_drop
                 ):
        super(DeformableGConv, self).__init__()
        self.n_kernels = n_kernels
        self.phi_dim = phi_dim
        self.hidden_dim = hidden_dim
        self.feat_drop = feat_drop
        self.W_phi = nn.Parameter(th.empty(raw_dim, phi_dim-1))

        self.delta_mlp = nn.Sequential(
            nn.Linear(raw_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, phi_dim*self.n_kernels)
        )

        if self.n_kernels <= phi_dim:
            tilde_phi = (th.eye(phi_dim)[-self.n_kernels:]).float() 
        else:
            tilde_phi = (th.eye(phi_dim)).float() 
            tilde_phi_added = th.rand(self.n_kernels-phi_dim, phi_dim)
            tilde_phi = th.cat((tilde_phi, tilde_phi_added), 0)
        self.tilde_phi = nn.Parameter(tilde_phi)  # (k, phi_dim)
        self.W = nn.Parameter(th.empty(self.hidden_dim, out_dim*self.n_kernels))
        self.bias = nn.Parameter(th.zeros(out_dim))

        self.reset_parameters()
        self.flag = 0

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W_phi, gain=gain)
        nn.init.xavier_normal_(self.delta_mlp[0].weight)
        nn.init.xavier_normal_(self.delta_mlp[2].weight)
        nn.init.xavier_normal_(self.W, gain=gain)

        
    def forward(self, h_l, e_l, e_graph):
        h_l = th.matmul(h_l, self.W)

        e_graph.ndata['h'] = h_l.view(h_l.size(0), self.n_kernels, -1) #(n, k, d)
        phi_k = th.matmul(e_l, self.W_phi) 
        tilde_phi_k = (self.tilde_phi/self.tilde_phi.norm(2, 1).unsqueeze(1)).unsqueeze(0) #(1, k, d_phi)

        delta_k = self.delta_mlp(e_l).view(e_l.size(0), self.n_kernels, -1) #(n, k, d_phi)
        delta_ik = tilde_phi_k + delta_k  #(n, k, d_phi)
        temp = th.zeros_like(delta_ik).to(delta_ik.device)

        e_graph.srcdata.update({'e1':phi_k, 'temp':temp})
        e_graph.dstdata.update({'e2':phi_k, 'delta_ik':delta_ik})
        e_graph.apply_edges(fn.u_add_v('temp', 'delta_ik', 'delta_ik'))
        e_graph.apply_edges(fn.u_sub_v('e1', 'e2', 'dist'))
        dist = e_graph.edata['dist']

        mask =  (dist == th.zeros(1, dist.size(1)).to(dist.device)).sum(1) == (dist.size(1))
        last_dim = mask.unsqueeze(1).float()
        dist = th.cat((dist, last_dim), 1)
        dist = dist/(dist.norm(2, 1).unsqueeze(1).clamp(min=1e-8))

        delta_ik = e_graph.edata['delta_ik']        
        a_ijk = (dist.unsqueeze(1)* delta_ik).sum(2, keepdim=True) #(e, k, 1)

        e_graph.edata['att'] = edge_softmax(e_graph, a_ijk)
        
        e_graph.update_all(fn.u_mul_e('h', 'att', 'm'),
                        fn.sum('m', 'h'))

        h_l = e_graph.dstdata['h'].sum(1) 
        h_l = h_l + self.bias.unsqueeze(0)

        h_l = h_l/(h_l.norm(2,1).unsqueeze(1).clamp(min=1e-8))

        l_sep = 0
        for kern in range(self.n_kernels):
            kern_tmp = tilde_phi_k[:,kern,:].unsqueeze(1)
            l_sep += (kern_tmp - tilde_phi_k).norm(2,2).pow(2).mean()
        l_focus = delta_k.norm(2,2).pow(2).mean()

        return h_l, l_sep, l_focus
