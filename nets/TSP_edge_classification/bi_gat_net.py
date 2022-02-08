import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import CustomGATLayer as GATLayer
from layers.bi_gat_layer import biGATLayer
from layers.mlp_readout_layer import MLPReadout

class biGATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        self.num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.edge_feat = net_params['edge_feat']
        self.sigma = net_params['sigma']
        self.pos_enc = net_params['pos_enc']
        self.sg_flag = True
        self.assign_dim = net_params['assign_dim']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * self.num_heads)
        
        if self.edge_feat:
            self.embedding_e = nn.Linear(in_dim_edge, hidden_dim * self.num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GATLayer(hidden_dim * self.num_heads, hidden_dim, self.num_heads,
                                              dropout, self.batch_norm, self.residual) ])
        self.layers.append(biGATLayer(hidden_dim * self.num_heads, hidden_dim, self.num_heads, 
                                                 dropout, self.batch_norm, self.assign_dim, self.sigma, self.residual))
        for _ in range(n_layers-3):
            self.layers.append(GATLayer(hidden_dim * self.num_heads, hidden_dim, self.num_heads,
                                              dropout, self.batch_norm, self.residual)) 
        self.layers.append(GATLayer(hidden_dim * self.num_heads, out_dim, 1,
                                    dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h, e):
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        
        #if not self.edge_feat:
            #e = torch.ones_like(e).to(self.device)
        #e = self.embedding_e(e.float())

        cnt = 0
        for conv in self.layers:
            if cnt == 1:
                h, self.s = conv(g, h)
            else:
                h = conv(g, h)
            h[torch.isinf(h)] = 1e+9 # clamping
            cnt+=1
        
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            e[torch.isinf(e)] = 1e+9 # clamping
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e'], self.s
    
    def sup_loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    
    def unsup_loss(self, g, soft_assign, mode):
        
        if mode == 'mincut':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            d = torch.sparse_coo_tensor(torch.tensor([range(adj.size()[0]),range(adj.size()[0])]), 
                                        torch.sparse.sum(adj,dim=1).to_dense())
            out_adj = torch.mm(soft_assign.transpose(0,1),torch.sparse.mm(adj,soft_assign))
            out_d = torch.mm(soft_assign.transpose(0,1),torch.sparse.mm(d,soft_assign))

            mincut_num = torch.einsum('ii->', out_adj)
            mincut_den = torch.einsum('ii->', out_d)
            mincut_loss = -(mincut_num / mincut_den + 1e-09)

            ss = torch.matmul(soft_assign.transpose(0, 1), soft_assign)
            i_s = torch.eye(soft_assign.shape[1]).type_as(ss)
            ortho_loss = torch.norm(
            ss / torch.norm(ss + 1e-09, dim=(-0, -1), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-0, -1)) 
        
            return mincut_loss + ortho_loss
        elif mode == 'diffpool':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            
            ent_loss = torch.distributions.Categorical(probs=soft_assign).entropy().mean(-1)
            linkpred_loss = torch.add( -soft_assign.matmul(soft_assign.transpose(0,1)),adj).norm(dim=(0,1)) / (adj.size(0)*adj.size(1))
            
            return ent_loss + linkpred_loss