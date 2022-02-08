import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.bi_gcn_layer import biGCNLayer
from layers.mlp_readout_layer import MLPReadout

class biGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.sigma = net_params['sigma']
        self.sg_flag = True
        self.assign_dim = net_params['assign_dim']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                    self.batch_norm, self.residual) ])
        self.layers.append(biGCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.assign_dim, self.sigma, self.residual))
        for _ in range(n_layers-3):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                    self.batch_norm, self.residual)) 
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)        

    def forward(self, g, h, e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        cnt=0
        for conv in self.layers:
            if cnt == 1: 
                h, self.s = conv(g, h)
            else:
                h = conv(g, h)
            cnt+=1
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg), self.s
    
    def sup_loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
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
            mincut_loss = -(mincut_num / mincut_den)

            ss = torch.matmul(soft_assign.transpose(0, 1), soft_assign)
            i_s = torch.eye(soft_assign.shape[1]).type_as(ss)
            ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-0, -1), keepdim=True) -
            i_s / torch.norm(i_s), dim=(-0, -1)) 
        
            return mincut_loss + ortho_loss
        elif mode == 'diffpool':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            
            ent_loss = torch.distributions.Categorical(probs=soft_assign).entropy().mean(-1)
            linkpred_loss = torch.add( -soft_assign.matmul(soft_assign.transpose(0,1)),adj).norm(dim=(0,1)) / (adj.size(0)*adj.size(1))
            
            return ent_loss + linkpred_loss    
    