import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.bi_gcn_layer import biGCNLayer
from layers.mlp_readout_layer import MLPReadout

class biGCNNet_for_Eval(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
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
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                    self.batch_norm, self.residual) ])
        self.layers.append(biGCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.assign_dim, self.sigma, self.residual))
        for _ in range(n_layers-3):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                                    self.batch_norm, self.residual)) 
        self.layers.append(GCNLayer(hidden_dim, hidden_dim, F.relu, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)        


    def forward(self, g, h, e):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        # GCN
        cnt=0
        hs = [] # for feature analysis
        hs.append(h) # initial features
        for conv in self.layers:
            if cnt == 1: 
                h, self.s = conv(g, h)
            else:
                h = conv(g, h)
                
            hs.append(h)
            cnt+=1
        Hs = torch.stack(hs,dim=0) # L x NODE x dim

        # output
        h_out = self.MLP_layer(h)

        return h_out, Hs, self.s

    
    def sup_loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
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











