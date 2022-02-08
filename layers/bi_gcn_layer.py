import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}

class biGCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, assign_dim, sigma=1, residual=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.assign_dim = assign_dim
        self.sigma = sigma
        
        if in_dim != out_dim:
            self.residual = False
        
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.apply_mod = NodeApplyModule(in_dim, out_dim)

        self.s1 = nn.Linear(in_dim,  out_dim, bias=True)
        self.s2 = nn.Linear(out_dim,  self.assign_dim, bias=True)
        self.metric = nn.Linear( self.assign_dim,  self.assign_dim, bias=True)
    
    def forward(self, g, feature):
        h_in = feature   # to be used for residual connection
        
        g.ndata['h'] = feature
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(feature))),dim=1) # soft assignment
        
        g.apply_edges(fn.u_sub_v('Sh', 'Sh', 'Sd')) # for cluster distance: (si - sj)
        Sd = g.edata['Sd'] # sum_edges, assign_dim
        Sd_h = self.metric(Sd) # sum_edges, assign_dim;  D = sqrt( (si - sj) W W^t (si - sj)^t )
        D = torch.sqrt(torch.sum(Sd_h*Sd_h,dim=1)).unsqueeze(1) # sum_edges, 1
        g.edata['GD'] = torch.exp( -D / ( 2*(self.sigma**2) ) ) # sum_edges, 1 # G = GaussianRBF(D)
        g.edata['sigma_GD'] = torch.sigmoid(g.edata['GD'])
        
        g.update_all(fn.u_mul_e('h', 'sigma_GD', 'm'), fn.mean('m', 'sum_sigma_GD_h'))
        g.update_all(fn.copy_e('sigma_GD', 'm'), fn.sum('m', 'sum_sigma_GD'))
        g.ndata['h'] = (g.ndata['sum_sigma_GD_h'] / (g.ndata['sum_sigma_GD'] + 1e-6))
        g.apply_nodes(func=self.apply_mod)
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.batchnorm_h(h) # batch normalization  
       
        if self.activation:
            h = self.activation(h)
        
        if self.residual:
            h = h_in + h # residual connection
            
        h = self.dropout(h)
        return h, g.ndata['Sh']
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.residual)