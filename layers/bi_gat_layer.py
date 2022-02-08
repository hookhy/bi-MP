import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class biGATHeadLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, batch_norm):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=True)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        a[torch.isinf(a)] = 1e+9 # clamping
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'] , 'sigma_GD': edges.data['sigma_GD']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        beta_sum = torch.sum(nodes.mailbox['sigma_GD'], dim=1).unsqueeze(-1)
        beta = nodes.mailbox['sigma_GD'] / (beta_sum + 1e-6) 
        alpha = F.dropout(alpha, self.dropout, training=self.training) 
        h = torch.sum(alpha * beta * nodes.mailbox['z'], dim=1)
        h[torch.isinf(h)] = 1e+9 # clamping
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h)
        z[torch.isinf(z)] = 1e+9 # clamping
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        h = F.elu(h)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h

    
class biGATLayer(nn.Module):
    """
        Param: [in_dim, out_dim, n_heads]
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, assign_dim, sigma=1, residual=True):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.residual = residual
        self.assign_dim = assign_dim
        self.sigma = sigma
        
        self.s1 = nn.Linear(in_dim,  out_dim, bias=True)
        self.s2 = nn.Linear(out_dim,  self.assign_dim, bias=True)
        self.metric = nn.Linear( self.assign_dim,  self.assign_dim, bias=True)
        
        if in_dim != (out_dim*num_heads):
            self.residual = False

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(biGATHeadLayer(in_dim, out_dim, dropout, batch_norm))
        self.merge = 'cat' 
        

    def forward(self, g, h):
        h_in = h # for residual connection
        
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(h))), dim=1) # soft assignment
        g.apply_edges(fn.u_sub_v('Sh', 'Sh', 'Sd')) # for cluster distance: (si - sj)
        Sd = g.edata['Sd'] # sum_edges, assign_dim
        Sd_h = F.relu(self.metric(Sd)) # sum_edges, assign_dim;  D = sqrt( (si - sj) W W^t (si - sj)^t )
        #dense_D = torch.matmul(Sd_h,Sd_h.transpose(0,1)) # sum_edges, sum_edges
        #D = torch.sqrt(torch.einsum("ii->i",dense_D).unsqueeze(1)) # sum_edges, 1
        D = torch.sqrt(torch.sum(Sd_h*Sd_h,dim=1)+ 1e-09).unsqueeze(1) # sum_edges, 1
        g.edata['GD'] = torch.exp( (-D / ( 2*(self.sigma**2) ))+ 1e-09) # sum_edges, 1 # G = GaussianRBF(D)
        g.edata['sigma_GD'] = torch.sigmoid(g.edata['GD'])
        
        head_outs = [attn_head(g, h) for attn_head in self.heads]

        if self.merge == 'cat':
            h = torch.cat(head_outs, dim=1)
        else:
            h = torch.mean(torch.stack(head_outs))

        if self.residual:
            h = h_in + h # residual connection
        
        return h, g.ndata['Sh']
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)

    