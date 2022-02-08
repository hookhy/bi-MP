import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

class biGraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, assign_dim, sigma=1, 
                 residual=False, bias=True):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.aggregator_type = aggregator_type
        self.residual = residual
        self.assign_dim = assign_dim
        self.sigma = sigma
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
                                   bias=True)
        
        if self.aggregator_type == "maxpool":
            self.maxlinear = nn.Linear(in_feats, out_feats, bias=True)
        
        self.s1 = nn.Linear(in_feats,  out_feats, bias=True)
        self.s2 = nn.Linear(out_feats,  self.assign_dim, bias=True)
        self.metric = nn.Linear( self.assign_dim,  self.assign_dim, bias=True)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
    
    def message_func(self, edges):
        return {'z': edges.src['h'], 'sigma_GD': edges.data['GD']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['sigma_GD'], dim=1)
        if self.aggregator_type == "maxpool":
            c = torch.max(alpha * nodes.mailbox['z'], dim=1)[0]
        else:
            c = torch.mean(alpha * nodes.mailbox['z'], dim=1)
        return {'c': c}
    
    def forward(self, g, h):
        h_in = h              # for residual connection
        
        h = self.dropout(h)

        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(h))),dim=1) # soft assignment
        
        g.apply_edges(fn.u_sub_v('Sh', 'Sh', 'Sd')) # for cluster distance: (si - sj)
        Sd = g.edata['Sd'] # sum_edges, assign_dim
        Sd_h = self.metric(Sd) # sum_edges, assign_dim;  D = sqrt( (si - sj) W W^t (si - sj)^t )
        D = torch.sqrt(torch.sum(Sd_h*Sd_h,dim=1)+ 1e-09).unsqueeze(1) # sum_edges, 1
        g.edata['GD'] = torch.exp( (-D / ( 2*(self.sigma**2) ))+ 1e-09) # sum_edges, 1 # G = GaussianRBF(D)
        g.edata['sigma_GD'] = torch.sigmoid(g.edata['GD'])
        
        if self.aggregator_type == "maxpool":
            g.ndata['h'] = F.relu(self.maxlinear(h))
        else:
            g.ndata['h'] = h
        #g.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'c'), self.nodeapply)
        g.update_all(self.message_func, self.reduce_func, self.nodeapply)
        h = g.ndata['h']

        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h, g.ndata['Sh']
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.residual)

    
    
"""
    Aggregators for GraphSage
"""
class Aggregator(nn.Module):
    """
    Base Aggregator class. 
    """

    def __init__(self):
        super().__init__()

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}

    def aggre(self, neighbour):
        # N x F
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Mean Aggregator for graphsage
    """

    def __init__(self):
        super().__init__()

    def aggre(self, neighbour):
        mean_neighbour = torch.mean(neighbour, dim=1)
        return mean_neighbour


class MaxPoolAggregator(Aggregator):
    """
    Maxpooling aggregator for graphsage
    """

    def __init__(self, in_feats, out_feats, activation, bias):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.activation = activation

    def aggre(self, neighbour):
        neighbour = self.linear(neighbour)
        if self.activation:
            neighbour = self.activation(neighbour)
        maxpool_neighbour = torch.max(neighbour, dim=1)[0]
        return maxpool_neighbour


class LSTMAggregator(Aggregator):
    """
    LSTM aggregator for graphsage
    """

    def __init__(self, in_feats, hidden_feats):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
        self.hidden_dim = hidden_feats
        self.hidden = self.init_hidden()

        nn.init.xavier_uniform_(self.lstm.weight,
                                gain=nn.init.calculate_gain('relu'))

    def init_hidden(self):
        """
        Defaulted to initialite all zero
        """
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def aggre(self, neighbours):
        """
        aggregation function
        """
        # N X F
        rand_order = torch.randperm(neighbours.size()[1])
        neighbours = neighbours[:, rand_order, :]

        (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0], neighbours.size()[1], -1))
        return lstm_out[:, -1, :]

    def forward(self, node):
        neighbour = node.mailbox['m']
        c = self.aggre(neighbour)
        return {"c": c}
    
    
class NodeApply(nn.Module):
    """
    Works -> the node_apply function in DGL paradigm
    """

    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_feats * 2, out_feats, bias)
        self.activation = activation

    def concat(self, h, aggre_result):
        bundle = torch.cat((h, aggre_result), 1)
        bundle = self.linear(bundle)
        return bundle

    def forward(self, node):
        h = node.data['h']
        c = node.data['c']
        bundle = self.concat(h, c)
        bundle = F.normalize(bundle, p=2, dim=1)
        if self.activation:
            bundle = self.activation(bundle)
        return {"h": bundle}
    
    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################



class GraphSageLayerEdgeFeat(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False, 
                 bias=True, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.activation = activation
        
        self.A = nn.Linear(in_feats, out_feats, bias=bias)
        self.B = nn.Linear(in_feats, out_feats, bias=bias)
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout, bias=bias)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
            
    def message_func(self, edges):
        Ah_j = edges.src['Ah']    
        e_ij = edges.src['Bh'] + edges.dst['Bh'] # e_ij = Bhi + Bhj
        edges.data['e'] = e_ij
        return {'Ah_j' : Ah_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        # Anisotropic MaxPool aggregation
        
        Ah_j = nodes.mailbox['Ah_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        
        Ah_j = sigma_ij * Ah_j
        if self.activation:
            Ah_j = self.activation(Ah_j)
           
        c = torch.max(Ah_j, dim=1)[0]
        return {'c' : c}

    def forward(self, g, h):
        h_in = h              # for residual connection
        h = self.dropout(h)
        
        g.ndata['h']  = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.update_all(self.message_func, 
                     self.reduce_func,
                     self.nodeapply)
        h = g.ndata['h']
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
        
        if self.residual:
            h = h_in + h       # residual connection
        
        return h
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.residual)


##############################################################


class GraphSageLayerEdgeReprFeat(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type, batch_norm, residual=False, 
                 bias=True, dgl_builtin=False):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.batch_norm = batch_norm
        self.residual = residual
        
        if in_feats != out_feats:
            self.residual = False
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.activation = activation
        
        self.A = nn.Linear(in_feats, out_feats, bias=bias)
        self.B = nn.Linear(in_feats, out_feats, bias=bias)
        self.C = nn.Linear(in_feats, out_feats, bias=bias)
        
        self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout, bias=bias)
        
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)
            self.batchnorm_e = nn.BatchNorm1d(out_feats)
            
    def message_func(self, edges):
        Ah_j = edges.src['Ah']    
        e_ij = edges.data['Ce'] + edges.src['Bh'] + edges.dst['Bh'] # e_ij = Ce_ij + Bhi + Bhj
        edges.data['e'] = e_ij
        return {'Ah_j' : Ah_j, 'e_ij' : e_ij}

    def reduce_func(self, nodes):
        # Anisotropic MaxPool aggregation
        
        Ah_j = nodes.mailbox['Ah_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e) # sigma_ij = sigmoid(e_ij)
        
        Ah_j = sigma_ij * Ah_j
        if self.activation:
            Ah_j = self.activation(Ah_j)
           
        c = torch.max(Ah_j, dim=1)[0]
        return {'c' : c}

    def forward(self, g, h, e):
        h_in = h              # for residual connection
        e_in = e
        h = self.dropout(h)
        
        g.ndata['h']  = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 
        g.update_all(self.message_func, 
                     self.reduce_func,
                     self.nodeapply)
        h = g.ndata['h']
        e = g.edata['e']
        
        if self.activation:
            e = self.activation(e) # non-linear activation
        
        if self.batch_norm:
            h = self.batchnorm_h(h)
            e = self.batchnorm_e(e)
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, 
            self.residual)