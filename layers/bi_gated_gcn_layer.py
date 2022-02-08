import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

class biGatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, assign_dim, sigma=1, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.assign_dim = assign_dim
        self.sigma = sigma
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.s1 = nn.Linear(input_dim,  output_dim, bias=True)
        self.s2 = nn.Linear(output_dim,  self.assign_dim, bias=True)
        self.metric = nn.Linear( self.assign_dim,  self.assign_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(h))),dim=1) # soft assignment
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma_e'] = torch.sigmoid(g.edata['e'])
        
        g.apply_edges(fn.u_sub_v('Sh', 'Sh', 'Sd')) # for cluster distance: (si - sj)
        Sd = g.edata['Sd'] # sum_edges, assign_dim
        Sd_h = self.metric(Sd) # sum_edges, assign_dim;  D = sqrt( (si - sj) W W^t (si - sj)^t )
        #dense_D = torch.matmul(Sd_h,Sd_h.transpose(0,1)) # sum_edges, sum_edges
        #D = torch.sqrt(torch.einsum("ii->i",dense_D).unsqueeze(1)) # sum_edges, 1
        D = torch.sqrt(torch.sum(Sd_h*Sd_h,dim=1)).unsqueeze(1) # sum_edges, 1
        g.edata['GD'] = torch.exp( -D / ( 2*(self.sigma**2) ) ) # sum_edges, 1 # G = GaussianRBF(D)
        g.edata['sigma_GD'] = torch.sigmoid(g.edata['GD'])
        
        g.edata['sigma'] = g.edata['sigma_e'] * g.edata['sigma_GD']
        
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma_e', 'm'), fn.sum('m', 'sum_sigma_e'))
        g.update_all(fn.copy_e('sigma_GD', 'm'), fn.sum('m', 'sum_sigma_GD'))
        g.ndata['h'] = ( g.ndata['Ah'] + g.ndata['sum_sigma_h'] / 
        ((g.ndata['sum_sigma_e'] + 1e-6) * (g.ndata['sum_sigma_GD'] + 1e-6)))
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, g.ndata['Sh']
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    
class biGatedGCNLayer2(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, assign_dim, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.assign_dim = assign_dim
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.s1 = nn.Linear(input_dim,  output_dim, bias=True)
        self.s2 = nn.Linear(output_dim,  self.assign_dim, bias=True)
        self.Q = nn.Linear(self.assign_dim,  output_dim, bias=True)
        self.K = nn.Linear(self.assign_dim,  output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def scaledDP(self, edges):
        return {'att': torch.sum(edges.data['Skq'],dim=1) / edges.src['Sk'].shape[1] }
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(h)))) # soft assignment
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['ehat'] = torch.sigmoid(g.edata['e'])
        
        Sh = g.ndata['Sh'] 
        g.ndata['Sq'] = self.Q(Sh) # quary for dst
        g.ndata['Sk'] = self.K(Sh) # key for src
        g.apply_edges(fn.u_mul_v('Sk', 'Sq', 'Skq')) 
        g.apply_edges(self.scaledDP) # scaled-dop product
        g.edata['sigma'] = g.edata['ehat'] * g.edata['att'].unsqueeze(1) # bilateral gating
        
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, Sh
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    
class biGatedGCNLayer3(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, assign_dim, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.assign_dim = assign_dim
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.s1 = nn.Linear(input_dim,  output_dim, bias=True)
        self.s2 = nn.Linear(output_dim,  self.assign_dim, bias=True)
        self.F = nn.Parameter(torch.FloatTensor(size=(self.assign_dim * 2, output_dim)))
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def concat_message_function(self, edges):
        return {'cSh': torch.cat([edges.src['Sh'], edges.dst['Sh']],dim=1)}
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(h)))) # soft assignment
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(self.concat_message_function) # concat the src&dst soft assignments
        g.edata['Se'] = g.edata['cSh'] @ self.F # update
        
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce'] + g.edata['Se'] # edge feat considering the soft assignments
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, g.ndata['Sh']
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

class biGatedGCNLayer4(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, assign_dim, sigma=1, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.assign_dim = assign_dim
        self.sigma = sigma
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.s1 = nn.Linear(input_dim,  output_dim, bias=True)
        self.s2 = nn.Linear(output_dim,  self.assign_dim, bias=True)
        self.metric = nn.Linear( self.assign_dim,  self.assign_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(h))),dim=1) # soft assignment
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma_e'] = torch.sigmoid(g.edata['e'])
        
        g.apply_edges(fn.u_sub_v('Sh', 'Sh', 'Sd')) # for cluster distance: (si - sj)
        Sd = g.edata['Sd'] # sum_edges, assign_dim
        Sd_h = self.metric(Sd) # sum_edges, assign_dim;  D = sqrt( (si - sj) W W^t (si - sj)^t )
        #dense_D = torch.matmul(Sd_h,Sd_h.transpose(0,1)) # sum_edges, sum_edges
        #D = torch.sqrt(torch.einsum("ii->i",dense_D).unsqueeze(1)) # sum_edges, 1
        D = torch.sqrt(torch.sum(Sd_h*Sd_h,dim=1)).unsqueeze(1) # sum_edges, 1
        g.edata['GD'] = torch.exp( -D / ( 2*(self.sigma**2) ) ) # sum_edges, 1 # G = GaussianRBF(D)
        g.edata['sigma_GD'] = torch.sigmoid(g.edata['GD'])
        
        g.edata['sigma'] = g.edata['sigma_e'] * g.edata['sigma_GD']
        
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma_e', 'm'), fn.sum('m', 'sum_sigma_e'))
        g.update_all(fn.copy_e('sigma_GD', 'm'), fn.sum('m', 'sum_sigma_GD'))
        g.ndata['h'] = ( g.ndata['Ah'] + g.ndata['sum_sigma_h'] / 
        ((g.ndata['sum_sigma_e'] + 1e-6) * (g.ndata['sum_sigma_GD'] + 1e-6)))
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e, g.ndata['Sh']
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)    