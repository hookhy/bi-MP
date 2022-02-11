# Applying the bilateral MP scheme to the SOTA MP GNN layers

## 1. SOTA MP GNN layer

Prepare the DGL-style class of a SOTA MP-GNN layer. (You can follow the instruction of [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns), optionally).
```
# GCN example (step1)
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

# Sends a message of node feature h
# Equivalent to => return {'m': edges.src['h']}
msg = fn.copy_src(src='h', out='m')
reduce = fn.mean('m', 'h')

class customGNNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, ...):
        super().__init__()  
        
        # write your init code here

    def forward(self, g, feature, ...):
          
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod) # PRE-defined class for node embedding (e.g. linear layer, ...)
        h = g.ndata['h'] # result of graph convolution
          
        return h
```

This layer class must be defined in the *layers/* directory.

## 2. Calculate the Modular gradient (See [the author's ArXiv paper](https://arxiv.org/abs/2202.04768) for a technical detail)

This goal suggests the need for the simultaneous definition of 
 1) Node class assignment, and 
 2) Measurement of the pairwise difference based on the class information.
in the __init()__, and __forward()__, respectively:

```
 def __init__(self, in_dim, out_dim, assign_dim, ...):
        super().__init__()  
        
        # write your init code here
        
        # (step2) calculate the modular gradient (assign_dim is a hyperparameter)
        self.s1 = nn.Linear(in_dim,  out_dim, bias=True)
        self.s2 = nn.Linear(out_dim,  self.assign_dim, bias=True)
        self.metric = nn.Linear( self.assign_dim,  self.assign_dim, bias=True)
```

```
 def forward(self, g, feature, ...):
          
        g.ndata['h'] = feature
        #g.update_all(msg, reduce)
        
        #(step2) apply the modular gradient to modify the message propagation
        g.ndata['Sh'] = F.softmax(self.s2(F.relu(self.s1(feature))),dim=1) # soft assignment forwarding
        
        g.apply_edges(fn.u_sub_v('Sh', 'Sh', 'Sd')) # for cluster distance: (si - sj)
        Sd = g.edata['Sd'] # sum_edges, assign_dim
        Sd_h = self.metric(Sd) # sum_edges, assign_dim;  D = sqrt( (si - sj) W W^t (si - sj)^t )
        D = torch.sqrt(torch.sum(Sd_h*Sd_h,dim=1)).unsqueeze(1) # sum_edges, 1
        g.edata['GD'] = torch.exp( -D / ( 2*(self.sigma**2) ) ) # sum_edges, 1 # G = GaussianRBF(D)
        g.edata['sigma_GD'] = torch.sigmoid(g.edata['GD'])
        
        # normalization
        g.update_all(fn.u_mul_e('h', 'sigma_GD', 'm'), fn.mean('m', 'sum_sigma_GD_h'))
        g.update_all(fn.copy_e('sigma_GD', 'm'), fn.sum('m', 'sum_sigma_GD'))
        g.ndata['h'] = (g.ndata['sum_sigma_GD_h'] / (g.ndata['sum_sigma_GD'] + 1e-6))
        
        # applying
        g.apply_nodes(func=self.apply_mod) # PRE-defined class for node embedding (e.g. linear layer, ...)
        h = g.ndata['h'] # result of graph convolution
          
        return h
```
