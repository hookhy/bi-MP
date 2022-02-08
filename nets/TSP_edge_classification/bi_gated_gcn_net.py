import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.bi_gated_gcn_layer import biGatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class biGatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.sg_flag = True
        self.assign_dim = net_params['assign_dim']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        self.sigma = net_params['sigma']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
    
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) ])
        self.layers.append(biGatedGCNLayer(hidden_dim, hidden_dim, dropout, self.batch_norm, 
                                           self.assign_dim, self.sigma, residual=self.residual))
        for _ in range(n_layers-3):
            self.layers.append(GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual)) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h, e):
        
        h = self.embedding_h(h.float())
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        
        # convnets
        cnt=0
        for conv in self.layers:
            if cnt ==1:
                h, e, self.s = conv(g, h, e)
            else:
                h, e = conv(g, h, e)
            cnt+=1
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
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

class biGatedGCNNet_IL(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.sg_flag = True
        self.assign_dim = net_params['assign_dim']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        self.sigma = net_params['sigma']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
    
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual)  ])
        for l in range(self.n_layers-1):
            if l == self.n_layers-2:
                if l % 2 == 0:
                    self.layers.append(biGatedGCNLayer(hidden_dim, out_dim, dropout,
                                                        self.batch_norm, self.assign_dim, self.sigma, self.residual)) 
                else:
                    self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout,
                                                        self.batch_norm, self.residual)) 
            else:
                if l % 2 == 0:
                    self.layers.append(biGatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                        self.batch_norm, self.assign_dim, self.sigma, self.residual)) 
                else:
                    self.layers.append(GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                        self.batch_norm, self.residual)) 
        
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h, e):
        
        h = self.embedding_h(h.float())
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        
        # convnets
        s = []
        for conv in self.layers:
            try:
                h, e, tmp = conv(g, h, e)
                s.append(tmp)
            except:
                h, e = conv(g, h, e)
        self.S = torch.stack(s,dim=0).to(self.device) # num_pool x sum_node x assign_dim
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e'], self.S
    
    def sup_loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    
    def unsup_loss(self, g, soft_assign, mode):
        if mode == 'mincut':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            d = torch.sparse_coo_tensor(torch.tensor([range(adj.size()[0]),range(adj.size()[0])]), 
                                        torch.sparse.sum(adj,dim=1).to_dense())
            mincut_loss, ortho_loss = 0, 0
            for l in range(soft_assign.shape[0]):
                one_s = soft_assign[l]
                out_adj = torch.mm(one_s.transpose(0,1),torch.sparse.mm(adj,one_s))
                out_d = torch.mm(one_s.transpose(0,1),torch.sparse.mm(d,one_s))

                mincut_num = torch.einsum('ii->', out_adj)
                mincut_den = torch.einsum('ii->', out_d)
                mincut_loss += -(mincut_num / mincut_den)

                ss = torch.matmul(one_s.transpose(0, 1), one_s)
                i_s = torch.eye(one_s.shape[1]).type_as(ss)
                ortho_loss += torch.norm(
                ss / torch.norm(ss, dim=(-0, -1), keepdim=True) -
                i_s / torch.norm(i_s), dim=(-0, -1)) 
        
            return ( mincut_loss + ortho_loss ) / soft_assign.shape[0]
        elif mode == 'diffpool':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            ent_loss, linkpred_loss = 0, 0
            for l in range(soft_assign.shape[0]):
                one_s = soft_assign[l]
                ent_loss += torch.distributions.Categorical(probs=one_s).entropy().mean(-1)
                linkpred_loss += torch.add( -one_s.matmul(one_s.transpose(0,1)),adj).norm(dim=(0,1)) / (adj.size(0)*adj.size(1))
            
            return (ent_loss + linkpred_loss) / soft_assign.shape[0]

class biGatedGCNNet_ALL(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        self.sg_flag = True
        self.assign_dim = net_params['assign_dim']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        self.sigma = net_params['sigma']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
    
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        
        self.layers = nn.ModuleList([ biGatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.assign_dim, self.sigma, self.residual ) for _ in range(self.n_layers-1) ]) 
        self.layers.append(biGatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.assign_dim, self.sigma, self.residual))
        self.MLP_layer = MLPReadout(2*out_dim, n_classes)
        
    def forward(self, g, h, e):
        
        h = self.embedding_h(h.float())
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        
        # convnets
        s = []
        for conv in self.layers:
            h, e, tmp = conv(g, h, e)
            s.append(tmp)
        self.S = torch.stack(s,dim=0).to(self.device) # num_pool x sum_node x assign_dim
        g.ndata['h'] = h
        
        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        return g.edata['e'], self.S
    
    def sup_loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    
    def unsup_loss(self, g, soft_assign, mode):
        if mode == 'mincut':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            d = torch.sparse_coo_tensor(torch.tensor([range(adj.size()[0]),range(adj.size()[0])]), 
                                        torch.sparse.sum(adj,dim=1).to_dense())
            mincut_loss, ortho_loss = 0, 0
            for l in range(self.n_layers):
                one_s = soft_assign[l]
                out_adj = torch.mm(one_s.transpose(0,1),torch.sparse.mm(adj,one_s))
                out_d = torch.mm(one_s.transpose(0,1),torch.sparse.mm(d,one_s))

                mincut_num = torch.einsum('ii->', out_adj)
                mincut_den = torch.einsum('ii->', out_d)
                mincut_loss += -(mincut_num / mincut_den)

                ss = torch.matmul(one_s.transpose(0, 1), one_s)
                i_s = torch.eye(one_s.shape[1]).type_as(ss)
                ortho_loss += torch.norm(
                ss / torch.norm(ss, dim=(-0, -1), keepdim=True) -
                i_s / torch.norm(i_s), dim=(-0, -1)) 
        
            return ( mincut_loss + ortho_loss ) / self.n_layers
        elif mode == 'diffpool':
            adj = g.adjacency_matrix(transpose=True, ctx=soft_assign.device)
            ent_loss, linkpred_loss = 0, 0
            for l in range(self.n_layers):
                one_s = soft_assign[l]
                ent_loss += torch.distributions.Categorical(probs=one_s).entropy().mean(-1)
                linkpred_loss += torch.add( -one_s.matmul(one_s.transpose(0,1)),adj).norm(dim=(0,1)) / (adj.size(0)*adj.size(1))
            
            return (ent_loss + linkpred_loss) / self.n_layers