"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.TUs_graph_classification.bi_gated_gcn_net import biGatedGCNNet
from nets.TUs_graph_classification.bi_gcn_net import biGCNNet
from nets.TUs_graph_classification.bi_gat_net import biGATNet
from nets.TUs_graph_classification.bi_graphsage_net import biGraphSageNet

def biGatedGCN(net_params):
    return biGatedGCNNet(net_params)

def biGCN(net_params):
    return biGCNNet(net_params)

def biGAT(net_params):
    return biGATNet(net_params)

def biGraphSage(net_params):
    return biGraphSageNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'biGatedGCN': biGatedGCN,
        'biGCN': biGCN,
        'biGAT': biGAT,
        'biGraphSage': biGraphSage
    }
        
    return models[MODEL_NAME](net_params)