"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.bi_gated_gcn_net import biGatedGCNNet, biGatedGCNNet_IL, biGatedGCNNet_ALL, biGatedGCNNet_lazy
from nets.SBMs_node_classification.bi_gcn_net import biGCNNet
from nets.SBMs_node_classification.bi_gcn_net_for_Eval import biGCNNet_for_Eval
from nets.SBMs_node_classification.gcn_net_for_Eval import GCNNet_for_Eval
from nets.SBMs_node_classification.bi_gat_net import biGATNet
from nets.SBMs_node_classification.bi_graphsage_net import biGraphSageNet

def biGatedGCN(net_params):
    return biGatedGCNNet(net_params)

def biGatedGCN_IL(net_params):
    return biGatedGCNNet_IL(net_params)

def biGatedGCN_ALL(net_params):
    return biGatedGCNNet_ALL(net_params)

def biGatedGCN_lazy(net_params):
    return biGatedGCNNet_lazy(net_params)

def biGCN_for_Eval(net_params):
    return biGCNNet_for_Eval(net_params)

def GCN_for_Eval(net_params):
    return GCNNet_for_Eval(net_params)

def biGCN(net_params):
    return biGCNNet(net_params)

def biGAT(net_params):
    return biGATNet(net_params)

def biGraphSage(net_params):
    return biGraphSageNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'biGatedGCN': biGatedGCN,
        'biGatedGCN_IL': biGatedGCN_IL,
        'biGatedGCN_ALL': biGatedGCN_ALL,
        'biGatedGCN_lazy': biGatedGCN_lazy,
        'biGCN': biGCN,
        'biGCN_for_Eval': biGCN_for_Eval,
        'GCN_for_Eval': GCN_for_Eval,
        'biGAT': biGAT,
        'biGraphSage': biGraphSage
    }
        
    return models[MODEL_NAME](net_params)