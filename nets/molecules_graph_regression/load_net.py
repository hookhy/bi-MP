"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.molecules_graph_regression.gated_gcn_net_for_Eval import GatedGCNNet_for_Eval
from nets.molecules_graph_regression.bi_gated_gcn_net_for_Eval import biGatedGCNNet_for_Eval
from nets.molecules_graph_regression.gcn_net_for_Eval import GCNNet_for_Eval
from nets.molecules_graph_regression.bi_gcn_net_for_Eval import biGCNNet_for_Eval
from nets.molecules_graph_regression.bi_gated_gcn_net import biGatedGCNNet
from nets.molecules_graph_regression.bi_gcn_net import biGCNNet
from nets.molecules_graph_regression.bi_gat_net import biGATNet
from nets.molecules_graph_regression.bi_graphsage_net import biGraphSageNet

def biGatedGCN(net_params):
    return biGatedGCNNet(net_params)

def GatedGCN_for_Eval(net_params):
    return GatedGCNNet_for_Eval(net_params)

def biGatedGCN_for_Eval(net_params):
    return biGatedGCNNet_for_Eval(net_params)

def biGCN(net_params):
    return biGCNNet(net_params)

def GCN_for_Eval(net_params):
    return GCNNet_for_Eval(net_params)

def biGCN_for_Eval(net_params):
    return biGCNNet_for_Eval(net_params)

def biGAT(net_params):
    return biGATNet(net_params)

def biGraphSage(net_params):
    return biGraphSageNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'biGCN': biGCN,
        'GCN_for_Eval': GCN_for_Eval,
        'biGCN_for_Eval': biGCN_for_Eval,
        'biGAT': biGAT,
        'biGraphSage': biGraphSage,
        'GatedGCN_for_Eval': GatedGCN_for_Eval,
        'biGatedGCN_for_Eval': biGatedGCN_for_Eval,
        'biGatedGCN': biGatedGCN
    }
        
    return models[MODEL_NAME](net_params)