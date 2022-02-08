"""
    Utility file to select GraphNN model as
    selected by the user
"""


from nets.superpixels_graph_classification.bi_gcn_net import biGCNNet
from nets.superpixels_graph_classification.bi_gat_net import biGATNet
from nets.superpixels_graph_classification.bi_graphsage_net import biGraphSageNet
from nets.superpixels_graph_classification.bi_gated_gcn_net import biGatedGCNNet, biGatedGCNNet_IL, biGatedGCNNet_ALL

def biGCN(net_params):
    return biGCNNet(net_params)

def biGAT(net_params):
    return biGATNet(net_params)

def biGraphSage(net_params):
    return biGraphSageNet(net_params)

def biGatedGCN(net_params):
    return biGatedGCNNet(net_params)

def biGatedGCN_IL(net_params):
    return biGatedGCNNet_IL(net_params)

def biGatedGCN_ALL(net_params):
    return biGatedGCNNet_ALL(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'biGCN': biGCN,
        'biGAT': biGAT,
        'biGraphSage': biGraphSage,
        'biGatedGCN': biGatedGCN,
        'biGatedGCN_IL': biGatedGCN_IL,
        'biGatedGCN_ALL': biGatedGCN_ALL
    }
        
    return models[MODEL_NAME](net_params)