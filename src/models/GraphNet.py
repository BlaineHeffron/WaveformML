import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, knn_graph, GCNConv

from src.models.BasicNetwork import *


# see https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
# try https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)


class GraphNet(nn.Module):
    def __init__(self, config):
        super(GraphNet, self).__init__()
        self.config = config
        if hasattr(config.net_config.hparams, "k"):
            self.k = config.net_config.hparams.k
        else:
            self.k = 6
        if config.net_config.net_type == "Edge":
            self.conv1 = DynamicEdgeConv(self.config.system_config.n_samples * 2, 16, k=self.k)
            self.conv2 = DynamicEdgeConv(16, self.config.system_config.n_type, k=self.k)
        else:
            self.conv1 = GCNConv(self.config.system_config.n_samples * 2, 16)
            self.conv2 = GCNConv(16, self.config.system_config.n_type)

    def forward(self, data):
        x, coo = data[1], data[0]
        x = self.conv1(x, coo[:, 0:2], batch=coo[:, 2])
        x = F.relu(x)
        x = self.conv2(x, coo[:, 0:2], batch=coo[:, 2])
        return x
