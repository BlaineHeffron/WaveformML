import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, knn_graph, GCNConv, TopKPooling, global_max_pool

from src.models.BasicNetwork import *


# see https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
# try https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv
from src.models.ConvBlocks import LinearBlock


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, feat, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(feat, edge_index)


class DynamicGraphConv(GCNConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicGraphConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, feat, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicGraphConv, self).forward(feat, edge_index)


class GraphNet(nn.Module):
    def __init__(self, config):
        super(GraphNet, self).__init__()
        self.log = logging.getLogger(__name__)
        self.config = config
        self.lin_outputs = 0
        self.n_lin = 0
        if hasattr(config.net_config.hparams, "n_lin"):
            self.n_lin = config.net_config.hparams.n_lin
            if hasattr(config.system_config, "n_type"):
                self.lin_outputs = config.system_config.n_type
            elif hasattr(config.net_config, "n_out"):
                self.lin_outputs = config.net_config.n_out
            else:
                raise IOError("Need to specify system_config.n_type for classifier or net_config.n_out to determine number of outputs of linear layer")
        if hasattr(config.net_config.hparams, "k"):
            self.k = config.net_config.hparams.k
        else:
            self.k = 6
        self.graph_out = 10
        if hasattr(config.net_config.hparams, "graph_params"):
            if hasattr(config.net_config.hparams.graph_params, "n_out"):
                self.graph_out = self.graph_out = config.net_config.hparams.graph_params.n_out
            else:
                self.log.debug("no net_config.hparams.graph_params.n_out set, setting graph outputs default to {}".format(self.graph_out))
        else:
            self.log.debug("no net_config.hparams.graph_params.n_out set, setting graph outputs default to {}".format(self.graph_out))
        if config.net_config.net_type == "Edge":
            self.conv1 = DynamicEdgeConv(self.config.system_config.n_samples * 2, 16, k=self.k)
            self.conv2 = DynamicEdgeConv(16, self.graph_out, k=self.k)
        else:
            self.conv1 = GCNConv(self.config.system_config.n_samples * 2, 16)
            self.conv2 = GCNConv(16, self.graph_out)
        if self.n_lin > 0:
            self.pool = TopKPooling(self.graph_out, 0.8)
            self.linear = LinearBlock(self.graph_out, self.lin_outputs, self.n_lin).func
        else:
            self.linear = None

    def forward(self, data):
        x, coo = data[1], data[0].long()
        edge_index = knn_graph(coo, self.k, coo[:, 2], loop=False)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        batch_size = coo[-1, 2] + 1
        print(batch_size.item())
        if self.n_lin > 0:
            #x, edge_index, _, batch, = self.pool(x, edge_index=edge_index, batch=coo[:, 2])
            #x, _, _, _, batch, _ = self.pool(x, edge_index=edge_index, batch=coo[:, 2])
            x = global_max_pool(x, coo[:, 2], size=batch_size.item())
            x = self.linear(x)
        return x
