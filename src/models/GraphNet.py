import torch.nn.functional as F
from torch import cat
from torch.nn import ReLU
from torch_geometric.nn import EdgeConv, knn_graph, GCNConv, global_max_pool, global_mean_pool, \
    MessagePassing, SAGEConv, GraphConv, GATConv, GATv2Conv, TransformerConv, TAGConv, GINConv, GINEConv, ARMAConv, \
    SGConv, GMMConv, EGConv, FeaStConv, LEConv, ClusterGCNConv, GENConv, HypergraphConv, GCN2Conv, PANConv, FiLMConv, \
    SuperGATConv

from src.models.BasicNetwork import *


# see https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
# try https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv
from src.models.ConvBlocks import LinearBlock, LinearPlanes
from src.utils.util import DictionaryUtility


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

class GraphLayer(nn.Module):
    """
    @param net: MessagePassing layer that must conform to __init__(self, in_channels, out_channels, *args, **kwargs)
    """
    def __init__(self, net: MessagePassing, out_channels, pool_ratio):
        super(GraphLayer, self).__init__()
        self.log = logging.getLogger(__name__)
        self.graph_net = net

    def forward(self, x, edge_index):
        x = F.relu(self.graph_net(x, edge_index))
        #x, edge_index, edge_attr, batch, perm, score = self.pool(x, edge_index, None, batch)
        #return x, cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1), edge_index, batch
        return x



class GraphNet(nn.Module):
    def __init__(self, config):
        super(GraphNet, self).__init__()
        self.log = logging.getLogger(__name__)
        self.config = config
        self.lin_outputs = 0
        self.feat_size = self.config.system_config.n_samples*2
        self.n_lin = 0
        self.n_graph = config.net_config.hparams.n_graph
        self.graph_index = config.net_config.hparams.graph_class_index
        self.graph_class = self.retrieve_class(config.net_config.hparams.graph_class_index)
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
        self.graph_params = {}
        self.graph_layers = []
        self.graph_planes = []
        if hasattr(config.net_config.hparams, "graph_params"):
            self.graph_params = DictionaryUtility.to_dict(config.net_config.hparams.graph_params)
        if hasattr(config.net_config.hparams, "graph_out"):
            self.graph_out = config.net_config.hparams.graph_out
        else:
            self.log.debug("no net_config.hparams.graph_out set, setting graph outputs default to {}".format(self.graph_out))
        self.pool_ratio = 0.8
        if hasattr(config.net_config.hparams, "pool_ratio"):
            self.pool_ratio = config.net_config.hparams.pool_ratio
            self.log.debug("Setting pool ratio to {}".format(self.pool_ratio))
        if self.n_lin > 0:
            self.linear = LinearBlock(self.graph_out, self.lin_outputs, self.n_lin).func
        else:
            self.linear = None
        self.reduction_type = "linear"
        if hasattr(config.net_config.hparams, "reduction_type"):
            self.reduction_type = config.net_config.hparams.reduction_type
        self.graph_planes = [self.feat_size]
        if self.reduction_type == "linear":
            red = int((self.graph_planes[0] - self.graph_out) / self.n_graph)
            for n in range(self.n_graph):
                self.graph_planes.append(self.graph_planes[-1] - red)
        elif self.reduction_type == "geometric":
            red = float(self.graph_out / self.graph_planes[0]) ** (1./self.n_graph)
            for n in range(self.n_graph):
                self.graph_planes.append(int(self.graph_planes[-1] * red))
        else:
            raise IOError("net_config.hparams.reduction_type must be either linear or geometric")
        self.graph_planes[-1] = int(self.graph_out)
        for i in range(self.n_graph):
            nin = self.graph_planes[i]
            nout = self.graph_planes[i+1]
            self.log.debug("Adding graph layer of type {0} with nin: {1}, nout: {2}".format(self.graph_class, nin, nout))
            if self.class_needs_nn(self.graph_index):
                self.graph_layers.append(GraphLayer(self.graph_class(LinearPlanes([nin, nout], activation=ReLU()), **self.graph_params), nout, self.pool_ratio))
            else:
                self.graph_layers.append(GraphLayer(self.graph_class(nin, nout, **self.graph_params), nout, self.pool_ratio))
        self.graph_layers = nn.Sequential(*self.graph_layers)


    def forward(self, data):
        x, coo = data[1], data[0].long()
        edge_index = knn_graph(coo, self.k, coo[:, 2], loop=False)
        for layer in self.graph_layers:
            #x, x1, edge_index, batch = layer(x, edge_index, batch)
            x, edge_index = layer(x, edge_index)
        if self.n_lin > 0:
            #x = cat([global_max_pool(x, coo[:, 2]), global_mean_pool(x, coo[:, 2])], dim=1)
            x = global_max_pool(x, coo[:, 2])
            x = self.linear(x)
        return x

    def class_needs_nn(self, index):
        if index in [7, 8, 13]:
            return True
        else:
            return False

    def retrieve_class(self, index):
        if index == 0:
            return GCNConv
        elif index == 1:
            return SAGEConv
        elif index == 2:
            return GraphConv
        elif index == 3:
            return GATConv
        elif index == 4:
            return GATv2Conv
        elif index == 5:
            return TransformerConv
        elif index == 6:
            return TAGConv
        elif index == 7:
            return GINConv
        elif index == 8:
            return GINEConv
        elif index == 9:
            return ARMAConv
        elif index == 10:
            return SGConv
        elif index == 11:
            return GMMConv
        elif index == 12:
            return EGConv
        elif index == 13:
            return EdgeConv
        elif index == 14:
            return FeaStConv
        elif index == 15:
            return LEConv
        elif index == 16:
            return ClusterGCNConv
        elif index == 17:
            return GENConv
        elif index == 18:
            return HypergraphConv
        elif index == 19:
            return GCN2Conv
        elif index == 20:
            return PANConv
        elif index == 21:
            return FiLMConv
        elif index == 22:
            return SuperGATConv



