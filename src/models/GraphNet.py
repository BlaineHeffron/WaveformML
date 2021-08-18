import torch.nn.functional as F
from torch_geometric.transforms import Cartesian, LocalCartesian
from torch.nn import ReLU, ModuleList, Linear
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, knn_graph, GCNConv, global_max_pool, global_mean_pool, \
    MessagePassing, SAGEConv, GraphConv, GATConv, GATv2Conv, TransformerConv, TAGConv, GINConv, GINEConv, ARMAConv, \
    SGConv, GMMConv, EGConv, FeaStConv, LEConv, ClusterGCNConv, GENConv, HypergraphConv, GCN2Conv, PANConv, FiLMConv, \
    SuperGATConv

from torch_geometric.nn import PointConv

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
    def __init__(self, net: MessagePassing, uses_edge_attr=False, edge_attr_dim_match=False, match_size=None, edge_attr_dim=None):
        super(GraphLayer, self).__init__()
        self.log = logging.getLogger(__name__)
        self.graph_net = net
        self.uses_edge_attr = uses_edge_attr
        self.edge_attr_dim_match = edge_attr_dim_match
        if edge_attr_dim_match:
            if not edge_attr_dim:
                raise RuntimeError("edge_attr_dim must be supplied if edge_attr_dim_match is true")
            if not match_size:
                raise RuntimeError("match_size must be supplied if edge_attr_dim_match is true")
            self.linear = Linear(edge_attr_dim, match_size)

    def forward(self, data: Data):
        if self.uses_edge_attr:
            if self.edge_attr_dim_match:
                edge_attr = self.linear(data.edge_attr)
                data.x = F.relu(self.graph_net(data.x, data.edge_index, edge_attr))
            else:
                data.x = F.relu(self.graph_net(data.x, data.edge_index, data.edge_attr))
        else:
            data.x = F.relu(self.graph_net(data.x, data.edge_index))
        #x, edge_index, edge_attr, batch, perm, score = self.pool(x, edge_index, None, batch)
        #return x, cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1), edge_index, batch
        return data



class GraphNet(nn.Module):
    def __init__(self, config):
        super(GraphNet, self).__init__()
        self.log = logging.getLogger(__name__)
        self.config = config
        self.lin_outputs = 0
        self.feat_size = self.config.system_config.n_samples*2
        self.n_lin = 0
        self.n_graph = 0
        self.n_expansion = 0
        self.expansion_factor = 1.0
        if hasattr(config.net_config.hparams, "n_graph"):
            self.n_graph = config.net_config.hparams.n_graph
        elif hasattr(config.net_config.hparams, "n_contract"):
            if not hasattr(config.net_config.hparams, "n_expand"):
                raise IOError("if net_config.hparams.n_graph not specified, must specify n_expand and n_contract")
            self.n_graph = config.net_config.hparams.n_contract + config.net_config.hparams.n_expand
        else:
            raise IOError("if net_config.hparams.n_graph not specified, must specify n_expand and n_contract")
        if hasattr(config.net_config.hparams, "expansion_factor"):
            self.expansion_factor = config.net_config.hparams.expansion_factor
        if hasattr(config.net_config.hparams, "n_expand"):
            self.n_expansion = config.net_config.hparams.n_expand
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
        self.graph_layers = ModuleList()
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
        self.use_self_loops = False
        if hasattr(config.net_config.hparams, "self_loop"):
            self.use_self_loops = config.net_config.hparams.self_loop
        if self.n_lin > 0:
            out_modifier = 1
            if "heads" in self.graph_params.keys():
                out_modifier = self.graph_params["heads"]
            self.linear = LinearBlock(self.graph_out*out_modifier, self.lin_outputs, self.n_lin).func
        else:
            self.linear = None
        self.edge_attr_transform = Cartesian()
        self.edge_attr_dim = 2
        if hasattr(config.net_config.hparams, "edge_transform"):
            if config.net_config.hparams.edge_transform == "cartesian":
                self.edge_attr_transform = Cartesian()
                self.edge_attr_dim = 2
            elif config.net_config.hparams.edge_transform == "localcartesian":
                self.edge_attr_transform = LocalCartesian()
                self.edge_attr_dim = 2
            else:
                raise IOError("net_config.hparams.edge_transform must be one of 'cartesian', 'localcartesian'")
        self.reduction_type = "linear"
        if hasattr(config.net_config.hparams, "reduction_type"):
            self.reduction_type = config.net_config.hparams.reduction_type
        self.graph_planes = [self.feat_size]
        self.n_contract = self.n_graph - self.n_expansion
        if self.reduction_type == "linear":
            if self.n_expansion > 0:
                exp = int((self.graph_planes[0]*self.expansion_factor - self.graph_planes[0]) / self.n_expansion)
                for n in range(self.n_expansion):
                    self.graph_planes.append(self.graph_planes[-1] + exp)
                if self.n_contract > 0:
                    red = int((self.graph_planes[-1] - self.graph_out) / self.n_contract)
                    for n in range(self.n_contract):
                        self.graph_planes.append(self.graph_planes[-1] - red)
            else:
                red = int((self.graph_planes[0] - self.graph_out) / self.n_graph)
                for n in range(self.n_graph):
                    self.graph_planes.append(self.graph_planes[-1] - red)
        elif self.reduction_type == "geometric":
            if self.n_expansion > 0:
                exp = float(self.expansion_factor) ** (1./self.n_expansion)
                for n in range(self.n_expansion):
                    self.graph_planes.append(int(self.graph_planes[-1] * exp))
                if self.n_contract > 0:
                    red = float(self.graph_out / self.graph_planes[-1]) ** (1. / self.n_contract)
                    for n in range(self.n_contract):
                        self.graph_planes.append(int(self.graph_planes[-1] * red))
            else:
                red = float(self.graph_out / self.graph_planes[0]) ** (1./self.n_graph)
                for n in range(self.n_graph):
                    self.graph_planes.append(int(self.graph_planes[-1] * red))
        else:
            raise IOError("net_config.hparams.reduction_type must be either linear or geometric")
        self.graph_planes[-1] = int(self.graph_out)
        default_params = self.default_positional_params(self.graph_index)
        self.uses_edge_attr = self.needs_edge_attr(self.graph_index)
        self.default_keyword_params(self.graph_index)
        for i in range(self.n_graph):
            nin = self.graph_planes[i]
            nout = self.output_modifier(self.graph_index, self.graph_planes[i+1])
            match_ind = nin
            dim_match = False
            if self.edge_attr_dimension_match(self.graph_index):
                match_ind = nin
                dim_match = True
            self.log.debug("Adding graph layer of type {0} with nin: {1}, nout: {2}".format(self.graph_class, nin, nout))
            if self.class_needs_nn(self.graph_index):
                nlin_in = self.nn_input_modifier(self.graph_index, i)*nin
                if dim_match:
                    match_ind = nlin_in
                self.graph_layers.append(GraphLayer(self.graph_class(LinearPlanes([nlin_in, nout], activation=ReLU()), *default_params, **self.graph_params), self.uses_edge_attr, dim_match, match_ind, self.edge_attr_dim))
            else:
                nlin_in = self.nn_input_modifier(self.graph_index, i)*nin
                self.graph_layers.append(GraphLayer(self.graph_class(nlin_in, nout, *default_params, **self.graph_params), self.uses_edge_attr, dim_match, match_ind, self.edge_attr_dim))


    def forward(self, data):
        coo = data[0].long()
        edge_index = knn_graph(coo, self.k, coo[:, 2], loop=self.use_self_loops)
        geom_data = Data(x=data[1], edge_index=edge_index, pos=coo[:, 0:2])
        if self.uses_edge_attr:
            self.edge_attr_transform(geom_data)
        for layer in self.graph_layers:
            #x, x1, edge_index, batch = layer(x, edge_index, batch)
            geom_data = layer(geom_data)
        if self.n_lin > 0:
            #x = cat([global_max_pool(x, coo[:, 2]), global_mean_pool(x, coo[:, 2])], dim=1)
            geom_data.x = global_max_pool(geom_data.x, coo[:, 2])
            geom_data.x = self.linear(geom_data.x)
        return geom_data.x

    def nn_input_modifier(self, index, num_layer):
        if index == 12:
            return 2
        else:
            if "heads" in self.graph_params and num_layer > 0:
                return self.graph_params["heads"]
            return 1

    def class_needs_nn(self, index):
        if index in [7, 12]:
            return True
        else:
            return False

    def default_positional_params(self, index):
        if index == 10:
            # degree of pseudodimensionality of coordinate and kernel size
            return [2, 2]
        else:
            return []

    def default_keyword_params(self, index):
        if index == 5:
            self.graph_params["edge_dim"] = self.edge_attr_dim

    def edge_attr_dimension_match(self, index):
        return index in [16]

    def needs_edge_attr(self, index):
        return index in [5, 10, 16]

    def output_modifier(self, index, out_channels):
        return out_channels

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
            return ARMAConv
        elif index == 9:
            return SGConv
        elif index == 10:
            return GMMConv
        elif index == 11:
            return FiLMConv
        elif index == 12:
            return EdgeConv
        elif index == 13:
            return FeaStConv
        elif index == 14:
            return LEConv
        elif index == 15:
            return ClusterGCNConv
        elif index == 16:
            return GENConv
        elif index == 17:
            return SuperGATConv


class PointNet(nn.Module):
    def __init__(self, config):
        super(PointNet, self).__init__()
        self.log = logging.getLogger(__name__)
        self.config = config
        self.lin_outputs = 0
        self.feat_size = self.config.system_config.n_samples*2
        self.n_lin = 0
        self.n_graph = 0
        self.n_expansion = 0
        self.expansion_factor = 1.0
        self.ndim = 2
        if hasattr(config.net_config.hparams, "n_graph"):
            self.n_graph = config.net_config.hparams.n_graph
        elif hasattr(config.net_config.hparams, "n_contract"):
            if not hasattr(config.net_config.hparams, "n_expand"):
                raise IOError("if net_config.hparams.n_graph not specified, must specify n_expand and n_contract")
            self.n_graph = config.net_config.hparams.n_contract + config.net_config.hparams.n_expand
        else:
            raise IOError("if net_config.hparams.n_graph not specified, must specify n_expand and n_contract")
        if hasattr(config.net_config.hparams, "expansion_factor"):
            self.expansion_factor = config.net_config.hparams.expansion_factor
        if hasattr(config.net_config.hparams, "n_expand"):
            self.n_expansion = config.net_config.hparams.n_expand
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
        self.graph_layers = ModuleList()
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
        self.use_self_loops = False
        if hasattr(config.net_config.hparams, "self_loop"):
            self.use_self_loops = config.net_config.hparams.self_loop
        if self.n_lin > 0:
            self.linear = LinearBlock(self.graph_out, self.lin_outputs, self.n_lin).func
        else:
            self.linear = None
        self.edge_attr_transform = Cartesian()
        self.edge_attr_dim = 2
        if hasattr(config.net_config.hparams, "edge_transform"):
            if config.net_config.hparams.edge_transform == "cartesian":
                self.edge_attr_transform = Cartesian()
                self.edge_attr_dim = 2
            elif config.net_config.hparams.edge_transform == "localcartesian":
                self.edge_attr_transform = LocalCartesian()
                self.edge_attr_dim = 2
            else:
                raise IOError("net_config.hparams.edge_transform must be one of 'cartesian', 'localcartesian'")
        self.reduction_type = "linear"
        if hasattr(config.net_config.hparams, "reduction_type"):
            self.reduction_type = config.net_config.hparams.reduction_type
        self.graph_planes = [self.feat_size]
        self.n_contract = self.n_graph - self.n_expansion
        if self.reduction_type == "linear":
            if self.n_expansion > 0:
                exp = int((self.graph_planes[0]*self.expansion_factor - self.graph_planes[0]) / self.n_expansion)
                for n in range(self.n_expansion):
                    self.graph_planes.append(self.graph_planes[-1] + exp)
                if self.n_contract > 0:
                    red = int((self.graph_planes[-1] - self.graph_out) / self.n_contract)
                    for n in range(self.n_contract):
                        self.graph_planes.append(self.graph_planes[-1] - red)
            else:
                red = int((self.graph_planes[0] - self.graph_out) / self.n_graph)
                for n in range(self.n_graph):
                    self.graph_planes.append(self.graph_planes[-1] - red)
        elif self.reduction_type == "geometric":
            if self.n_expansion > 0:
                exp = float(self.expansion_factor) ** (1./self.n_expansion)
                for n in range(self.n_expansion):
                    self.graph_planes.append(int(self.graph_planes[-1] * exp))
                if self.n_contract > 0:
                    red = float(self.graph_out / self.graph_planes[-1]) ** (1. / self.n_contract)
                    for n in range(self.n_contract):
                        self.graph_planes.append(int(self.graph_planes[-1] * red))
            else:
                red = float(self.graph_out / self.graph_planes[0]) ** (1./self.n_graph)
                for n in range(self.n_graph):
                    self.graph_planes.append(int(self.graph_planes[-1] * red))
        else:
            raise IOError("net_config.hparams.reduction_type must be either linear or geometric")
        self.graph_planes[-1] = int(self.graph_out)
        for i in range(self.n_graph):
            nin = self.graph_planes[i]
            nout = self.graph_planes[i+1]
            self.graph_layers.append(PointConv(LinearPlanes([nin + self.ndim, nout], activation=ReLU()), LinearPlanes([nout, nout], activation=ReLU()), **self.graph_params))

    def forward(self, data):
        coo = data[0].long()
        edge_index = knn_graph(coo, self.k, coo[:, 2], loop=self.use_self_loops)
        geom_data = Data(x=data[1], edge_index=edge_index, pos=coo[:, 0:2])
        self.edge_attr_transform(geom_data)
        for layer in self.graph_layers:
            geom_data.x = layer(geom_data.x, geom_data.pos, geom_data.edge_index)
        if self.n_lin > 0:
            geom_data.x = global_max_pool(geom_data.x, coo[:, 2])
            geom_data.x = self.linear(geom_data.x)
        return geom_data.x
