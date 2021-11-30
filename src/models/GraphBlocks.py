import logging
from torch import nn
from copy import copy
from torch.nn import ModuleList
from torch import abs, sqrt, sum

from torch_geometric.data import Data
from torch_geometric.transforms import Cartesian
from torch_geometric.nn import EdgeConv, knn_graph, GCNConv, \
    MessagePassing, SAGEConv, GraphConv, GATConv, GATv2Conv, TransformerConv, TAGConv, GINConv, GINEConv, ARMAConv, \
    SGConv, GMMConv, EGConv, FeaStConv, LEConv, ClusterGCNConv, GENConv, HypergraphConv, GCN2Conv, PANConv, FiLMConv, \
    SuperGATConv, BatchNorm

from src.models.ConvBlocks import LinearPlanes
from src.utils.GraphUtils import window_edges



class GraphZ(nn.Module):
    def __init__(self, in_planes, out_planes=1, neighbors=1, kernel=3, n_conv=1, n_point=3, conv_position=3,
                 pointwise_factor=0.8, batchnorm=True, self_loops=True, graph_index=0):
        """
        @type out_planes: int
        """
        super(GraphZ, self).__init__()
        self.log = logging.getLogger(__name__)
        self.self_loops = self_loops
        self.use_edge_weights = self.check_edge_weights(graph_index)
        self.edge_attr = self.use_edge_attr(graph_index)
        self.graph_index = graph_index
        self.kernel = kernel
        n_layers = n_conv + n_point
        if n_conv > 0:
            if conv_position < 1:
                raise ValueError("conv position must be >= 1 if n_conv > 0")
        if n_point > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
            increment = int(round(int(round(in_planes * pointwise_factor - out_planes)) / float(n_layers - 1)))
        else:
            increment = int(round(float(in_planes - out_planes) / float(n_layers)))
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = copy(in_planes)
        inp = copy(in_planes)
        self.neighbors = []
        self.norms = ModuleList()
        self.nets = ModuleList()
        self.max_dist = neighbors
        self.edge_attr_transform = Cartesian(norm=False, max_value=neighbors)
        if n_conv > 0:
            conv_positions = [i for i in range(conv_position - 1, conv_position - 1 + n_conv)]
        else:
            conv_positions = []
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = copy(out_planes)
            else:
                out -= increment
                if i == 0 and n_point > 0:
                    if pointwise_factor > 0:
                        out = int(round(pointwise_factor * in_planes))
            if i in conv_positions:
                curr_neighbors = neighbors - int((i + 1 - conv_position))
                if curr_neighbors < 1:
                    curr_neighbors = 1
            else:
                curr_neighbors = 0
            self.log.debug("appending layer {0} -> {1} planes, neighbor window of {2}".format(inp, out, curr_neighbors))
            self.nets.append(self.choose_network(inp, out).jittable())
            self.neighbors.append(curr_neighbors)
            if i != (n_layers - 1):
                if batchnorm:
                    self.norms.append(nn.BatchNorm1d(out))
                    self.log.debug("appending batch norm 1d of size {}".format(out))
            inp = out

    def forward(self, data: Data):
        batch = data.pos[:, 2]
        data.pos = data.pos[:, 0:2]
        for i, layer in enumerate(self.nets):
            if self.neighbors[i] == 0:
                data.edge_index = knn_graph(data.pos, 1, batch, loop=True)
            else:
                data.edge_index = window_edges(data.pos, batch, self.neighbors[i], self.self_loops)
            data.edge_attr = None
            if self.use_edge_weights:
                self.edge_attr_transform(data)
                if self.edge_attr:
                    data.edge_attr = 1.0 - abs(data.edge_attr)/(self.max_dist + 1)
                    data.x = layer(data.x, data.edge_index, data.edge_attr)
                else:
                    data.edge_attr = 1. - sqrt(sum(data.edge_attr**2, dim=1)) / ((2*self.max_dist**2)**0.5)
                    data.x = layer(data.x, data.edge_index, data.edge_attr)
            else:
                data.x = layer(data.x, data.edge_index)
            if i < len(self.norms):
                data.x = self.norms[i](data.x)
        return data.x

    def check_edge_weights(self, ind):
        return ind in [0, 2, 5, 6, 8, 9, 10, 14]

    def use_edge_attr(self, ind):
        return ind in [3, 5, 10]

    def choose_network(self, inp, out):
        if self.graph_index == 0:
            return GCNConv(inp, out)
        elif self.graph_index == 1:
            return SAGEConv(inp, out)
        elif self.graph_index == 2:
            return GraphConv(inp, out)
        elif self.graph_index == 3:
            return GATConv(inp, out, add_self_loops=False)
        elif self.graph_index == 4:
            return GATv2Conv(inp, out)
        elif self.graph_index == 5:
            return TransformerConv(inp, out, edge_dim=2)
        elif self.graph_index == 6:
            return TAGConv(inp, out)
        elif self.graph_index == 7:
            return GINConv(LinearPlanes([inp, out], activation=nn.ReLU()))
        elif self.graph_index == 8:
            return ARMAConv(inp, out)
        elif self.graph_index == 9:
            return SGConv(inp, out)
        elif self.graph_index == 10:
            return GMMConv(inp, out, 2, self.kernel)
        elif self.graph_index == 11:
            return FiLMConv(inp, out)
        elif self.graph_index == 12:
            return EdgeConv(LinearPlanes([inp*2, out], activation=nn.ReLU()))
        elif self.graph_index == 13:
            return FeaStConv(inp, out)
        elif self.graph_index == 14:
            return LEConv(inp, out)
        elif self.graph_index == 15:
            return ClusterGCNConv(inp, out)
        elif self.graph_index == 16:
            return GENConv(inp, out)
        elif self.graph_index == 17:
            return SuperGATConv(inp, out)
