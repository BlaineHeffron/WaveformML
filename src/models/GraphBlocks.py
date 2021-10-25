import logging
from torch import nn
from copy import copy
from torch.nn import ModuleList

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, GMMConv
from torch_geometric.transforms import Cartesian

from src.utils.GraphUtils import window_edges


class GraphZ(nn.Module):
    def __init__(self, in_planes, out_planes=1, neighbors=1, kernel=3, n_conv=1, n_point=3, conv_position=3,
                 pointwise_factor=0.8, batchnorm=True, self_loops=True):
        """
        @type out_planes: int
        """
        super(GraphZ, self).__init__()
        self.log = logging.getLogger(__name__)
        self.self_loops = self_loops
        n_layers = n_conv + n_point
        if n_conv > 0:
            if conv_position < 1:
                raise ValueError("conv position must be >= 1 if n_conv > 0")
        if n_point > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
            increment = int(round(int(round(in_planes * pointwise_factor - out_planes)) / float(n_layers - 1)))
        else:
            increment = int(round(float(in_planes-out_planes) / float(n_layers)))
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = copy(in_planes)
        inp = copy(in_planes)
        self.neighbors = []
        self.norms = ModuleList()
        self.nets = ModuleList()
        self.edge_attr_transform = Cartesian(max_value=neighbors)
        if n_conv > 0:
            conv_positions = [i for i in range(conv_position-1,conv_position-1+n_conv)]
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
                curr_neighbors = neighbors
            else:
                curr_neighbors = 0
            self.log.debug("appending layer {0} -> {1} planes, neighbor window of {2}".format(inp, out, curr_neighbors))
            if curr_neighbors == 0:
                self.nets.append(GMMConv(inp, out, 2, 1))
            else:
                self.nets.append(GMMConv(inp, out, 2, kernel))
            self.neighbors.append(curr_neighbors)
            if i != (n_layers - 1):
                if batchnorm:
                    self.norms.append(nn.BatchNorm1d(out))
            inp = out

    def forward(self, data: Data):
        batch = data.pos[:, 2]
        data.pos = data.pos[:, 0:2]
        for i, layer in enumerate(self.nets):
            if self.neighbors[i] == 0:
                data.edge_index = knn_graph(data.pos, self.neighbors[i], batch, loop=True)
            else:
                data.edge_index = window_edges(data.pos, batch, self.neighbors[i], self.self_loops)
            data.edge_attr = None
            self.edge_attr_transform(data)
            data.x = layer(data.x, data.edge_index, data.edge_attr)
            if i < len(self.norms):
                data.x = self.norms[i](data.x)
        return data.x


