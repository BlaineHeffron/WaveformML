import spconv
from torch import nn, LongTensor
from util import *
import sparseconvnet as scn


class SpConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.net_class = self.modules.retrieve_class(self.net_config.net_class)
        self.net = self.net_class(*self.modules.create_class_instances(self.net_config.algorithm))

    def forward(self, features, coors, batch_size):
        coors = coors.int()  # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


# two-dimensional SparseConvNet
class SCNet(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.net_class = self.modules.retrieve_class(self.net_config.net_class)
        self.sparseModel = self.net_class(*self.modules.create_class_instances(self.net_config.algorithm))
        self.spatial_size = self.sparseModel.input_spatial_size(LongTensor([11, 14]))
        self.inputLayer = scn.InputLayer(2, self.spatial_size, 0)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        #x = x.view(-1, 64)
        #x = self.linear(x)
        return x

