import spconv
from torch import nn
from util import *

class SpConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.sequence_class = self.modules.retrieve_class(self.net_config.sequence_class)
        self.net = self.sequence_class(*self.modules.create_class_instances(self.net_config.algorithm))

    def forward(self, features, coors, batch_size):
        coors = coors.int()  # unlike torch, this library only accept int coordinates.
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()

