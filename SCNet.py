from torch import nn, LongTensor
from util import *
import sparseconvnet as scn


# two-dimensional SparseConvNet
class SCNet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.sequence_class = self.modules.retrieve_class(self.net_config.sequence_class)
        self.sparseModel = self.sequence_class(*self.modules.create_class_instances(self.net_config.algorithm))
        self.spatial_size = LongTensor([11, 14])
        self.inputLayer = scn.InputLayer(2, self.spatial_size, 0)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        # x = x.view(-1, self.spatial_size)
        # x = self.linear(x)
        return x
