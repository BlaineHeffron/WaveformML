from torch import nn, LongTensor
from src.utils.util import *
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
        sparse_funcs = []
        linear_funcs = []
        for i, f in enumerate(self.net_config.algorithm):
            if isinstance(f, str):
                if f == "nn.Linear":
                    linear_funcs = self.net_config.algorithm[i:]
                    break
            sparse_funcs.append(f)
        print(sparse_funcs)
        self.sparseModel = self.sequence_class(*self.modules.create_class_instances(sparse_funcs))
        self.spatial_size = LongTensor([14, 11])
        self.inputLayer = scn.InputLayer(2, self.spatial_size, mode=0)
        self.linear = nn.Sequential(*self.modules.create_class_instances(linear_funcs))
        self.n_linear = linear_funcs[1][0]

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.n_linear)
        x = self.linear(x)
        return x
