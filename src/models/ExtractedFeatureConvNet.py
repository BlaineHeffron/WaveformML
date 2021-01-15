from copy import copy
import logging
from torch import nn, LongTensor
import spconv
from numpy import array

from src.models.ConvBlocks import LinearBlock
from src.models.SPConvBlocks import ExtractedFeatureConv
from src.utils.util import ModuleUtility
from src.utils.util import DictionaryUtility


class ExtractedFeatureConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        if config.net_config.net_type != "2DConvolution":
            raise IOError("config.net_config.net_type must be 2DConvolution")
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nfeatures = self.system_config.n_features
        self.modules = ModuleUtility(self.net_config.imports)
        self.spatial_size = array([14, 11])
        self.size = [14,11,self.system_config.n_features]
        self.model = ExtractedFeatureConv(self.nfeatures, self.net_config.hparams.out_planes,
                                          self.net_config.hparams.n_conv, size,
                                          **DictionaryUtility.to_dict(self.net_config.hparams.conv))
        hparams = self.net_config.hparams
        flat_size = 1
        for s in self.model.out_size:
            flat_size *= s
        self.n_linear = copy(flat_size)
        self.log.debug("Flattened size of the SCN network output is {}".format(flat_size))
        self.linear = LinearBlock(flat_size, self.system_config.n_type, hparams.n_lin).func
        self.permute_tensor = LongTensor([2, 0, 1])  # needed because spconv requires batch index first

    def forward(self, x):
        batch_size = x[0][-1, -1] + 1
        x = spconv.SparseConvTensor(x[1], x[0][:, self.permute_tensor], self.spatial_size, batch_size)
        x = self.sparseModel(x)
        x = x.view(-1, self.n_linear)
        x = self.linear(x)
        return x
