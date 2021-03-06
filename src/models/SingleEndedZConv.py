import logging
from torch import nn, LongTensor
import spconv
from numpy import array
from src.models.SPConvBlocks import SparseConv2DForZ, Pointwise2DForZ, SparseConv2DForEZ
from src.utils.util import ModuleUtility
from src.utils.util import DictionaryUtility


class SingleEndedZConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        if config.net_config.net_type != "2DConvolution":
            raise IOError("config.net_config.net_type must be 2DConvolution")
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.modules = ModuleUtility(self.net_config.imports)
        if not hasattr(self.net_config, "algorithm"):
            setattr(self.net_config, "algorithm", "conv")
        if hasattr(self.net_config, "version"):
            self.version = self.net_config.version
        else:
            self.version = 0
        if self.net_config.algorithm == "conv":
            if self.version == 0:
                self.model = SparseConv2DForZ(self.nsamples*2, **DictionaryUtility.to_dict(self.net_config.hparams.conv))
            else:
                self.model = SparseConv2DForEZ(self.nsamples*2, out_planes=1, **DictionaryUtility.to_dict(self.net_config.hparams))
        elif self.net_config.algorithm == "point":
            self.model = Pointwise2DForZ(self.nsamples*2, **DictionaryUtility.to_dict(self.net_config.hparams.point))
        elif self.net_config.algorithm == "features":
            if self.version == 0:
                self.model = SparseConv2DForZ(self.nsamples, **DictionaryUtility.to_dict(self.net_config.hparams.conv))
            else:
                self.model = SparseConv2DForEZ(self.nsamples, out_planes=1, **DictionaryUtility.to_dict(self.net_config.hparams))
        self.spatial_size = array([14, 11])
        self.permute_tensor = LongTensor([2, 0, 1])  # needed because spconv requires batch index first

    def forward(self, x):
        batch_size = x[0][-1, -1] + 1
        x = spconv.SparseConvTensor(x[1], x[0][:, self.permute_tensor], self.spatial_size, batch_size)
        x = self.model(x)
        return x
