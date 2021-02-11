import logging

import torch
from torch import nn, LongTensor
import spconv
from numpy import array
from src.models.SPConvBlocks import SparseConv2DForEZ
from src.utils.util import ModuleUtility, get_config
from src.utils.util import DictionaryUtility
from src.engineering.LitZ import LitZ


class SingleEndedEZConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        if config.net_config.net_type != "2DConvolution":
            raise IOError("config.net_config.net_type must be 2DConvolution")
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.modules = ModuleUtility(self.net_config.imports)
        if hasattr(self.net_config, "z_weights"):
            self.use_z_model = True
            if not hasattr(self.net_config, "z_config"):
                raise ValueError("if specifying z_weights, you must also specify corresponding z_config")
            z_config = get_config(self.net_config.z_config)
            #setattr(z_config.net_config.hparams.conv, "todense", False)
            self.log.info("Using Z model from {}".format(self.net_config.z_weights))
            self.z_model = LitZ.load_from_checkpoint(self.net_config.z_weights, config=z_config)
            self.z_model.freeze()
        else:
            self.use_z_model = False
        if not hasattr(self.net_config, "algorithm"):
            setattr(self.net_config, "algorithm", "conv")
        if self.use_z_model:
            if self.net_config.algorithm == "conv":
                self.model = SparseConv2DForEZ(self.nsamples * 2, out_planes=1,
                                               **DictionaryUtility.to_dict(self.net_config.hparams))
            elif self.net_config.algorithm == "features":
                self.model = SparseConv2DForEZ(self.nsamples, out_planes=1,
                                               **DictionaryUtility.to_dict(self.net_config.hparams))
        else:
            if self.net_config.algorithm == "conv":
                self.model = SparseConv2DForEZ(self.nsamples * 2, **DictionaryUtility.to_dict(self.net_config.hparams))
            elif self.net_config.algorithm == "features":
                self.model = SparseConv2DForEZ(self.nsamples, **DictionaryUtility.to_dict(self.net_config.hparams))
        self.spatial_size = array([14, 11])
        self.permute_tensor = LongTensor([2, 0, 1])  # needed because spconv requires batch index first

    def forward(self, x):
        batch_size = x[0][-1, -1] + 1
        if self.use_z_model:
            z = self.z_model(x)
            x = spconv.SparseConvTensor(x[1], x[0][:, self.permute_tensor], self.spatial_size, batch_size)
            #x.features = torch.cat((x.features, z.features), dim=1)
            # new_features = torch.cat((x.features, z.features), dim=1)
            # x = spconv.SparseConvTensor(new_features, x[0][:, self.permute_tensor], self.spatial_size, batch_size)
            x = self.model(x)
            x = torch.cat((x, z), dim=1)
        else:
            x = spconv.SparseConvTensor(x[1], x[0][:, self.permute_tensor], self.spatial_size, batch_size)
            x = self.model(x)
        return x
