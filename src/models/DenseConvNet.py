from copy import copy

from torch import nn
import logging
from torch import sparse_coo_tensor

from src.models.ConvBlocks import Conv2DBlock, LinearBlock
from src.utils.util import ModuleUtility, DictionaryUtility


class DenseConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.sequence_class = self.modules.retrieve_class(self.net_config.sequence_class)
        self.x = 14
        self.y = 11
        self.get_algorithm()

    def forward(self, x):
        batch_size = x[0][-1, -1] + 1
        spacial_size = [batch_size, self.x, self.y]
        sparse_tensor = sparse_coo_tensor(x[0], x[1], size=spacial_size)
        x = self.model(sparse_tensor.to_dense())
        x = x.view(-1, self.n_linear)
        x = self.linear(x)
        return x

    def get_algorithm(self):
        sparse_funcs = []
        linear_funcs = []
        if hasattr(self.net_config, "hparams"):
            try:
                self.create_algorithm(self.net_config.hparams, self.ntype)
            except AssertionError as e:
                raise AssertionError("Parameters {0} \nlead to error : {1}".format(DictionaryUtility.to_dict(self.net_config.hparams), e))
        else:
            raise IOError("net_config must contain one of either 'algorithm' or 'hparams'")

    def create_algorithm(self, hparams, n_classes):
        # TODO: get this working with 3d
        requirements = ["n_conv", "n_lin", "out_planes"]
        extras = ["conv_params", "lin_params"]
        size = [14, 11, int(self.nsamples * 2)]
        if hasattr(hparams, "n_conv"):
            for rq in requirements:
                if not hasattr(hparams, rq):
                    raise IOError(rq + " is required to create the conv algorithm.")
            for p_name in extras:
                params = {} if not hasattr(hparams, p_name) else DictionaryUtility.to_dict(getattr(hparams, p_name))
                if p_name == "conv_params":
                    self.model = Conv2DBlock(size[2], hparams.out_planes, hparams.n_conv, size, True, **params)
                    size = self.model.out_size
                elif p_name == "lin_params":
                    flat_size = 1
                    for s in size:
                        flat_size = flat_size * s
                    self.n_linear = copy(flat_size)
                    self.log.debug("Flattened size of the SCN network output is {}".format(flat_size))
                    self.linear = LinearBlock(flat_size, n_classes, hparams.n_lin).func
            self.log.debug("n_linear: {}".format(self.n_linear))
        else:
            raise IOError("hparams must be a dictionary containing the following minimal settings:\n"
                          "  n_dil: int, number of dilation waveform layers\n"
                          "  n_conv: int, number of n-d sparse convolutional layers\n"
                          "  n_lin: int, number of linear layers")
