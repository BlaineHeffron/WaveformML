from copy import copy
from typing import Any

import spconv
from torch import nn, LongTensor
from src.utils.util import DictionaryUtility
from numpy import array
from src.utils.util import *
from src.models.ConvBlocks import *
from src.models.SPConvBlocks import *


class SPConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.sequence_class = self.modules.retrieve_class(self.net_config.sequence_class)
        self.get_algorithm()
        if self.net_config.net_type == "2DConvolution":
            self.ndim = 2
        elif self.net_config.net_type == "3DConvolution":
            self.ndim = 3
        else:
            self.log.warning("Warning: unknown net_type in net_config: {}".format(self.net_config.net_type))
            self.ndim = 2
        if self.ndim == 3:
            self.spatial_size = array([14, 11, self.nsamples])
            self.permute_tensor = LongTensor([3, 0, 1, 2])  # needed because spconv requires batch index first
        else:
            self.spatial_size = array([14, 11])
            self.permute_tensor = LongTensor([2, 0, 1])  # needed because spconv requires batch index first

    def forward(self, x):
        #xlen = x[1].shape[0]
        if hasattr(self, "waveformLayer"):
            # pytorch expects 1d convolutions in with shape (N, Cin, Lin) where N is batch size, Cin is number of input feature planes, Lin is length of data
            #x[1] = x[1].reshape(xlen, 2, self.nsamples)
            x[1].unsqueeze_(1)
            x[1] = self.waveformLayer(x[1])
            x[1].squeeze_(1)
            #x[1] = x[1].reshape(xlen, self.waveformOutputLength)
        batch_size = x[0][-1, -1] + 1
        x = spconv.SparseConvTensor(x[1], x[0][:, self.permute_tensor], self.spatial_size, batch_size)
        x = self.sparseModel(x)
        #self.log.debug("output shape from sparse model : {}".format(x.shape))
        x = x.view(-1, self.n_linear)
        x = self.linear(x)
        return x

    def create_algorithm(self, hparams, n_classes):
        # TODO: get this working with 3d
        requirements = ["n_dil", "n_conv", "n_lin", "out_planes"]
        extras = ["wf_params", "conv_params", "lin_params"]
        size = [14, 11, int(self.nsamples * 2)]
        if hasattr(hparams, "n_conv"):
            for rq in requirements:
                if not hasattr(hparams, rq):
                    raise IOError(rq + " is required to create the sparse conv algorithm.")
            for p_name in extras:
                params = {} if not hasattr(hparams, p_name) else DictionaryUtility.to_dict(getattr(hparams, p_name))
                if p_name == "wf_params":
                    if hparams.n_dil > 0:
                        """
                        wfLayer = DilationBlock(2, 2, hparams.n_dil, int(size[2] / 2), **params)
                        self.waveformLayer = wfLayer.func
                        size[2] = int(wfLayer.out_length * 2)
                        self.waveformOutputLength = copy(size[2])
                        """
                        self.waveformLayer = TemporalConvNet(1, [1]*hparams.n_dil, **params)
                        size[2] = int(self.nsamples*2)
                        self.waveformOutputLength = int(self.nsamples*2)
                elif p_name == "conv_params":
                    spModel = SparseConv2DBlock(size[2], hparams.out_planes, hparams.n_conv, size, True, **params)
                    self.sparseModel = spModel.func
                    size = spModel.out_size
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

    def get_algorithm(self):
        sparse_funcs = []
        linear_funcs = []
        waveform_funcs = []
        has_wf = False
        if not hasattr(self.net_config, "algorithm"):
            if hasattr(self.net_config, "hparams"):
                try:
                    self.create_algorithm(self.net_config.hparams, self.ntype)
                except AssertionError as e:
                    print("Parameters {0} \nlead to error : {1}".format(self.net_config.hparams, e))
                    raise AssertionError
            else:
                raise IOError("net_config must contain one of either 'algorithm' or 'hparams'")
        else:

            for i, f in enumerate(self.net_config.algorithm):
                if i == 0:
                    if isinstance(f, str):
                        if f == "nn.Conv1d":
                            has_wf = True
                            waveform_funcs.append(f)
                            continue
                if has_wf:
                    if isinstance(f, str):
                        if f.startswith("nn."):
                            waveform_funcs.append(f)
                        else:
                            has_wf = False
                            sparse_funcs.append(f)
                    else:
                        waveform_funcs.append(f)
                    continue
                if isinstance(f, str):
                    if f == "nn.Linear":
                        linear_funcs = self.net_config.algorithm[i:]
                        break
                sparse_funcs.append(f)
            self.log.info("Sparse function list: {0}".format(str(sparse_funcs)))
            if len(waveform_funcs):
                self.log.info("Adding an initial waveform processing layer: {0}".format(str(waveform_funcs)))
                self.waveformLayer = nn.Sequential(*self.modules.create_class_instances(waveform_funcs))
                self.waveformOutputLength = sparse_funcs[1][0]
            self.sparseModel = self.sequence_class(*self.modules.create_class_instances(sparse_funcs))
            self.linear = nn.Sequential(*self.modules.create_class_instances(linear_funcs))
            self.n_linear = linear_funcs[1][0]
            self.log.debug("linear functions: {}".format(linear_funcs))
            self.log.debug("n_linear: {}".format(self.n_linear))
