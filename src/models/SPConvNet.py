import spconv
from torch import nn, LongTensor
from src.utils.util import *

class SpConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.ntype = self.system_config.n_type
        self.modules = ModuleUtility(self.net_config.imports)
        self.sequence_class = self.modules.retrieve_class(self.net_config.sequence_class)
        sparse_funcs = []
        linear_funcs = []
        waveform_funcs = []
        has_wf = False
        # TODO: try running 1d dilated convolution on x[1] (feature vectors) before running it through input layer
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
        self.sparseModel = self.sequence_class(*self.modules.create_class_instances(sparse_funcs))
        self.linear = nn.Sequential(*self.modules.create_class_instances(linear_funcs))
        self.n_linear = linear_funcs[1][0]
        self.spatial_size = LongTensor([14, 11])
        self.permute_tensor = LongTensor([2,0,1])

    def forward(self, x):
        batch_size = x[0][-1,2] + 1
        x = spconv.SparseConvTensor(x[1], x[0][:,self.permute_tensor], self.spatial_size, batch_size)
        x = self.sparseModel(x)
        x = x.view(-1, self.n_linear)
        x = self.linear(x)
        return x
