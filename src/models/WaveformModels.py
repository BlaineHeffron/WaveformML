from src.models.ConvBlocks import *
from src.models.RecurrentBlocks import RecurrentNet
from src.utils.util import DictionaryUtility


class TemporalWaveformNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.flattened_size = self.nsamples
        if hasattr(config.net_config.hparams, "out_size"):
            self.output_size = config.net_config.hparams.out_size
        else:
            self.output_size = 1
        expand_factor = float(config.net_config.hparams.expansion_factor / config.net_config.hparams.n_expand)
        planes = [int(round(expand_factor*(i+1))) for i in range(config.net_config.hparams.n_expand)]
        contract_factor = float((config.net_config.hparams.expansion_factor - config.net_config.hparams.out_planes) / config.net_config.hparams.n_contract)
        planes += [int(round(contract_factor*(config.net_config.hparams.n_contract-i-1))) for i in range(config.net_config.hparams.n_contract)]
        planes[-1] = config.net_config.hparams.out_planes
        if config.net_config.net_type == "TemporalConvolution":
            self.model = TemporalConvNet(1, planes,
                                         **DictionaryUtility.to_dict(config.net_config.hparams.conv_params))
        if config.net_config.hparams.n_lin > 0:
            self.linear = LinearBlock(self.flattened_size*planes[-1], self.output_size, config.net_config.hparams.n_lin).func
            self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        if hasattr(self, "linear"):
            x = self.flatten(x)
            x = self.linear(x)
        return x


class LinearWaveformNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.flattened_size = self.nsamples
        if hasattr(self.net_config.hparams, "out_size"):
            out_size = self.net_config.hparams.out_size
        else:
            out_size = 1
        if config.net_config.hparams.n_lin > 0:
            self.linear = LinearBlock(self.flattened_size, out_size, config.net_config.hparams.n_lin).func
        else:
            raise IOError("config.net_config.hparams.n_lin must be >= 1")

    def forward(self, x):
        x = self.linear(x)
        return x


class RecurrentWaveformNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        if config.net_config.net_type == "RNN":
            self.model = RecurrentNet(self.nsamples, 1, self.net_config.hparams.n_hidden, self.net_config.hparams.n_layers,
                                      self.net_config.hparams.n_lin, self.net_config.hparams.out_size,
                                      **DictionaryUtility.to_dict(self.net_config.hparams.rnn_params))
        else:
            raise IOError("{} not supported net type".format(config.net_config.net_type))

    def forward(self, x):
        x = self.model(x)
        return x
