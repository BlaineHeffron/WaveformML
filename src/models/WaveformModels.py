from src.models.ConvBlocks import *
from src.models.RecurrentBlocks import RecurrentNet
from src.utils.util import DictionaryUtility
from torch import cat


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
        planes = [int(round(expand_factor * (i + 1))) for i in range(config.net_config.hparams.n_expand)]
        contract_factor = float((
                                            config.net_config.hparams.expansion_factor - config.net_config.hparams.out_planes) / config.net_config.hparams.n_contract)
        planes += [int(round(contract_factor * (config.net_config.hparams.n_contract - i - 1))) for i in
                   range(config.net_config.hparams.n_contract)]
        planes[-1] = config.net_config.hparams.out_planes
        if config.net_config.net_type == "TemporalConvolution":
            self.model = TemporalConvNet(1, planes,
                                         **DictionaryUtility.to_dict(config.net_config.hparams.conv_params))
        if config.net_config.hparams.n_lin > 0:
            self.linear = LinearBlock(self.flattened_size * planes[-1], self.output_size,
                                      config.net_config.hparams.n_lin).func
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
        planes = [self.nsamples]
        if hasattr(config.net_config.hparams, "n_expand"):
            if config.net_config.hparams.n_expand > 0:
                if not hasattr(config.net_config.hparams, "expansion_factor"):
                    raise IOError("config.net_config.hparams.expansion_factor must be set if n_expand > 0")
                expand_factor = float((planes[0] * config.net_config.hparams.expansion_factor - planes[
                    0]) / config.net_config.hparams.n_expand)
                planes += [int(round(planes[0] + expand_factor * (i + 1))) for i in
                           range(config.net_config.hparams.n_expand)]
            if not hasattr(config.net_config.hparams, "n_contract"):
                if hasattr(config.net_config.hparams, "n_lin"):
                    n_contract = config.net_config.hparams.n_lin - config.net_config.hparams.n_expand
                else:
                    raise IOError("if n_expand is set, must either set n_contract or n_lin")
            else:
                n_contract = config.net_config.hparams.n_contract
            contract_factor = float((planes[-1] - out_size) / n_contract)
            start_n = planes[-1]
            planes += [int(round(start_n - contract_factor * (i + 1))) for i in
                       range(config.net_config.hparams.n_contract)]
            planes[-1] = out_size
        if len(planes) == 1:
            if hasattr(config.net_config.hparams, "n_lin"):
                self.linear = LinearBlock(self.nsamples, out_size, config.net_config.hparams.n_lin)
            else:
                raise IOError("config.net_config.hparams.n_lin must be >= 1 if n_expand and n_contract not set")
        else:
            self.linear = LinearPlanes(planes)

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
            self.model = RecurrentNet(self.nsamples, 1, self.net_config.hparams.n_hidden,
                                      self.net_config.hparams.n_layers,
                                      self.net_config.hparams.n_lin, self.net_config.hparams.out_size,
                                      **DictionaryUtility.to_dict(self.net_config.hparams.rnn_params))
        else:
            raise IOError("{} not supported net type".format(config.net_config.net_type))

    def forward(self, x):
        x = self.model(x)
        return x


class ConvWaveformNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.system_config = config.system_config
        self.net_config = config.net_config
        self.nsamples = self.system_config.n_samples
        self.num_inputs = self.nsamples
        if hasattr(config.net_config, "use_detector_number"):
            self.use_detector_number = config.net_config.use_detector_number
            self.num_inputs -= 3
        else:
            self.use_detector_number = False
        if config.net_config.net_type == "CNN":
            self.model = Conv1DNet(self.num_inputs, **DictionaryUtility.to_dict(self.net_config.hparams.cnn_params))
        else:
            raise IOError("{} not supported net type".format(config.net_config.net_type))
        if hasattr(self.net_config.hparams, "n_lin"):
            if self.use_detector_number:
                self.linear = LinearBlock(self.model.out_size[0]*self.model.out_size[1] + 3,
                                      self.net_config.hparams.out_size, self.net_config.hparams.n_lin)
            else:
                self.linear = LinearBlock(self.model.out_size[0]*self.model.out_size[1],
                                          self.net_config.hparams.out_size, self.net_config.hparams.n_lin)

    def forward(self, x):
        det = None
        if self.use_detector_number:
            det = x[:, 0,  self.nsamples - 3:]
            x = self.model(x[:, :, 0:self.nsamples - 3])
        else:
            x = self.model(x)
        if hasattr(self, "linear"):
            x = self.flatten(x)
            if self.use_detector_number:
                x = cat((x, det), dim=1)
            x = self.linear(x)
        return x
