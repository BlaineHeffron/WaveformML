from math import floor, pow, ceil
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm
from src.models.Algorithm import *
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL
from src.models.SPConvBlocks import _get_frame_expansion, _get_frame_contraction


class DilationBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        self.log.debug("dilation block called with args {}".format(args))
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n, length, size_factor=3, pad_factor=0, stride_factor=1, dil_factor=2.,
                 trainable_weights=False):
        self.out_length = length
        self.log = logging.getLogger(__name__)
        self.log.debug("output lenght is {}".format(self.out_length))
        self.alg = []
        if nin != nout:
            diff = float(nin - nout) / n
            nframes = [int(floor(nin - diff * i)) for i in range(n + 1)]
        else:
            nframes = [nin] * (n + 1)
        for i in range(n):
            fs = int(floor(size_factor / (i + 1.)))
            if fs < 3:
                fs = 3
            st = stride_factor - int(floor((stride_factor - 1) / (i + 1.)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(floor(pad_factor * (fs - 1) * dil_factor))
            self.alg.append(nn.Conv1d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
            arg_dict = {DIM: 2, NIN: nframes[i], NOUT: nframes[i + 1], FS: fs, STR: st, PAD: pd, DIL: dil}
            self.out_length = ModelValidation.calc_output_size_1d(self.out_length, arg_dict)
            self.log.debug("loop {0}, output length is {1}".format(i, self.out_length))
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())

        self.func = nn.Sequential(*self.alg)


class LinearPlanes(nn.Module):
    def __init__(self, planes, activation=None):
        super(LinearPlanes, self).__init__()
        self.log = logging.getLogger(__name__)
        alg = []
        for i in range(len(planes) - 1):
            alg.append(nn.Linear(int(round(planes[i])), int(round(planes[i + 1]))))
            if activation is not None:
                alg.append(activation)
            self.log.debug("Adding linear block {0} -> {1}\n".format(int(round(planes[i])), int(round(planes[i + 1]))))
        self.net = nn.Sequential(*alg)

    def forward(self, x):
        return self.net(x)


class LinearBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        self.log.debug("linear block called with args {}".format(args))
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n):
        assert (n > 0)
        assert (nin > 0)
        self.alg = []
        self.log = logging.getLogger(__name__)
        self.log.debug("Creating Linear block, nin: {0}, nout:{1}, n:{2}\n".format(nin, nout, n))
        factor = pow(float(nout) / nin, 1. / n)
        for i in range(n):
            self.alg.append(nn.Linear(int(round(nin * pow(factor, i))), int(round(nin * pow(factor, i + 1)))))
            self.log.debug("Adding linear block {0} -> {1}\n".format(int(round(nin * pow(factor, i))),
                                                                     int(round(nin * pow(factor, i + 1)))))
        self.func = nn.Sequential(*self.alg)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Implementation found athttps://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        if dropout != 0:
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                     self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.log = logging.getLogger(__name__)
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            self.log.debug(
                "Adding temporal block block {0} -> {1}, kernel {2}, dilation {3}\n".format(in_channels, out_channels,
                                                                                            kernel_size, dilation_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Conv1DNet(nn.Module):
    def __init__(self, length, num_channels, out_size, num_expand, num_contract, expand_factor, size_factor=3,
                 pad_factor=1, stride_factor=0, min_kernel=2):
        super(Conv1DNet, self).__init__()
        self.log = logging.getLogger(__name__)
        planes = [num_channels]
        conv_layers = []
        if num_expand > 0:
            expand = float((planes[0] * expand_factor - planes[0]) / num_expand)
            planes += [int(round(planes[0] + expand * (i + 1))) for i in range(num_expand)]
        contract_factor = float((planes[-1] - out_size) / num_contract)
        start_n = planes[-1]
        planes += [int(round(start_n - contract_factor * (i + 1))) for i in range(num_contract)]
        planes[-1] = out_size
        self.out_size = [length, num_channels]
        n = num_expand + num_contract
        for i in range(n):
            if n > 1:
                decay_factor = 1. - i / (n - 1)
                st = int(round(stride_factor * i / (n - 1)))
            else:
                decay_factor = 1.
                st = int(stride_factor)
            if st < 1:
                st = 1
            fs = int(ceil(size_factor * decay_factor))
            if fs < min_kernel:
                fs = min_kernel
            pd = int(round(pad_factor * ((fs - 1) / 2.) * decay_factor))
            self.log.debug("Initializing 1d convolution block for vector of length {0}: nin {1}, nout {2}, "
                           "kernel size {3}, padding {4}, stride {5}".format(self.out_size, planes[i], planes[i + 1],
                                                                             fs, pd, st))
            conv_layers.append(nn.Conv1d(planes[i], planes[i + 1], fs, stride=st, padding=pd))
            arg_dict = {DIM: 1, NIN: planes[i], NOUT: planes[i + 1], FS: [fs] * 4, STR: [st] * 4,
                        PAD: [pd] * 4, DIL: [1] * 4}
            self.out_size = ModelValidation.calc_output_size(arg_dict, self.out_size, "cur", "prev", 1)
        self.network = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.network(x)


class Conv2DBlock(nn.Module):
    def __init__(self, nin, nout, n, size,
                 size_factor=3, pad_factor=0., stride_factor=1.0,
                 dil_factor=1., expansion_factor=1., n_expansion=0,
                 pointwise_factor=0., dropout=None,
                 trainable_weights=False):
        super(Conv2DBlock, self).__init__()
        self.alg = []
        self.out_size = size
        self.dropout = dropout
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block with nin {0}, nout {1}, size {2}".format(nin, nout, size))
        self.ndim = len(size) - 1
        if pointwise_factor > 0:
            n_contraction = n - 1 - n_expansion
            if n_contraction < 1:
                raise ValueError("n_contraction too large, must be < n - 1")
        else:
            n_contraction = n - n_expansion
            if n_contraction < 1:
                raise ValueError("n_contraction too large, must be < n")
        nframes = [nin]
        if pointwise_factor > 0:
            nframes.append(nin - int(floor((nin - nout) * pointwise_factor)))
        if n_expansion > 0:
            nframes += _get_frame_expansion(nframes[-1], expansion_factor, n_expansion)
        if n_contraction > 0:
            nframes += _get_frame_contraction(nframes[-1], nout, n_contraction)
        for i in range(n):
            if pointwise_factor > 0:
                if n > 1:
                    decay_factor = 1. - (i - 1) / (n - 1)
                else:
                    decay_factor = 1.
            else:
                if n > 1:
                    decay_factor = 1. - i / (n - 1)
                else:
                    decay_factor = 1.
            fs = int(ceil(size_factor * decay_factor))
            if fs < 2:
                fs = 2
            st = int(round(stride_factor * i / (n - 1)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * ((fs - 1) / 2.) * dil_factor * decay_factor))
            if i == 0 and pointwise_factor > 0:
                pd, fs, dil, st = 0, 1, 1, 1
                self.alg.append(
                    Conv2d(nframes[i], nframes[i + 1], (fs, fs), (st, st), pd, (dil, dil), 1, trainable_weights))
                # self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added pointwise convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            else:
                self.alg.append(
                    Conv2d(nframes[i], nframes[i + 1], (fs, fs), (st, st), pd, (dil, dil), 1, trainable_weights))
                self.log.debug("added regular convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            self.log.debug("filter size: {0}, stride: {1}, pad: {2}, dil: {3}".format(fs, st, pd, dil))
            self.alg.append(nn.BatchNorm2d(nframes[i + 1]))
            self.alg.append(nn.ReLU())
            if self.dropout:
                self.alg.append(nn.Dropout(self.dropout))
            arg_dict = {DIM: self.ndim, NIN: nframes[i], NOUT: nframes[i + 1], FS: [fs] * 4, STR: [st] * 4,
                        PAD: [pd] * 4, DIL: [dil] * 4}
            self.out_size = ModelValidation.calc_output_size(arg_dict, self.out_size, "cur", "prev", self.ndim)
            self.log.debug("Loop {0}, output size is {1}".format(i, self.out_size))
        self.model = nn.Sequential(*self.alg)

    def forward(self, x):
        return self.model(x)
