import logging
from math import floor, pow
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.models.Algorithm import *
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL


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


class LinearBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        self.log.debug("linear block called with args {}".format(args))
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n):
        assert(n > 0)
        assert(nin > 0)
        self.alg = []
        self.log = logging.getLogger(__name__)
        self.log.debug("Creating Linear block\n    nin: {0}\n   nout:{1}\n  n:{2}".format(nin, nout, n))
        factor = pow(float(nout) / nin, 1. / n)
        for i in range(n):
            self.alg.append(nn.Linear(int(round(nin * pow(factor, i))), int(round(nin * pow(factor, i + 1)))))
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

