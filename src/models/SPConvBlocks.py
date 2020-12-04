from math import floor

import spconv
import torch.nn as nn
from src.models.Algorithm import *
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL


class SparseConv2DForZ(nn.Module):
    def __init__(self, in_planes, kernel_size=3, n_layers=2):
        super(SparseConv2DForZ, self).__init__()
        layers = []
        increment = int(round(float(in_planes) / float(n_layers)))
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be an odd integer")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = in_planes
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = 1
            else:
                out -= increment
            pd = int((kernel_size - 1) / 2)
            layers.append(spconv.SparseConv2d(in_planes, out, kernel_size, 1, pd))
            if i != (n_layers - 1):
                layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            in_planes = out
            if kernel_size > 3:
                kernel_size -= 2
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def forward(self, x):
        return self.network(x)


class Pointwise2DForZ(nn.Module):
    def __init__(self, in_planes, pointwise_layers=2):
        super(Pointwise2DForZ, self).__init__()
        layers = []
        n_layers = pointwise_layers
        increment = int(round(float(in_planes) / float(n_layers)))
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = in_planes
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = 1
            else:
                out -= increment
            layers.append(spconv.SparseConv2d(in_planes, out, 1, 1, 0))
            layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            in_planes = out
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def forward(self, x):
        return self.network(x)

class SparseConv2DBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n, size, to_dense,
                 size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                 pointwise_factor=0, depth_factor=0, dropout=0, trainable_weights=False,
                 version=0):
        if version == 0:
            self._version0(nin, nout, n, size, to_dense,
                           size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor,
                           dil_factor=dil_factor,
                           pointwise_factor=pointwise_factor, depth_factor=depth_factor, dropout=dropout,
                           trainable_weights=trainable_weights)
        else:
            self._version1(nin, nout, n, size, to_dense,
                           size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor,
                           dil_factor=dil_factor,
                           pointwise_factor=pointwise_factor, depth_factor=depth_factor, dropout=dropout,
                           trainable_weights=trainable_weights)

    def _version0(self, nin, nout, n, size, to_dense,
                  size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                  pointwise_factor=0, depth_factor=0, dropout=0, trainable_weights=False):
        assert (n > 0)
        self.alg = []
        self.out_size = size
        self.dropout = dropout
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block with nin {0}, nout {1}, size {2}".format(nin, nout, size))
        self.ndim = len(size) - 1
        if nin != nout:
            if pointwise_factor > 0:
                nframes = [nin, nin - int(floor((nin - nout) * pointwise_factor))]
                if n > 1:
                    diff = float(nin - nout) / n
                    for i in range(n - 1):
                        val = int(floor(nframes[-1] - diff))
                        if val > nout:
                            nframes += [val]
                        else:
                            nframes += [nout]
            elif depth_factor > 0:
                nframes = [nin, int(nin * depth_factor)]
                if n > 1:
                    diff = float(nframes[-1] - nout) / (n - 1)
                    for i in range(n - 1):
                        val = int(floor(nframes[-1] - diff))
                        if val > nout:
                            nframes += [val]
                        else:
                            nframes += [nout]
            else:
                diff = float(nin - nout) / n
                nframes = [int(floor(nin - diff * i)) for i in range(n + 1)]
        else:
            nframes = [nin] * (n + 1)
        for i in range(n):
            decay_factor = i / (n + 1)
            fs = int(floor(size_factor / (i + 1.)))
            if fs < 3:
                fs = 3
            st = stride_factor - int(floor((stride_factor - 1) / (i + 1.)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * (fs - 1) * dil_factor) * decay_factor)
            if i == 0 and pointwise_factor > 0:
                pd, fs, dil, st = 0, 1, 1, 1
                self.alg.append(spconv.SparseConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                # self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added pointwise convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            else:
                self.alg.append(spconv.SparseConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added regular convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            self.log.debug("filter size: {0}, stride: {1}, pad: {2}, dil: {3}".format(fs, st, pd, dil))
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())
            if self.dropout:
                self.alg.append(nn.Dropout(self.dropout))
            arg_dict = {DIM: self.ndim, NIN: nframes[i], NOUT: nframes[i + 1], FS: [fs] * 4, STR: [st] * 4,
                        PAD: [pd] * 4, DIL: [dil] * 4}
            self.out_size = ModelValidation.calc_output_size(arg_dict, self.out_size, "cur", "prev", self.ndim)
            self.log.debug("Loop {0}, output size is {1}".format(i, self.out_size))

        if to_dense:
            self.alg.append(spconv.ToDense())
        self.func = spconv.SparseSequential(*self.alg)

    def _version1(self, nin, nout, n, size, to_dense,
                  size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                  pointwise_factor=0, depth_factor=0, dropout=0, trainable_weights=False):
        assert (n > 0)
        self.alg = []
        self.out_size = size
        self.dropout = dropout
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block with nin {0}, nout {1}, size {2}".format(nin, nout, size))
        self.ndim = len(size) - 1
        if nin != nout:
            if pointwise_factor > 0:
                nframes = [nin, nin - int(floor((nin - nout) * pointwise_factor))]
                if n > 1:
                    diff = float(nin - nout) / n
                    for i in range(n - 1):
                        val = int(floor(nframes[-1] - diff))
                        if val > nout:
                            nframes += [val]
                        else:
                            nframes += [nout]
            elif depth_factor > 0:
                nframes = [nin, int(nin * depth_factor)]
                if n > 1:
                    diff = float(nframes[-1] - nout) / (n - 1)
                    for i in range(n - 1):
                        val = int(floor(nframes[-1] - diff))
                        if val > nout:
                            nframes += [val]
                        else:
                            nframes += [nout]
            else:
                diff = float(nin - nout) / n
                nframes = [int(floor(nin - diff * i)) for i in range(n + 1)]
        else:
            nframes = [nin] * (n + 1)
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
            fs = int(floor(size_factor / (i + 1.)))
            if fs < 2:
                fs = 2
            st = int(round(stride_factor * i / (n - 1)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * (fs - 1) * dil_factor * decay_factor))
            if i == 0 and pointwise_factor > 0:
                pd, fs, dil, st = 0, 1, 1, 1
                self.alg.append(spconv.SparseConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                # self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added pointwise convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            else:
                self.alg.append(spconv.SparseConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added regular convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            self.log.debug("filter size: {0}, stride: {1}, pad: {2}, dil: {3}".format(fs, st, pd, dil))
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())
            if self.dropout:
                self.alg.append(nn.Dropout(self.dropout))
            arg_dict = {DIM: self.ndim, NIN: nframes[i], NOUT: nframes[i + 1], FS: [fs] * 4, STR: [st] * 4,
                        PAD: [pd] * 4, DIL: [dil] * 4}
            self.out_size = ModelValidation.calc_output_size(arg_dict, self.out_size, "cur", "prev", self.ndim)
            self.log.debug("Loop {0}, output size is {1}".format(i, self.out_size))

        if to_dense:
            self.alg.append(spconv.ToDense())
        self.func = spconv.SparseSequential(*self.alg)
