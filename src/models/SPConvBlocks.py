from math import floor, ceil
from copy import copy

import spconv
import torch.nn as nn
from src.models.Algorithm import *
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL

class SparseConv2DForEZ(nn.Module):
    def __init__(self, in_planes, out_planes=2, kernel_size=3, n_conv=1, n_point=3, conv_position=3,
                 pointwise_factor=0.8, batchnorm=True, version=0, n_expand=0):
        """
        @type out_planes: int
        """
        super(SparseConv2DForEZ, self).__init__()
        self.log = logging.getLogger(__name__)
        if version == 0:
            self._version0(in_planes, out_planes, kernel_size, n_conv, n_point, conv_position,pointwise_factor, batchnorm)
        elif version == 1:
            self._version1(in_planes, out_planes, kernel_size, n_conv, n_point, conv_position,pointwise_factor, batchnorm)
        elif version == 2:
            self._version2(in_planes, out_planes, kernel_size, n_conv, n_point, conv_position, pointwise_factor,
                           batchnorm)
        elif version == 3:
            self._version3(in_planes, out_planes, kernel_size, n_conv, n_point, conv_position, pointwise_factor,
                           n_expand, batchnorm)

        else:
            raise IOError("no version {} available, choose a version <= 1".format(version))

    def _version0(self, in_planes, out_planes=2, kernel_size=3, n_conv=1, n_point=3, conv_position=3,
                 pointwise_factor=0.8, batchnorm=True):
        layers = []
        n_layers = n_conv + n_point
        if n_conv > 0:
            if conv_position < 1:
                raise ValueError("conv position must be >= 1 if n_conv > 0")
        if n_point > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
            increment = int(round(int(round(in_planes * pointwise_factor - out_planes)) / float(n_layers - 1)))
        else:
            increment = int(round(float(in_planes-out_planes) / float(n_layers)))
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be an odd integer")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = copy(in_planes)
        inp = copy(in_planes)
        curr_kernel = copy(kernel_size)
        if n_conv > 0:
            conv_positions = [i for i in range(conv_position-1,conv_position-1+n_conv)]
        else:
            conv_positions = []
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = copy(out_planes)
            else:
                out -= increment
                if i == 0 and n_point > 0:
                    if pointwise_factor > 0:
                        out = int(round(pointwise_factor * in_planes))
            if not i in conv_positions:
                curr_kernel = 1
            else:
                curr_kernel = kernel_size - int((i+1 - conv_position)*2)
                if curr_kernel < 3:
                    curr_kernel = 3
            if curr_kernel % 2 == 0:
                raise ValueError("error: kernel size is even")
            pd = int((curr_kernel - 1) / 2)
            self.log.debug(
                "appending layer {0} -> {1} planes, kernel size of {2}, padding of {3}".format(inp, out,
                                                                                               curr_kernel, pd))
            layers.append(spconv.SparseConv2d(inp, out, curr_kernel, 1, pd))
            if i != (n_layers - 1):
                if batchnorm:
                    layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            inp = out
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def _version1(self, in_planes, out_planes=2, kernel_size=3, n_conv=1, n_point=3, conv_position=3,
                  pointwise_factor=0.8, batchnorm=True):
        layers = []
        n_layers = n_conv + n_point
        if n_conv > 0:
            if conv_position < 1:
                raise ValueError("conv position must be >= 1 if n_conv > 0")
        if n_point > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
            increment = int(round(int(round(in_planes * pointwise_factor - out_planes)) / float(n_layers - 1)))
        else:
            increment = int(round(float(in_planes-out_planes) / float(n_layers)))
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be an odd integer")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = copy(in_planes)
        inp = copy(in_planes)
        curr_kernel = copy(kernel_size)
        if n_conv > 0:
            conv_positions = [i for i in range(conv_position-1,conv_position-1+n_conv)]
        else:
            conv_positions = []
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = copy(out_planes)
            else:
                out -= increment
                if i == 0 and n_point > 0:
                    if pointwise_factor > 0:
                        out = int(round(pointwise_factor * in_planes))
            if not i in conv_positions:
                curr_kernel = 1
            else:
                curr_kernel = kernel_size - int((i+1 - conv_position)*2)
                if curr_kernel < 3:
                    curr_kernel = 3
            if curr_kernel % 2 == 0:
                raise ValueError("error: kernel size is even")
            pd = int((curr_kernel - 1) / 2)
            self.log.debug(
                "appending layer {0} -> {1} planes, kernel size of {2}, padding of {3}".format(inp, out,
                                                                                               curr_kernel, pd))
            if curr_kernel < 4:
                indkey = "subm0"
            else:
                indkey = "subm{}".format(curr_kernel)
            if out <= 0:
                out = 1
            layers.append(spconv.SubMConv2d(inp, out, curr_kernel, 1, pd, indice_key=indkey))
            if i != (n_layers - 1):
                if batchnorm:
                    layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            inp = out
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def _version2(self, in_planes, out_planes=2, kernel_size=3, n_conv=1, n_point=3, conv_position=3,
                  pointwise_factor=0.8, batchnorm=True):
        layers = []
        n_layers = n_conv + n_point
        if n_conv > 0:
            if conv_position < 1:
                raise ValueError("conv position must be >= 1 if n_conv > 0")
        if n_point > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
            increment = int(round(int(round(in_planes * pointwise_factor - out_planes)) / float(n_layers - 1)))
        else:
            increment = int(round(float(in_planes-out_planes) / float(n_layers)))
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be an odd integer")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = copy(in_planes)
        inp = copy(in_planes)
        curr_kernel = copy(kernel_size)
        if n_conv > 0:
            conv_positions = [i for i in range(conv_position-1,conv_position-1+n_conv)]
        else:
            conv_positions = []
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = copy(out_planes)
            else:
                out -= increment
                if i == 0 and n_point > 0:
                    if pointwise_factor > 0:
                        out = int(round(pointwise_factor * in_planes))
            if not i in conv_positions:
                curr_kernel = 1
            else:
                curr_kernel = kernel_size
                if curr_kernel < 3:
                    curr_kernel = 3
            if curr_kernel % 2 == 0:
                raise ValueError("error: kernel size is even")
            pd = int((curr_kernel - 1) / 2)
            self.log.debug(
                "appending layer {0} -> {1} planes, kernel size of {2}, padding of {3}".format(inp, out,
                                                                                               curr_kernel, pd))
            if curr_kernel < 4:
                indkey = "subm0"
            else:
                indkey = "subm{}".format(curr_kernel)
            layers.append(spconv.SubMConv2d(inp, out, curr_kernel, 1, pd, indice_key=indkey))
            if i != (n_layers - 1):
                if batchnorm:
                    layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            inp = out
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def _version3(self, in_planes, out_planes=2, kernel_size=3, n_conv=1, n_point=3, conv_position=3,
                  expansion_factor=2.0, n_expand=3, batchnorm=True):
        layers = []
        n_layers = n_conv + n_point
        if n_conv > 0:
            if conv_position < 1:
                raise ValueError("conv position must be >= 1 if n_conv > 0")
        if n_point > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
        nframes = [in_planes]
        n_contraction = n_layers - n_expand
        if n_contraction < 1:
            raise ValueError("n expand must be <= (n_point + n_conv - 1)")
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be an odd integer")
        if not isinstance(n_conv, int) or n_conv < 1:
            raise ValueError("n_conv must be an integer >= 1 ")
        if n_expand > 0:
            nframes += _get_frame_expansion(nframes[-1],expansion_factor,n_expand, True)
        if n_contraction > 0:
            nframes += _get_frame_contraction(nframes[-1], out_planes, n_contraction, True)
        nframes[-1] = out_planes
        if n_conv > 0:
            conv_positions = [i for i in range(conv_position-1,conv_position-1+n_conv)]
        else:
            conv_positions = []
        for i in range(n_layers):
            if not i in conv_positions:
                fs = 1
                pd = 1
            else:
                if n_conv > 1:
                    decay_factor = 1. - conv_positions.index(i) / (n_conv - 1)
                else:
                    decay_factor = 1.
                fs = int(ceil(kernel_size * decay_factor))
                if fs % 2 == 0:
                    fs -= 1
                if fs < 3:
                    fs = 3
                pd = int((fs - 1) / 2)
            self.log.debug(
                "appending layer {0} -> {1} planes, kernel size of {2}, padding of {3}".format(nframes[i], nframes[i+1],
                                                                                               fs, pd))
            if fs < 4:
                indkey = "subm0"
            else:
                indkey = "subm{}".format(fs)
            layers.append(spconv.SubMConv2d(nframes[i], nframes[i+1], fs, 1, pd, indice_key=indkey))
            if i != (n_layers - 1):
                if batchnorm:
                    layers.append(nn.BatchNorm1d(nframes[i+1]))
            layers.append(nn.ReLU())
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def forward(self, x):
        return self.network(x)


class SparseConv2DForZ(nn.Module):
    def __init__(self, in_planes, kernel_size=3, n_layers=2, pointwise_layers=0, pointwise_factor=0.8,
                 todense=True):
        super(SparseConv2DForZ, self).__init__()
        self.log = logging.getLogger(__name__)
        layers = []
        if pointwise_layers > 0:
            if n_layers == 1:
                raise ValueError("n_layers must be > 1 if using pointwise convolution")
            increment = int(round(int(round(in_planes * pointwise_factor)) / float(n_layers - 1)))
        else:
            increment = int(round(float(in_planes) / float(n_layers)))
        if kernel_size % 2 != 1:
            raise ValueError("Kernel size must be an odd integer")
        if not isinstance(n_layers, int) or n_layers < 1:
            raise ValueError("n_layers must be  integer >= 1")
        out = in_planes
        reset_kernel = False
        orig_kernel = kernel_size
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = 1
            else:
                out -= increment
                if i == 0 and pointwise_layers > 0:
                    if pointwise_factor > 0:
                        out = int(round(pointwise_factor * in_planes))
            pd = int((kernel_size - 1) / 2)
            if pointwise_layers > 0:
                pd = 0
                kernel_size = 1
                pointwise_layers -= 1
                if pointwise_layers == 0:
                    reset_kernel = True
            self.log.debug(
                "appending layer {0} -> {1} planes, kernel size of {2}, padding of {3}".format(in_planes, out,
                                                                                               kernel_size, pd))
            layers.append(spconv.SparseConv2d(in_planes, out, kernel_size, 1, pd))
            if reset_kernel:
                kernel_size = orig_kernel
                reset_kernel = False
            if i != (n_layers - 1):
                layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            in_planes = out
            if kernel_size > 1:
                kernel_size -= 2
        if todense:
            layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def forward(self, x):
        return self.network(x)


class Pointwise2DForZ(nn.Module):
    def __init__(self, in_planes, pointwise_layers=2):
        super(Pointwise2DForZ, self).__init__()
        self.log = logging.getLogger(__name__)
        layers = []
        n_layers = pointwise_layers
        if not isinstance(n_layers, int) or n_layers < 2:
            raise ValueError("n_layers must be  integer >= 2")
        increment = int(round(float(in_planes) / float(n_layers - 1)))
        out = in_planes
        for i in range(n_layers):
            if i == (n_layers - 1):
                out = 1
            elif i == 0:
                out = in_planes
            else:
                out -= increment
            self.log.debug(
                "appending layer {0} -> {1} planes, kernel size of {2}, padding of {3}".format(in_planes, out, 1, 0))
            layers.append(spconv.SparseConv2d(in_planes, out, 1, 1, 0))
            layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU())
            in_planes = out
        layers.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExtractedFeatureConv(nn.Module):
    def __init__(self, nin, nout, n, size, expansion_factor=10., size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                 dropout=0, trainable_weights=False):
        super(ExtractedFeatureConv, self).__init__()
        assert (n > 1)
        self.alg = []
        self.out_size = size
        self.dropout = dropout
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block with nin {0}, nout {1}, size {2}".format(nin, nout, size))
        self.ndim = len(size) - 1
        nframes = [nin, int(round(nin*expansion_factor))]
        diff = float(nframes[1] - nout) / (n - 1)
        nframes += [int(floor(nframes[1] - diff * i)) for i in range(n - 1)]
        for i in range(n):
            decay_factor = 1. - (i - 1) / (n - 1)
            fs = int(floor(size_factor / (i + 1.)))
            if fs < 2:
                fs = 2
            st = int(round(stride_factor * i / (n - 1)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * (fs - 1) * dil_factor * decay_factor))
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

        self.alg.append(spconv.ToDense())
        self.network = spconv.SparseSequential(*self.alg)

    def forward(self, x):
        return self.network(x)


def _get_frame_expansion(initial_number, factor, n, use_round=False):
    frames = [initial_number]
    diff = float(int(round(factor*initial_number)) - initial_number) / n
    for i in range(n):
        if use_round:
            frames += [int(round(frames[-1] + diff))]
        else:
            frames += [int(floor(frames[-1] + diff))]
    return frames[1:]


def _get_frame_contraction(initial_number, nout, n, use_round=False):
    frames = [initial_number]
    diff = float(initial_number - nout) / n
    for i in range(n):
        if use_round:
            frames += [int(round(frames[-1] - diff))]
        else:
            frames += [int(floor(frames[-1] - diff))]
    return frames[1:]


class SparseConv2DBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n, size, to_dense,
                 size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                 pointwise_factor=0, depth_factor=0, dropout=0, trainable_weights=False,
                 version=0, expansion_factor=0, n_expansion=0):
        if version == 0:
            self._version0(nin, nout, n, size, to_dense,
                           size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor,
                           dil_factor=dil_factor,
                           pointwise_factor=pointwise_factor, depth_factor=depth_factor, dropout=dropout,
                           trainable_weights=trainable_weights)
        elif version == 1:
            self._version1(nin, nout, n, size, to_dense,
                           size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor,
                           dil_factor=dil_factor,
                           pointwise_factor=pointwise_factor, depth_factor=depth_factor, dropout=dropout,
                           trainable_weights=trainable_weights)

        elif version == 2:
            self._version2(nin, nout, n, size, to_dense,
                           size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor,
                           dil_factor=dil_factor, expansion_factor=expansion_factor, n_expansion=n_expansion,
                           pointwise_factor=pointwise_factor, dropout=dropout,
                           trainable_weights=trainable_weights)

        elif version == 3:
            self._version3(nin, nout, n, size, to_dense,
                           size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor,
                           dil_factor=dil_factor, expansion_factor=expansion_factor, n_expansion=n_expansion,
                           pointwise_factor=pointwise_factor, dropout=dropout,
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


    def _version2(self, nin, nout, n, size, to_dense,
                  size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                  expansion_factor=0, n_expansion=0,
                  pointwise_factor=0, dropout=0, trainable_weights=False):
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
            nframes += _get_frame_expansion(nframes[-1],expansion_factor,n_expansion)
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
            fs = int(floor(size_factor / (i + 1.)))
            if fs < 2:
                fs = 2
            st = int(round(stride_factor * i / (n - 1)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * ((fs - 1)/2.) * dil_factor * decay_factor))
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

    def _version3(self, nin, nout, n, size, to_dense,
                  size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                  expansion_factor=0, n_expansion=0,
                  pointwise_factor=0, dropout=0, trainable_weights=False):
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
            nframes += _get_frame_expansion(nframes[-1],expansion_factor,n_expansion)
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
            fs = int(ceil(size_factor*decay_factor))
            if fs < 2:
                fs = 2
            st = int(round(stride_factor * i / (n - 1)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * ((fs - 1)/2.) * dil_factor * decay_factor))
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


class SparseConv2DPreserve(nn.Module):

    def __init__(self, nin, nout, n,
                 size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
                 pointwise_factor=0, dropout=0, trainable_weights=False,
                 expansion_factor=0, n_expansion=0, version=0, n_contraction=1):
        super(SparseConv2DPreserve, self).__init__()
        if version == 0:
            self._version0(nin, nout, n,
                 size_factor=size_factor, pad_factor=pad_factor, stride_factor=stride_factor, dil_factor=dil_factor,
                 pointwise_factor=pointwise_factor, dropout=dropout, trainable_weights=trainable_weights,
                 expansion_factor=expansion_factor, n_expansion=n_expansion)
        elif version == 1:
            self._version1(nin, nout, n_contraction=n_contraction,
                           size_factor=size_factor,
                           pointwise_factor=pointwise_factor, dropout=dropout, trainable_weights=trainable_weights,
                           expansion_factor=expansion_factor, n_expansion=n_expansion)
        else:
            raise ValueError("no version {} available".format(version))

    def _version0(self,nin, nout, n,
             size_factor=3, pad_factor=0.0, stride_factor=1, dil_factor=1,
             pointwise_factor=0, dropout=0, trainable_weights=False,
             expansion_factor=0, n_expansion=0):
        self.alg = []
        #self.out_size = size
        self.dropout = dropout
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block preserving input size with nin {0}, nout {1}".format(nin, nout))
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
            nframes += _get_frame_expansion(nframes[-1],expansion_factor,n_expansion)
        if n_contraction > 0:
            nframes += _get_frame_contraction(nframes[-1], nout, n_contraction)
        nframes[-1] = nout
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
            fs = int(ceil(size_factor*decay_factor))
            if fs < 2:
                fs = 2
            st = int(round(stride_factor * i / (n - 1)))
            if st < 1:
                st = 1
            dil = int(round(dil_factor ** i))
            pd = int(round(pad_factor * ((fs - 1)/2.) * dil_factor * decay_factor))
            if i == 0 and pointwise_factor > 0:
                pd, fs, dil, st = 0, 1, 1, 1
                self.alg.append(spconv.SparseConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, bias=trainable_weights, indice_key='ind_{}'.format(i)))
                self.alg.append(spconv.SparseInverseConv2d(nframes[i + 1], nframes[i + 1], fs, 'ind_{}'.format(i), bias=trainable_weights))
                # self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added pointwise convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
                self.log.debug("added inverse pointwise convolution, frames: {0} -> {1}".format(nframes[i + 1], nframes[i + 1]))
            else:
                self.alg.append(spconv.SparseConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights, indice_key='ind_{}'.format(i)))
                self.alg.append(spconv.SparseInverseConv2d(nframes[i + 1], nframes[i + 1], fs, 'ind_{}'.format(i), bias=trainable_weights))
                self.log.debug("added regular convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
                self.log.debug("added inverse convolution, frames: {0} -> {1}".format(nframes[i + 1], nframes[i + 1]))
            self.log.debug("filter size: {0}, stride: {1}, pad: {2}, dil: {3}".format(fs, st, pd, dil))
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())
            if self.dropout:
                self.alg.append(nn.Dropout(self.dropout))
            #arg_dict = {DIM: self.ndim, NIN: nframes[i], NOUT: nframes[i + 1], FS: [fs] * 4, STR: [st] * 4,
            #            PAD: [pd] * 4, DIL: [dil] * 4}
            #self.out_size = ModelValidation.calc_output_size(arg_dict, self.out_size, "cur", "prev", self.ndim)
            #self.log.debug("Loop {0}, output size is {1}".format(i, self.out_size))
        self.func = spconv.SparseSequential(*self.alg)

    def _version1(self, nin, nout, n_contraction=1,
            size_factor=3,
            pointwise_factor=0, dropout=0, trainable_weights=False,
            expansion_factor=0, n_expansion=0):

        n = n_contraction + n_expansion
        if pointwise_factor > 0:
            n_expansion -= 1
        if n < 1:
            raise ValueError("n_contraction + n_expansion must be >=1")
        self.alg = []
        self.dropout = dropout
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block preserving input size with nin {0}, nout {1}".format(nin, nout))
        if size_factor % 2 != 1:
            raise ValueError("size factor must be odd if version == 1")
        nframes = [nin]
        if pointwise_factor > 0:
            nframes.append(int(nin*pointwise_factor))
        if n_expansion > 0:
            nframes += _get_frame_expansion(nframes[-1],expansion_factor,n_expansion)
        if n_contraction > 0:
            nframes += _get_frame_contraction(nframes[-1], nout, n_contraction)
        nframes[-1] = nout
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
            fs = int(ceil(size_factor*decay_factor))
            if fs % 2 != 1:
                fs -= 1
            if fs < 3:
                fs = 3
            pd = int((fs - 1) / 2)
            if i == 0 and pointwise_factor > 0:
                pd, fs, dil, st = 0, 1, 1, 1
                indkey = "ind_0"
                self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, bias=trainable_weights, indice_key=indkey))
                # self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
                self.log.debug("added pointwise convolution, frames: {0} -> {1}".format(nframes[i], nframes[i + 1]))
            else:
                if fs > 3:
                    indkey = "ind_{}".format(fs)
                else:
                    indkey = "ind_0"
                st, dil = 1, 1
                self.alg.append(spconv.SubMConv2d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights, indice_key='ind_{}'.format(fs)))
                self.log.debug("added regular convolution, frames: {0} -> {1}, fs {2} pad {3}".format(nframes[i], nframes[i + 1], fs, pd))
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())
            if self.dropout:
                self.alg.append(nn.Dropout(self.dropout))
        self.func = spconv.SparseSequential(*self.alg)

    def forward(self, x):
        return self.func(x)
