import logging
from math import floor, ceil

import spconv
import torch.nn as nn
from src.models.Algorithm import *
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL


class SparseConv2DBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n, size, to_dense,
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
