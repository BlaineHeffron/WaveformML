import torch.nn as nn
from src.models.Algorithm import *
from math import floor, pow
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL


class DilationBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n, length, size_factor=3, pad_factor=0, stride_factor=1, dil_factor=2,
                 trainable_weights=False):
        self.out_length = length
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
            dil = dil_factor ** i
            pd = int(floor(pad_factor * (fs - 1) * dil_factor))
            self.alg.append(nn.Conv1d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
            arg_dict = {DIM: 2, NIN: nframes[i], NOUT: nframes[i+1], FS: fs, STR: st, PAD: pd, DIL: dil}
            self.out_length = ModelValidation.calc_output_size_1d(self.out_length, arg_dict)
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())

        self.func = nn.Sequential(*self.alg)


class LinearBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()

    def __init__(self, nin, nout, n):
        self.alg = []
        factor = pow(float(nout) / nin, 1. / n)
        for i in range(n):
            self.alg.append(nn.Linear(nin * pow(factor, i), nin * pow(factor, i + 1)))
        self.func = nn.Sequential(*self.alg)
