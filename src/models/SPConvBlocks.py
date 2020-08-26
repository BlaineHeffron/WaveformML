import spconv
import torch.nn as nn
from src.models.Algorithm import *
from math import floor, ceil
from src.utils.ModelValidation import ModelValidation, DIM, NIN, NOUT, FS, STR, PAD, DIL

class SparseConv2DBlock(Algorithm):

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)

    def __str__(self):
        super().__str__()


    def __init__(self, nin, nout, n, spacial_size, size_factor=3, pad_factor=0, stride_factor=1, dil_factor=1,
                 trainable_weights=False):
        self.alg = []
        self.out_size = spacial_size + [nin]
        self.log = logging.getLogger(__name__)
        self.log.debug("Initializing convolution block with size {}".format(out_size))
        self.ndim = len(spacial_size)
        if nin != nout:
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
            dil = dil_factor ** i
            pd = round(int(floor(pad_factor * (fs - 1) * dil_factor)) * decay_factor)
            self.alg.append(spconv.SparseConv3d(nframes[i], nframes[i + 1], fs, st, pd, dil, 1, trainable_weights))
            self.alg.append(nn.BatchNorm1d(nframes[i + 1]))
            self.alg.append(nn.ReLU())
            arg_dict = {DIM: self.ndim, NIN: nframes[i], NOUT: nframes[i+1], FS: [fs]*4, STR: [st]*4, PAD: [pd]*4, DIL: [dil]*4}
            self.out_size = ModelValidation.calc_output_size(arg_dict, self.out_size, "cur", "prev", self.ndim)
            self.log.debug("Loop {0}, output size is {1}".format(i,self.out_size))

        self.func = spconv.SparseSequential(*self.alg)



