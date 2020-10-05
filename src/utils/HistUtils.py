from matplotlib.pyplot import hist as histplt
from numpy import histogram,  array, zeros, int32
from torch import tensor

class HistCollator(histogram):
    def __init__(self):
        self.hist = None

    def add_histogram(self, hist):
        if self.hist is None:
            self._init_hist(hist)
        self._add_hist(hist)

    def _create_hist_from_array(self,array):
        self.hist = histogram(zeros(array.shape, dtype=int32))

    def _create_hist_from_tensor(self,tensor):
        self.hist = histogram(zeros((tensor.shape), dtype=int32))

    def _init_hist(self,hist):
        if isinstance(hist,type(array)):
            self._create_hist_from_array(array)
        elif isinstance(hist,type(tensor)):
            self._create_hist_from_tensor(array)

    def _add_hist(self,hist):
        for i in range(len(hist.shape)):
            for j in range(len(hist.shape[i])):
                self.hist.set(hist[i,j])

