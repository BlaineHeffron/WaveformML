import numpy as np

from src.utils.PlotUtils import plot_z_acc_matrix
from src.utils.SparseUtils import z_deviation, safe_divide_2d


class ZEvaluator:
    def __init__(self, logger):
        self.logger = logger
        self.nmult = 10
        self.nx = 14
        self.ny = 11
        self.z_scale = 1200.
        self._init_results()

    def _init_results(self):
        self.results = {
            "seg_mult_adev": (
                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32))
        }

    def add(self, predictions, target):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        z_deviation(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_adev"][0],
                    self.results["seg_mult_adev"][1], self.nx, self.ny,
                    self.nmult)

    def dump(self):
        for i in range(self.nmult):
            self.logger.experiment.add_figure("evaluation/z_seg_mult_{0}_adev".format(i + 1),
                                              plot_z_acc_matrix(
                                                  self.z_scale * safe_divide_2d(self.results["seg_mult_adev"][0][:, :, i],
                                                                             self.results["seg_mult_adev"][1][:, :, i]),
                                                  self.nx, self.ny, "mult = {0}".format(i + 1)))
        self._init_results()
