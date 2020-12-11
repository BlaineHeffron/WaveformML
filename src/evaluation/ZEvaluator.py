import os

import numpy as np
from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.Calibrator import Calibrator
from src.utils.PlotUtils import plot_z_acc_matrix
from src.utils.SQLUtils import CalibrationDB
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import z_deviation, safe_divide_2d, calc_calib_z


class ZEvaluator:
    def __init__(self, logger, calgroup=None):
        self.logger = logger
        self.nmult = 10
        self.nx = 14
        self.ny = 11
        self.z_scale = 1200.
        self.sample_width = 4
        self.hascal = False
        if calgroup is not None:
            self.hascal = True
            if "PROSPECT_CALDB" not in os.environ.keys():
                raise ValueError(
                    "Error: could not find PROSPECT_CALDB environment variable. Please set PROSPECT_CALDB to be the "
                    "path of the sqlite3 calibration database.")
            gains = get_gains(os.environ["PROSPECT_CALDB"], calgroup)
            self.gain_factor = np.divide(np.full((14, 11, 2), MAX_RANGE), gains)
            self.t_center = np.arange(2, 599, 4)
            self.calibrator = Calibrator(CalibrationDB(os.environ["PROSPECT_CALDB"], calgroup))
        self._init_results()

    def _init_results(self):
        self.results = {
            "seg_mult_mae": (
                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32))
        }
        if self.hascal:
            self.results["seg_mult_mae_cal"] = (np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
                                        np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32))

    def add(self, predictions, target, c, f):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        z_deviation(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                    self.results["seg_mult_mae"][1], self.nx, self.ny,
                    self.nmult)
        if self.hascal:
            self.z_from_cal(c, f, targ)

    def dump(self):
        for i in range(self.nmult):
            self.logger.experiment.add_figure("evaluation/z_seg_mult_{0}_mae".format(i + 1),
                                              plot_z_acc_matrix(
                                                  self.z_scale * safe_divide_2d(
                                                      self.results["seg_mult_mae"][0][:, :, i],
                                                      self.results["seg_mult_mae"][1][:, :, i]),
                                                  self.nx, self.ny, "mult = {0}".format(i + 1)))
        if self.hascal:
            for i in range(self.nmult):
                self.logger.experiment.add_figure("evaluation/cal_z_seg_mult_{0}_mae".format(i + 1),
                                          plot_z_acc_matrix(
                                              self.z_scale * safe_divide_2d(
                                                  self.results["seg_mult_mae_cal"][0][:, :, i],
                                                  self.results["seg_mult_mae_cal"][1][:, :, i]),
                                              self.nx, self.ny, "mult = {0}".format(i + 1)))
        self._init_results()

    def z_from_cal(self, c, f, targ):
        c, f = c.detach().cpu().numpy(), f.detach().cpu().numpy()
        f = f.reshape((-1, 150, 2))
        pred = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        calc_calib_z(c, f, pred, self.sample_width, self.calibrator.t_interp_curves, self.calibrator.sampletime,
                     self.calibrator.rel_times, self.gain_factor, self.calibrator.eres,
                     self.calibrator.time_pos_curves, self.calibrator.light_pos_curves, self.z_scale)
        z_deviation(pred, targ[:, 0, :, :], self.results["seg_mult_mae_cal"][0],
                    self.results["seg_mult_mae_cal"][1], self.nx, self.ny,
                    self.nmult)
