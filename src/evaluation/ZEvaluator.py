import os
from math import floor

import matplotlib.pyplot as plt
import numpy as np
from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.Calibrator import Calibrator
# from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.utils.PlotUtils import plot_z_acc_matrix, plot_hist2d
from src.utils.SQLUtils import CalibrationDB
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import z_deviation, safe_divide_2d, calc_calib_z_E
from src.utils.util import get_bins


class ZEvaluator:
    def __init__(self, logger, calgroup=None):
        self.logger = logger
        self.nmult = 10
        self.nx = 14
        self.ny = 11
        self.z_scale = 1200.
        self.sample_width = 4
        self.n_samples = 150
        self.n_bins = 20
        self.zmin = -1. * self.z_scale / 2
        self.zmax = self.z_scale / 2
        self.z_bin_edges = get_bins(self.zmin, self.zmax, self.n_bins)
        self.mult_bin_edges = get_bins(0.5, self.nmult + 0.5, self.nmult)
        self.hascal = False
        self.colormap = plt.cm.viridis
        SE_dead_pmts = [1, 0, 2, 4, 6, 7, 9, 10, 12, 13, 16, 19, 20, 21, 22, 24, 26, 27, 34, 36, 37, 43, 46, 48,
                             55,
                             54, 56, 58, 65, 68, 72, 80, 82, 85, 88, 93, 95, 97, 96, 105, 111, 112, 120, 122, 137, 138,
                             139, 141, 147, 158, 166, 173, 175, 188, 195, 215, 230, 243, 244, 245, 252, 255, 256, 261,
                             273, 279, 282]
        self.seg_status = np.zeros((self.nx, self.ny), dtype=np.float32)  # 0 for good, 0.5 for single ended, 1 for dead
        self.set_SE_segs(SE_dead_pmts)
        if calgroup is not None:
            self.hascal = True
            if "PROSPECT_CALDB" not in os.environ.keys():
                raise ValueError(
                    "Error: could not find PROSPECT_CALDB environment variable. Please set PROSPECT_CALDB to be the "
                    "path of the sqlite3 calibration database.")
            gains = get_gains(os.environ["PROSPECT_CALDB"], calgroup)
            self.gain_factor = np.divide(np.full((self.nx, self.ny, 2), MAX_RANGE), gains)
            self.t_center = np.arange(2, self.n_samples * self.sample_width - 1, self.sample_width)
            self.calibrator = Calibrator(CalibrationDB(os.environ["PROSPECT_CALDB"], calgroup))
        # self.metrics = []
        self._init_results()

    def set_SE_segs(self, SE_dead_pmts):
        for pmt in SE_dead_pmts:
            r = pmt % 2
            seg = int((pmt - r) / 2)
            x = seg % 14
            y = floor(seg / 14)
            self.seg_status[x, y] += 0.5

    def _init_results(self):
        # metric_names = ["energy", "multiplicity", "true_z", "pred_z"]
        # metric_params = [[0.0, 10.0, 40], [0.5, 10.5, 10], [-600.,600.,40],[-600.,600.,40]]
        #i = 0
        # for name in metric_names:
        #    self.metrics.append(MetricAggregator(name, *metric_params[i], ["positron"]))
        #    i += 1
        # self.metric_pairs = MetricPairAggregator(self.metrics)
        self.results = {
            "seg_mult_mae": (
                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32)),
            "z_mult_mae_single": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "z_mult_mae_dual": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32))
        }
        if self.hascal:
            self.results["seg_mult_mae_cal"] = (np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
                                                np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32))
            self.results["z_mult_mae_single_cal"] = (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32))
            self.results["z_mult_mae_dual_cal"] = (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32))

    def add(self, predictions, target, c, f):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        z_deviation(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                    self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                    self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                    self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                    self.nmult, self.n_bins, self.z_scale)
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

        self.logger.experiment.add_figure("evaluation/z_mult_dual",
                                          plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                      self.results["z_mult_mae_dual"][1][1:self.n_bins + 1,
                                                      0:self.nmult],
                                                      "Total - double ended", "Z [mm]", "multiplicity",
                                                      r'# Pulses [$mm^{-1}$', cm=self.colormap))
        self.logger.experiment.add_figure("evaluation/z_mult_single",
                                          plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                      self.results["z_mult_mae_single"][1][1:self.n_bins + 1,
                                                      0:self.nmult],
                                                      "Total - single ended", "Z [mm]", "multiplicity",
                                                      r'# Pulses [$mm^{-1}$]', cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/z_mult_mae_dual",
                                          plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(self.results["z_mult_mae_dual"][0][1:self.n_bins + 1,
                                                                     0:self.nmult],
                                                                     self.results["z_mult_mae_dual"][1][1:self.n_bins + 1,
                                                                     0:self.nmult])*self.z_scale,
                                                      "MAE - double ended", "Z [mm]", "multiplicity",
                                                      r'# mean average error [mm]',norm_to_bin_width=False,logz=False,
                                                      cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/z_mult_mae_single",
                                          plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(self.results["z_mult_mae_single"][0][1:self.n_bins + 1,
                                                                     0:self.nmult],
                                                                     self.results["z_mult_mae_single"][1][1:self.n_bins + 1,
                                                                     0:self.nmult])*self.z_scale,
                                                      "MAE - single ended", "Z [mm]", "multiplicity",
                                                      r'# mean average error [mm]',norm_to_bin_width=False,logz=False,
                                                      cm=self.colormap))
        if self.hascal:
            for i in range(self.nmult):
                self.logger.experiment.add_figure("evaluation/cal_z_seg_mult_{0}_mae".format(i + 1),
                                                  plot_z_acc_matrix(
                                                      self.z_scale * safe_divide_2d(
                                                          self.results["seg_mult_mae_cal"][0][:, :, i],
                                                          self.results["seg_mult_mae_cal"][1][:, :, i]),
                                                      self.nx, self.ny, "mult = {0}".format(i + 1)))

            self.logger.experiment.add_figure("evaluation/cal_z_mult_dual",
                                              plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                          self.results["z_mult_mae_dual_cal"][1][1:self.n_bins + 1,
                                                          0:self.nmult],
                                                          "Total - double ended", "Z [mm]", "multiplicity",
                                                          r'# Pulses [$mm^{-1}$', cm=self.colormap))
            self.logger.experiment.add_figure("evaluation/cal_z_mult_single",
                                              plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                          self.results["z_mult_mae_single_cal"][1][1:self.n_bins + 1,
                                                          0:self.nmult],
                                                          "Total - single ended", "Z [mm]", "multiplicity",
                                                          r'# Pulses [$mm^{-1}$]', cm=self.colormap))

            self.logger.experiment.add_figure("evaluation/cal_z_mult_mae_dual",
                                              plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                          safe_divide_2d(self.results["z_mult_mae_dual_cal"][0][1:self.n_bins + 1,
                                                                         0:self.nmult],
                                                                         self.results["z_mult_mae_dual_cal"][1][1:self.n_bins + 1,
                                                                         0:self.nmult])*self.z_scale,
                                                          "MAE - double ended", "Z [mm]", "multiplicity",
                                                          r'# mean average error [mm]',norm_to_bin_width=False,logz=False,
                                                          cm=self.colormap))

            self.logger.experiment.add_figure("evaluation/cal_z_mult_mae_single",
                                              plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                          safe_divide_2d(self.results["z_mult_mae_single_cal"][0][1:self.n_bins + 1,
                                                                         0:self.nmult],
                                                                         self.results["z_mult_mae_single_cal"][1][1:self.n_bins + 1,
                                                                         0:self.nmult])*self.z_scale,
                                                          "MAE - single ended", "Z [mm]", "multiplicity",
                                                          r'# mean average error [mm]',norm_to_bin_width=False,logz=False,
                                                          cm=self.colormap))
        self._init_results()

    def z_from_cal(self, c, f, targ):
        c, f = c.detach().cpu().numpy(), f.detach().cpu().numpy()
        pred = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        E = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        calc_calib_z_E(c, f, pred, E, self.sample_width, self.calibrator.t_interp_curves, self.calibrator.sampletime,
                       self.calibrator.rel_times, self.gain_factor, self.calibrator.eres,
                       self.calibrator.time_pos_curves, self.calibrator.light_pos_curves,
                       self.calibrator.light_sum_curves, self.z_scale, self.n_samples)
        z_deviation(pred, targ[:, 0, :, :], self.results["seg_mult_mae_cal"][0],
                    self.results["seg_mult_mae_cal"][1], self.results["z_mult_mae_dual_cal"][0],
                    self.results["z_mult_mae_dual_cal"][1], self.results["z_mult_mae_single_cal"][0],
                    self.results["z_mult_mae_single_cal"][1], self.seg_status, self.nx, self.ny,
                    self.nmult, self.n_bins, self.z_scale)
