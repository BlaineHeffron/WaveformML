import os
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import spconv
import torch

from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.AD1Evaluator import PhysCoordEvaluator
from src.evaluation.Calibrator import Calibrator
# from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.utils.PlotUtils import plot_z_acc_matrix, plot_hist2d, plot_hist1d, MultiLinePlot
from src.utils.SQLUtils import CalibrationDB
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import z_deviation, safe_divide_2d, calc_calib_z_E, z_basic_prediction, z_error, \
    z_deviation_with_E
from src.utils.util import get_bins, get_bin_midpoints


class ZEvaluatorBase:
    def __init__(self, logger):
        self.hascal = False
        self.logger = logger
        self.nmult = 10
        self.nx = 14
        self.ny = 11
        self.z_scale = 1200.
        self.n_bins = 20
        self.n_err_bins = 50
        self.error_low = -1000.
        self.error_high = 1000.
        self.E_high = 12.0
        self.E_low = 0.0
        self.true_E_high = 9.0
        self.has_true_E = False
        self.spatial_size = np.array([14, 11])
        self.permute_tensor = torch.LongTensor([2, 0, 1])  # needed because spconv requires batch index first
        self.zmin = -1. * self.z_scale / 2
        self.zmax = self.z_scale / 2
        self.z_err_edges = get_bins(self.error_low, self.error_high, self.n_err_bins)
        self.z_bin_edges = get_bins(self.zmin, self.zmax, self.n_bins)
        self.E_bin_edges = get_bins(self.E_low, self.E_high, self.n_bins)
        self.true_E_bin_edges = get_bins(self.E_low, self.true_E_high, self.n_bins)
        self.E_bin_centers = get_bin_midpoints(self.E_low, self.E_high, self.n_bins)
        self.true_E_bin_centers = get_bin_midpoints(self.E_low, self.true_E_high, self.n_bins)
        self.E_label = "Visible Energy [MeV]"
        self.E_scale = 12.
        self.mult_bin_edges = get_bins(0.5, self.nmult + 0.5, self.nmult)
        self.colormap = plt.cm.viridis
        SE_dead_pmts = [1, 0, 2, 4, 6, 7, 9, 10, 12, 13, 16, 19, 20, 21, 22, 24, 26, 27, 34, 36, 37, 43, 46, 48,
                        55,
                        54, 56, 58, 65, 68, 72, 80, 82, 85, 88, 93, 95, 97, 96, 105, 111, 112, 120, 122, 137, 138,
                        139, 141, 147, 158, 166, 173, 175, 188, 195, 215, 230, 243, 244, 245, 252, 255, 256, 261,
                        273, 279, 282]
        self.seg_status = np.zeros((self.nx, self.ny), dtype=np.float32)  # 0 for good, 0.5 for single ended, 1 for dead
        self.set_SE_segs(SE_dead_pmts)
        # self.metrics = []
        self._init_results()

    def set_SE_segs(self, SE_dead_pmts):
        for pmt in SE_dead_pmts:
            r = pmt % 2
            seg = int((pmt - r) / 2)
            x = seg % 14
            y = floor(seg / 14)
            self.seg_status[x, y] += 0.5

    def set_true_E(self):
        if not self.has_true_E:
            self.has_true_E = True
            self.E_label = "True Energy Deposited [MeV]"
            self.E_high = self.true_E_high
            self.E_bin_edges = self.true_E_bin_edges
            self.E_bin_centers = self.true_E_bin_centers

    def _init_results(self):
        # metric_names = ["energy", "multiplicity", "true_z", "pred_z"]
        # metric_params = [[0.0, 10.0, 40], [0.5, 10.5, 10], [-600.,600.,40],[-600.,600.,40]]
        # i = 0
        # for name in metric_names:
        #    self.metrics.append(MetricAggregator(name, *metric_params[i], ["positron"]))
        #    i += 1
        # self.metric_pairs = MetricPairAggregator(self.metrics)
        self.sample_segs = np.array([[5, 4], [10, 3], [7, 5]], dtype=np.int32)
        self.results = {"seg_mult_mae": (
            np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
            np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32)),
            "z_mult_mae_single": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "z_mult_mae_dual": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "E_mult_mae_single": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "E_mult_mae_dual": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "seg_mult_mae_cal": (np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.float32),
                                 np.zeros((self.nx, self.ny, self.nmult + 1), dtype=np.int32)),
            "z_mult_mae_single_cal": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "z_mult_mae_dual_cal": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "E_mult_mae_single_cal": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "E_mult_mae_dual_cal": (
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.float32),
                np.zeros((self.n_bins + 2, self.nmult + 1), dtype=np.int32)),
            "seg_sample_error": np.zeros((len(self.sample_segs), self.nmult + 1, self.n_err_bins + 2), dtype=np.int32),
            "seg_sample_error_cal": np.zeros((len(self.sample_segs), self.nmult + 1, self.n_err_bins + 2),
                                             dtype=np.int32)
        }

    def add(self, predictions, target, c, f, E=None):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        z_deviation(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                    self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                    self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                    self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                    self.nmult, self.n_bins, self.z_scale)
        z_error(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
        if self.hascal:
            self.z_from_cal(c, f, targ, E)

    def retrieve_error_metrics(self):
        single_err = np.sum(self.results["z_mult_mae_single"][0]) / np.sum(self.results["z_mult_mae_single"][1])
        dual_err = np.sum(self.results["z_mult_mae_dual"][0]) / np.sum(self.results["z_mult_mae_dual"][1])
        plot_cal = False
        if self.hascal:
            single_cal_err = np.sum(self.results["z_mult_mae_single_cal"][0]) / np.sum(
                self.results["z_mult_mae_single_cal"][1])
            dual_cal_err = np.sum(self.results["z_mult_mae_dual_cal"][0]) / np.sum(
                self.results["z_mult_mae_dual_cal"][1])
            plot_cal = True
        else:
            single_cal_err = 0
            dual_cal_err = 0
        self.logger.experiment.add_scalar("evaluation/single_mae", single_err * self.z_scale)
        self.logger.experiment.add_scalar("evaluation/dual_mae", dual_err * self.z_scale)
        self.logger.experiment.add_scalar("evaluation/single_mae_cal", single_cal_err * self.z_scale)
        self.logger.experiment.add_scalar("evaluation/dual_mae_cal", dual_cal_err * self.z_scale)
        single_err_mult = []
        dual_err_mult = []
        single_err_mult_cal = []
        dual_err_mult_cal = []
        for i in range(self.nmult):
            single_err_mult.append(
                self.z_scale * np.sum(self.results["z_mult_mae_single"][0][:, i]) / np.sum(
                    self.results["z_mult_mae_single"][1][:, i]))
            self.logger.experiment.add_scalar("evaluation/single_mae_mult", single_err_mult[-1], global_step=i + 1)
            dual_err_mult.append(
                self.z_scale * np.sum(self.results["z_mult_mae_dual"][0][:, i]) / np.sum(
                    self.results["z_mult_mae_dual"][1][:, i]))
            self.logger.experiment.add_scalar("evaluation/dual_mae_mult", dual_err_mult[-1], global_step=i + 1)
            if plot_cal:
                single_err_mult_cal.append(
                    self.z_scale * np.sum(self.results["z_mult_mae_single_cal"][0][:, i]) / np.sum(
                        self.results["z_mult_mae_single_cal"][1][:, i]))
                self.logger.experiment.add_scalar("evaluation/single_mae_mult_cal", single_err_mult_cal[-1],
                                                  global_step=i + 1)
                dual_err_mult_cal.append(
                    self.z_scale * np.sum(self.results["z_mult_mae_dual_cal"][0][:, i]) / np.sum(
                        self.results["z_mult_mae_dual_cal"][1][:, i]))
                self.logger.experiment.add_scalar("evaluation/dual_mae_mult_cal", dual_err_mult_cal[-1],
                                                  global_step=i + 1)
        single_err_E = []
        dual_err_E = []
        single_err_E_cal = []
        dual_err_E_cal = []
        if plot_cal:
            for i in range(1, self.n_bins + 1):
                single_err_E.append(
                    self.z_scale * np.sum(self.results["E_mult_mae_single"][0][i, :]) / np.sum(
                        self.results["E_mult_mae_single"][1][i, :]))
                self.logger.experiment.add_scalar("evaluation/single_mae_E", single_err_E[-1], global_step=i)
                dual_err_E.append(
                    self.z_scale * np.sum(self.results["E_mult_mae_dual"][0][i, :]) / np.sum(
                        self.results["E_mult_mae_dual"][1][i, :]))
                self.logger.experiment.add_scalar("evaluation/dual_mae_E", dual_err_E[-1], global_step=i)
                single_err_E_cal.append(
                    self.z_scale * np.sum(self.results["E_mult_mae_single_cal"][0][i, :]) / np.sum(
                        self.results["E_mult_mae_single_cal"][1][i, :]))
                self.logger.experiment.add_scalar("evaluation/single_mae_E_cal", single_err_E_cal[-1],
                                                  global_step=i)
                dual_err_E_cal.append(
                    self.z_scale * np.sum(self.results["E_mult_mae_dual_cal"][0][i, :]) / np.sum(
                        self.results["E_mult_mae_dual_cal"][1][i, :]))
                self.logger.experiment.add_scalar("evaluation/dual_mae_E_cal", dual_err_E_cal[-1], global_step=i)
        labels = ["single NN", "dual NN", "single cal", "dual cal"]
        xlabel = "multiplicity"
        ylabel = "MAE [mm]"
        if plot_cal:
            self.logger.experiment.add_figure("evaluation/z_error_summary_mult",
                                              MultiLinePlot([i for i in range(1, self.nmult + 1)],
                                                            [single_err_mult, dual_err_mult, single_err_mult_cal,
                                                             dual_err_mult_cal],
                                                            labels, xlabel, ylabel, ylog=False))
            self.logger.experiment.add_figure("evaluation/z_error_summary_E_single",
                                              MultiLinePlot(self.E_bin_centers,
                                                            [single_err_E, single_err_E_cal],
                                                            ["NN", "calibration"], self.E_label, ylabel,
                                                            ylog=False, title="Single Ended"))
            self.logger.experiment.add_figure("evaluation/z_error_summary_E_dual",
                                              MultiLinePlot(self.E_bin_centers,
                                                            [dual_err_E, dual_err_E_cal],
                                                            ["NN", "calibration"], self.E_label, ylabel,
                                                            ylog=False, title="Dual Ended"))
        else:
            self.logger.experiment.add_figure("evaluation/error_summary_mult",
                                              MultiLinePlot([i for i in range(1, self.nmult + 1)],
                                                            [single_err_mult, dual_err_mult],
                                                            labels[0:2], xlabel, ylabel, ylog=False))

    def dump(self):
        self.retrieve_error_metrics()
        for i in range(self.nmult):
            for j in range(self.sample_segs.shape[0]):
                self.logger.experiment.add_figure("evaluation/z_seg_{0}_{1}_mult_{2}_error".format(
                    self.sample_segs[j, 0] + 1, self.sample_segs[j, 1] + 1, i + 1),
                    plot_hist1d(self.z_err_edges, self.results["seg_sample_error"][j, i, 1:self.n_err_bins + 1],
                                "segment {0},{1} mult {2}".format(
                                    self.sample_segs[j, 0] + 1,
                                    self.sample_segs[j, 1] + 1, i + 1),
                                "z error [mm]", r'total / bin width [$mm^{-1}$]'))
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
                                                      safe_divide_2d(
                                                          self.results["z_mult_mae_dual"][0][1:self.n_bins + 1,
                                                          0:self.nmult],
                                                          self.results["z_mult_mae_dual"][1][1:self.n_bins + 1,
                                                          0:self.nmult]) * self.z_scale,
                                                      "MAE - double ended", "Z [mm]", "multiplicity",
                                                      r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                      logz=False,
                                                      cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/z_mult_mae_single",
                                          plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(
                                                          self.results["z_mult_mae_single"][0][1:self.n_bins + 1,
                                                          0:self.nmult],
                                                          self.results["z_mult_mae_single"][1][1:self.n_bins + 1,
                                                          0:self.nmult]) * self.z_scale,
                                                      "MAE - single ended", "Z [mm]", "multiplicity",
                                                      r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                      logz=False,
                                                      cm=self.colormap))
        if self.hascal:
            for i in range(self.nmult):
                for j in range(self.sample_segs.shape[0]):
                    self.logger.experiment.add_figure("evaluation/cal_z_seg_{0}_{1}_mult_{2}_error".format(
                        self.sample_segs[j, 0] + 1, self.sample_segs[j, 1] + 1, i + 1),
                        plot_hist1d(self.z_err_edges, self.results["seg_sample_error_cal"][j, i, 1:self.n_err_bins + 1],
                                    "segment {0},{1} mult {2}".format(
                                        self.sample_segs[j, 0] + 1,
                                        self.sample_segs[j, 1] + 1, i + 1),
                                    "z error [mm]", r'total / bin width [$mm^{-1}$]'))
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
                                                          safe_divide_2d(
                                                              self.results["z_mult_mae_dual_cal"][0][1:self.n_bins + 1,
                                                              0:self.nmult],
                                                              self.results["z_mult_mae_dual_cal"][1][1:self.n_bins + 1,
                                                              0:self.nmult]) * self.z_scale,
                                                          "MAE - double ended", "Z [mm]", "multiplicity",
                                                          r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                          logz=False,
                                                          cm=self.colormap))

            self.logger.experiment.add_figure("evaluation/cal_z_mult_mae_single",
                                              plot_hist2d(self.z_bin_edges, self.mult_bin_edges,
                                                          safe_divide_2d(self.results["z_mult_mae_single_cal"][0][
                                                                         1:self.n_bins + 1,
                                                                         0:self.nmult],
                                                                         self.results["z_mult_mae_single_cal"][1][
                                                                         1:self.n_bins + 1,
                                                                         0:self.nmult]) * self.z_scale,
                                                          "MAE - single ended", "Z [mm]", "multiplicity",
                                                          r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                          logz=False,
                                                          cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/E_mult_dual",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      self.results["E_mult_mae_dual"][1][1:self.n_bins + 1,
                                                      0:self.nmult],
                                                      "Total - double ended", self.E_label, "multiplicity",
                                                      r'# Pulses [$MeV^{-1}$', cm=self.colormap))
        self.logger.experiment.add_figure("evaluation/E_mult_single",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      self.results["E_mult_mae_single"][1][1:self.n_bins + 1,
                                                      0:self.nmult],
                                                      "Total - single ended", self.E_label, "multiplicity",
                                                      r'# Pulses [$MeV^{-1}$]', cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/E_mult_mae_dual",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(
                                                          self.results["E_mult_mae_dual"][0][1:self.n_bins + 1,
                                                          0:self.nmult],
                                                          self.results["E_mult_mae_dual"][1][1:self.n_bins + 1,
                                                          0:self.nmult]) * self.z_scale,
                                                      "MAE - double ended", self.E_label, "multiplicity",
                                                      r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                      logz=False,
                                                      cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/E_mult_mae_single",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(self.results["E_mult_mae_single"][0][
                                                                     1:self.n_bins + 1,
                                                                     0:self.nmult],
                                                                     self.results["E_mult_mae_single"][1][
                                                                     1:self.n_bins + 1,
                                                                     0:self.nmult]) * self.z_scale,
                                                      "MAE - single ended", self.E_label, "multiplicity",
                                                      r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                      logz=False,
                                                      cm=self.colormap))
        self.logger.experiment.add_figure("evaluation/E_mult_dual",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      self.results["E_mult_mae_dual"][1][1:self.n_bins + 1,
                                                      0:self.nmult],
                                                      "Total - double ended", self.E_label, "multiplicity",
                                                      r'# Pulses [$MeV^{-1}$', cm=self.colormap))
        self.logger.experiment.add_figure("evaluation/cal_E_mult_single",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      self.results["E_mult_mae_single_cal"][1][1:self.n_bins + 1,
                                                      0:self.nmult],
                                                      "Total - single ended", self.E_label, "multiplicity",
                                                      r'# Pulses [$MeV^{-1}$]', cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/cal_E_mult_mae_dual",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(
                                                          self.results["E_mult_mae_dual_cal"][0][1:self.n_bins + 1,
                                                          0:self.nmult],
                                                          self.results["E_mult_mae_dual_cal"][1][1:self.n_bins + 1,
                                                          0:self.nmult]) * self.z_scale,
                                                      "MAE - double ended", self.E_label, "multiplicity",
                                                      r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                      logz=False,
                                                      cm=self.colormap))

        self.logger.experiment.add_figure("evaluation/cal_E_mult_mae_single",
                                          plot_hist2d(self.E_bin_edges, self.mult_bin_edges,
                                                      safe_divide_2d(self.results["E_mult_mae_single_cal"][0][
                                                                     1:self.n_bins + 1,
                                                                     0:self.nmult],
                                                                     self.results["E_mult_mae_single_cal"][1][
                                                                     1:self.n_bins + 1,
                                                                     0:self.nmult]) * self.z_scale,
                                                      "MAE - single ended", self.E_label, "multiplicity",
                                                      r'# mean absolute error [mm]', norm_to_bin_width=False,
                                                      logz=False,
                                                      cm=self.colormap))
        self._init_results()

    def z_from_cal(self, c, f, targ, E=None):
        pass

    def get_dense_matrix(self, data: torch.tensor, c: torch.tensor):
        batch_size = c[-1, -1] + 1
        data = spconv.SparseConvTensor(data.unsqueeze(1), c[:, self.permute_tensor],
                                       self.spatial_size, batch_size)
        data = data.dense()
        data = data.detach().cpu().numpy()
        return data


class ZEvaluatorPhys(ZEvaluatorBase, PhysCoordEvaluator):
    def __init__(self, logger, e_scale=None):
        ZEvaluatorBase.__init__(self, logger)
        PhysCoordEvaluator.__init__(self, e_scale=e_scale)
        self.hascal = True

    def z_from_cal(self, c, f, targ, E=None):
        pred = np.zeros(f[:, 4].shape)
        coo = c.detach().cpu().numpy()
        z = f[:, 4].detach().cpu().numpy()
        z_basic_prediction(coo, z, pred)
        if E is None:
            E = f[:, 0] * self.E_scale
            E = self.get_dense_matrix(E, c)
        else:
            E = E.unsqueeze(1)
        pred = torch.tensor(pred)
        pred = self.get_dense_matrix(pred, c)
        z_deviation_with_E(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae_cal"][0],
                           self.results["seg_mult_mae_cal"][1], self.results["z_mult_mae_dual_cal"][0],
                           self.results["z_mult_mae_dual_cal"][1], self.results["z_mult_mae_single_cal"][0],
                           self.results["z_mult_mae_single_cal"][1], self.seg_status, self.nx, self.ny,
                           self.nmult, self.n_bins, self.z_scale, E[:, 0, :, :], self.results["E_mult_mae_dual_cal"][0],
                           self.results["E_mult_mae_dual_cal"][1], self.results["E_mult_mae_single_cal"][0],
                           self.results["E_mult_mae_single_cal"][1], self.E_low, self.E_high)
        z_error(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error_cal"], self.n_err_bins,
                self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)

    def add(self, predictions, target, c, f, E=None):
        if E is not None:
            self.set_true_E()
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        if E is None:
            E = f[:, 0] * self.E_scale
            E = self.get_dense_matrix(E, c)
        else:
            E = np.expand_dims(E, dim=1)
        z_deviation_with_E(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                           self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                           self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                           self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                           self.nmult, self.n_bins, self.z_scale, E[:, 0, :, :], self.results["E_mult_mae_dual"][0],
                           self.results["E_mult_mae_dual"][1], self.results["E_mult_mae_single"][0],
                           self.results["E_mult_mae_single"][1], self.E_low, self.E_high)
        z_error(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
        if self.hascal:
            self.z_from_cal(c, f, targ, E)
        """
        self.logger.experiment.add_histogram("Energy", f[:, self.E_index])
        self.logger.experiment.add_histogram("dt", f[:, self.dt_index])
        self.logger.experiment.add_histogram("PE0", f[:, self.PE0_index])
        self.logger.experiment.add_histogram("PE1", f[:, self.PE1_index])
        self.logger.experiment.add_histogram("Z", f[:, self.z_index])
        self.logger.experiment.add_histogram("PSD", f[:, self.PSD_index])
        self.logger.experiment.add_histogram("t_offset", f[:, self.toffset_index])
        """


class ZEvaluatorWF(ZEvaluatorBase):
    def __init__(self, logger, calgroup=None):
        super(ZEvaluatorWF, self).__init__(logger)
        self.sample_width = 4
        self.n_samples = 150
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

    def z_from_cal(self, c, f, targ, E=None):
        c, f = c.detach().cpu().numpy(), f.detach().cpu().numpy()
        pred = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        cal_E = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        calc_calib_z_E(c, f, pred, cal_E, self.sample_width, self.calibrator.t_interp_curves,
                       self.calibrator.sampletime,
                       self.calibrator.rel_times, self.gain_factor, self.calibrator.eres,
                       self.calibrator.time_pos_curves, self.calibrator.light_pos_curves,
                       self.calibrator.light_sum_curves, self.z_scale, self.n_samples)
        if E is None:
            E = cal_E
        z_deviation_with_E(pred, targ[:, 0, :, :], self.results["seg_mult_mae_cal"][0],
                           self.results["seg_mult_mae_cal"][1], self.results["z_mult_mae_dual_cal"][0],
                           self.results["z_mult_mae_dual_cal"][1], self.results["z_mult_mae_single_cal"][0],
                           self.results["z_mult_mae_single_cal"][1], self.seg_status, self.nx, self.ny,
                           self.nmult, self.n_bins, self.z_scale, E, self.results["E_mult_mae_dual_cal"][0],
                           self.results["E_mult_mae_dual_cal"][1], self.results["E_mult_mae_single_cal"][0],
                           self.results["E_mult_mae_single_cal"][1], self.E_low, self.E_high)
        z_error(pred, targ[:, 0, :, :], self.results["seg_sample_error_cal"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
        return E

    def add(self, predictions, target, c, f, E=None):
        """
        @param predictions:
        @param target:
        @param c:
        @param f:
        @param E: true energy if available
        @return:
        """
        if E is not None:
            self.set_true_E()
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        if self.hascal:
            if E is None:
                E = self.z_from_cal(c, f, targ, E)
            else:
                self.z_from_cal(c, f, targ, E)
            z_deviation_with_E(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                               self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                               self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                               self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                               self.nmult, self.n_bins, self.z_scale, E, self.results["E_mult_mae_dual"][0],
                               self.results["E_mult_mae_dual"][1], self.results["E_mult_mae_single"][0],
                               self.results["E_mult_mae_single"][1], self.E_low, self.E_high)
        else:
            z_deviation(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                        self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                        self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                        self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                        self.nmult, self.n_bins, self.z_scale)
        z_error(pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
