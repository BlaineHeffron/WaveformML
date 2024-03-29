import os
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import spconv
import torch

from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.AD1Evaluator import AD1Evaluator, CELL_LENGTH
from src.evaluation.Calibrator import Calibrator
from src.evaluation.EnergyEvaluator import EnergyEvaluatorPhys
from src.evaluation.RealDataEvaluator import RealDataEvaluator
from src.evaluation.WaveformEvaluator import WaveformEvaluator
from src.utils.PlotUtils import plot_z_acc_matrix, plot_hist2d, plot_hist1d, MultiLinePlot
from src.utils.SQLUtils import CalibrationDB
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import z_deviation, safe_divide_2d, calc_calib_z_E, z_basic_prediction, z_error, \
    z_deviation_with_E, z_basic_prediction_dense, z_deviation_with_E_full_correlation, E_basic_prediction_dense, \
    mean_absolute_error_dense, swap_sparse_from_dense
from src.utils.util import get_bins, get_bin_midpoints


class ZEvaluatorBase:
    def __init__(self, logger):
        self.hascal = False
        self.logger = logger
        self.nmult = 6
        self.nx = 14
        self.ny = 11
        self.z_scale = 1200.
        self.n_bins = 20
        self.n_err_bins = 50
        self.error_low = -1000.
        self.error_high = 1000.
        self.E_high = 10.0
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

    def set_logger(self, logger):
        self.logger = logger

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

    def add(self, predictions, target, c, f, E=None, additional_fields=None):
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        coo = c.detach().cpu().numpy()
        z_deviation(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                    self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                    self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                    self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                    self.nmult, self.n_bins, self.z_scale)
        z_error(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
        if self.hascal:
            self.z_from_cal(coo, f, targ, E)

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


class ZEvaluatorPhys(ZEvaluatorBase, AD1Evaluator):
    def __init__(self, logger, e_scale=None):
        ZEvaluatorBase.__init__(self, logger)
        AD1Evaluator.__init__(self, logger, e_scale=e_scale)
        self.hascal = True

    def z_from_cal(self, c, f, targ, E=None):
        pred = np.zeros(f[:, 4].shape)
        z = f[:, 4].detach().cpu().numpy()
        z_basic_prediction(c, z, pred)
        if E is None:
            E = f[:, 0] * self.E_scale
            E = self.get_dense_matrix(E, c)
        pred = torch.tensor(pred)
        pred = self.get_dense_matrix(pred, c)
        z_deviation_with_E(c, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae_cal"][0],
                           self.results["seg_mult_mae_cal"][1], self.results["z_mult_mae_dual_cal"][0],
                           self.results["z_mult_mae_dual_cal"][1], self.results["z_mult_mae_single_cal"][0],
                           self.results["z_mult_mae_single_cal"][1], self.seg_status, self.nx, self.ny,
                           self.nmult, self.n_bins, self.z_scale, E[:, 0, :, :], self.results["E_mult_mae_dual_cal"][0],
                           self.results["E_mult_mae_dual_cal"][1], self.results["E_mult_mae_single_cal"][0],
                           self.results["E_mult_mae_single_cal"][1], self.E_low, self.E_high)
        z_error(c, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error_cal"], self.n_err_bins,
                self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)

    def add(self, predictions, target, c, f, E=None, additional_fields=None):
        if E is not None:
            self.set_true_E()
            E = E.detach().cpu().numpy() * self.E_scale
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        coo = target.detach().cpu().numpy()
        if E is None:
            E = f[:, 0] * self.E_scale
            E = self.get_dense_matrix(E, c)
        else:
            E = np.expand_dims(E, axis=1)
        z_deviation_with_E(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                           self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                           self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                           self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                           self.nmult, self.n_bins, self.z_scale, E[:, 0, :, :], self.results["E_mult_mae_dual"][0],
                           self.results["E_mult_mae_dual"][1], self.results["E_mult_mae_single"][0],
                           self.results["E_mult_mae_single"][1], self.E_low, self.E_high)
        z_error(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
        if self.hascal:
            self.z_from_cal(coo, f, targ, E)
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

    def z_from_cal(self, c, f, targ, E=None, target_is_cal=False):
        pred = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        cal_E = np.zeros((targ.shape[0], targ.shape[2], targ.shape[3]))
        calc_calib_z_E(c, f, pred, cal_E, self.sample_width, self.calibrator.t_interp_curves,
                       self.calibrator.sampletime,
                       self.calibrator.rel_times, self.gain_factor, self.calibrator.eres,
                       self.calibrator.time_pos_curves, self.calibrator.light_pos_curves,
                       self.calibrator.light_sum_curves, self.z_scale, self.n_samples)

        if target_is_cal:
            pred = self.get_dense_matrix(torch.full((c.shape[0],), 0.5, dtype=torch.float32), c).squeeze(1)
            pred[:, self.seg_status != 0.5] = targ[:, 0, self.seg_status != 0.5]
            z_basic_prediction_dense(c, pred, targ[:, 0, :, :], target_is_cal)
        if E is None:
            E = cal_E
        z_deviation_with_E(c, pred, targ[:, 0, :, :], self.results["seg_mult_mae_cal"][0],
                           self.results["seg_mult_mae_cal"][1], self.results["z_mult_mae_dual_cal"][0],
                           self.results["z_mult_mae_dual_cal"][1], self.results["z_mult_mae_single_cal"][0],
                           self.results["z_mult_mae_single_cal"][1], self.seg_status, self.nx, self.ny,
                           self.nmult, self.n_bins, self.z_scale, E, self.results["E_mult_mae_dual_cal"][0],
                           self.results["E_mult_mae_dual_cal"][1], self.results["E_mult_mae_single_cal"][0],
                           self.results["E_mult_mae_single_cal"][1], self.E_low, self.E_high)
        z_error(c, pred, targ[:, 0, :, :], self.results["seg_sample_error_cal"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)
        return E

    def add(self, predictions, target, c, f, E=None, target_is_cal=False, additional_fields=None):
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
            E = E.detach().cpu().numpy() * self.E_scale
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        coo = c.detach().cpu().numpy()
        if self.hascal:
            if E is None:
                E = self.z_from_cal(coo, f, targ, E, target_is_cal)
            else:
                self.z_from_cal(coo, f, targ, E, target_is_cal)
            z_deviation_with_E(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                               self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                               self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                               self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                               self.nmult, self.n_bins, self.z_scale, E, self.results["E_mult_mae_dual"][0],
                               self.results["E_mult_mae_dual"][1], self.results["E_mult_mae_single"][0],
                               self.results["E_mult_mae_single"][1], self.E_low, self.E_high)
        else:
            z_deviation(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_mult_mae"][0],
                        self.results["seg_mult_mae"][1], self.results["z_mult_mae_dual"][0],
                        self.results["z_mult_mae_dual"][1], self.results["z_mult_mae_single"][0],
                        self.results["z_mult_mae_single"][1], self.seg_status, self.nx, self.ny,
                        self.nmult, self.n_bins, self.z_scale)
        z_error(coo, pred[:, 0, :, :], targ[:, 0, :, :], self.results["seg_sample_error"], self.n_err_bins, self.error_low,
                self.error_high, self.nmult, self.sample_segs, self.z_scale)


class ZEvaluatorRealWFNorm(RealDataEvaluator, WaveformEvaluator):

    def __init__(self, logger, calgroup=None, namespace=None, e_scale=None, additional_field_names=None, **kwargs):
        WaveformEvaluator.__init__(self, logger, calgroup=calgroup, e_scale=e_scale, additional_field_names=additional_field_names, **kwargs)
        RealDataEvaluator.__init__(self, logger, calgroup=calgroup, e_scale=e_scale,
                                   additional_field_names=additional_field_names, metric_name="mean absolute error", metric_unit="mm",
                                   target_has_phys=True, scaling=self.z_scale, **kwargs)
        if calgroup is not None:
            self.EnergyEvaluator = EnergyEvaluatorPhys(logger, calgroup=None, e_scale=e_scale, namespace=namespace)
        self.E_bounds = self.default_bins[0][0:2]
        self.mult_bounds = [0.5, 6.5]
        self.n_mult = 6
        self.n_E = self.default_bins[0][-1]
        self.E_bin_centers = get_bin_midpoints(*self.default_bins[0])
        self.n_z = 100
        self.z_bounds = [0., CELL_LENGTH]
        self.E_mult_names = ["E_mult_single", "E_mult_single_cal", "E_mult_dual", "E_mult_dual_cal"]
        self.Z_mult_names = ["z_mult_single", "z_mult_single_cal", "z_mult_dual", "z_mult_dual_cal"]
        self.E_mult_titles = ["Single Ended", "Single Ended", "Double Ended", "Double Ended"]
        self.z_E_names = ["z_E_single", "z_E_single_cal", "z_E_dual", "z_E_dual_cal"]
        self.seg_mult_names = ["seg_mult_zmae", "seg_mult_zmae_cal"]
        if namespace:
            self.namespace = "evaluation/{}_".format(namespace)
        else:
            self.namespace = "evaluation/"
        self.initialize()

    def set_logger(self, logger):
        self.logger = logger
        if hasattr(self, "EnergyEvaluator"):
            self.EnergyEvaluator.logger = logger

    def initialize(self):
        self.register_duplicates(self.E_mult_names,
                                 [self.n_E, self.n_mult], [self.E_bounds[0], self.mult_bounds[0]],
                                 [self.E_bounds[1], self.mult_bounds[1]], 2, ["Visible Energy", "Multiplicity"],
                                 ["MeVee", ""],
                                 "Z Mean Absolute Error", "mm", underflow=(1, 0), scale=self.z_scale)
        self.register_duplicates(self.Z_mult_names,
                                 [self.n_z, self.n_mult], [self.z_bounds[0], self.mult_bounds[0]],
                                 [self.z_bounds[1], self.mult_bounds[1]], 2, ["Distance from PMT", "Multiplicity"],
                                 ["mm", ""],
                                 "Z Mean Absolute Error", "mm", underflow=(1, 0), scale=self.z_scale)
        self.register_duplicates(self.z_E_names, [self.n_z, self.n_E],
                                 [self.z_bounds[0], self.E_bounds[0]],
                                 [self.z_bounds[1], self.E_bounds[1]], 2,
                                 ["Distance from PMT", "Visible Energy"], ["mm", "MeVee"],
                                 "Z Mean Absolute Error", "mm", scale=self.z_scale)
        self.register_duplicates(self.seg_mult_names, [self.nx, self.ny, self.n_mult],
                                 [0.5, 0.5, 0.5],
                                 [self.nx + 0.5, self.ny + 0.5, self.n_mult + 0.5], 3,
                                 ["x segment", "y segment", "Multiplicity"], [""] * 3,
                                 "Z Mean Absolute Error", "mm",
                                 underflow=False, overflow=(0, 0, 1), scale=self.z_scale)

    def add(self, predictions, target, c, f, additional_fields=None):
        """
        @param predictions: tensor of dimension 4: (batch, predictions, x, y)  here predictions is length 1
        @param target: tensor of dimension 4 (batch, phys quantities, x, y) here phys quantities is length 7 of normalized phys quantities
        @param c: tensor of dimension 2 (batch, 3) coordinates
        @param additional_fields: list of additional fields (tensors)
        """
        pred = predictions.detach().cpu().numpy()
        targ = target.detach().cpu().numpy()
        coo = c.detach().cpu().numpy()
        if self.has_PID:
            results = np.zeros_like(pred[:, 0, :, :])
            mean_absolute_error_dense(pred[:, 0, :, :], targ[:, self.z_index, :, :], results)
            RealDataEvaluator.add(self, results, targ, c, additional_fields)

        if self.analyze_waveforms:
            z_pred = (pred[:, 0, :, :] - 0.5) * self.z_scale
            z_real = (targ[:, self.z_index, :, :] - 0.5) * self.z_scale
            z_list = np.zeros(coo.shape[0])
            z_pred_list = np.zeros(coo.shape[0])
            swap_sparse_from_dense(z_pred_list, z_pred, coo)
            swap_sparse_from_dense(z_list, z_real, coo)
            self.analyze_wf_z(f, coo, z_list, z_pred_list, additional_fields)
        z_deviation_with_E_full_correlation(coo, pred[:, 0, :, :], targ[:, self.z_index, :, :],
                                            self.results["seg_mult_zmae"][0],
                                            self.results["seg_mult_zmae"][1],
                                            self.results["z_mult_dual"][0], self.results["z_mult_dual"][1],
                                            self.results["z_mult_single"][0], self.results["z_mult_single"][1],
                                            self.results["z_E_single"][0], self.results["z_E_single"][1],
                                            self.results["z_E_dual"][0], self.results["z_E_dual"][1],
                                            self.results["E_mult_single"][0], self.results["E_mult_single"][1],
                                            self.results["E_mult_dual"][0], self.results["E_mult_dual"][1],
                                            self.seg_status, self.blind_detl, self.nx, self.ny, self.n_mult,
                                            self.n_z, self.z_scale, targ[:, self.E_index, :, :],
                                            self.E_bounds[0] / self.E_scale, self.E_bounds[1] / self.E_scale, self.n_E)

        cal_pred = self.get_dense_matrix(torch.full((c.shape[0], 1), 0.5, dtype=torch.float32), c, to_numpy=True).squeeze(1)
        cal_pred[:, self.seg_status != 0.5] = targ[:, self.z_index, self.seg_status != 0.5]
        z_basic_prediction_dense(coo, cal_pred, targ[:, self.z_index, :, :], truth_is_cal=True)
        z_deviation_with_E_full_correlation(coo, cal_pred, targ[:, self.z_index, :, :],
                                            self.results["seg_mult_zmae_cal"][0],
                                            self.results["seg_mult_zmae_cal"][1],
                                            self.results["z_mult_dual_cal"][0], self.results["z_mult_dual_cal"][1],
                                            self.results["z_mult_single_cal"][0], self.results["z_mult_single_cal"][1],
                                            self.results["z_E_single_cal"][0], self.results["z_E_single_cal"][1],
                                            self.results["z_E_dual_cal"][0], self.results["z_E_dual_cal"][1],
                                            self.results["E_mult_single_cal"][0], self.results["E_mult_single_cal"][1],
                                            self.results["E_mult_dual_cal"][0], self.results["E_mult_dual_cal"][1],
                                            self.seg_status, self.blind_detl, self.nx, self.ny, self.n_mult,
                                            self.n_z, self.z_scale, targ[:, self.E_index, :, :],
                                            self.E_bounds[0] / self.E_scale, self.E_bounds[1] / self.E_scale, self.n_E)

        if hasattr(self, "calibrator"):
            cal_E_pred = np.zeros(predictions[:, 0, :, :].shape)
            PE0 = targ[:, self.PE0_index, :, :] * self.PE_scale
            PE1 = targ[:, self.PE1_index, :, :] * self.PE_scale
            PE0[:, self.blind_detl == 1] = 0
            PE1[:, self.blind_detr == 1] = 0
            e = targ[:, self.E_index, :, :] * self.E_scale
            dense_E = np.concatenate((np.expand_dims(e, 1), np.expand_dims(PE0, 1), np.expand_dims(PE1, 1)), axis=1)
            z_pred = (pred[:, 0, :, :] - 0.5) * self.z_scale
            E_basic_prediction_dense(dense_E, z_pred, self.blind_detl, self.blind_detr,
                                     self.calibrator.light_pos_curves,
                                     self.calibrator.light_sum_curves, cal_E_pred)
            cal_E_pred = cal_E_pred / self.E_scale
            self.EnergyEvaluator.add(np.expand_dims(cal_E_pred, 1), target[:, self.E_index, :, :].unsqueeze(1), c,
                                     target, True, Z_pred=np.expand_dims(targ[:, self.z_index, :, :], 1))


    def dump(self):
        self.retrieve_error_metrics()
        for name, title in zip(self.E_mult_names, self.E_mult_titles):
            self.log_total(name, "{0}{1}".format(self.namespace, name), title)
            self.log_metric(name, "{0}{1}_{2}".format(self.namespace, name, "MAE"), title)
        for name, title in zip(self.z_E_names, self.E_mult_titles):
            self.log_total(name, "{0}{1}".format(self.namespace, name), title)
            self.log_metric(name, "{0}{1}_{2}".format(self.namespace, name, "MAE"), title)
        for name, title in zip(self.Z_mult_names, self.E_mult_titles):
            self.log_total(name, "{0}{1}".format(self.namespace, name), title)
            self.log_metric(name, "{0}{1}_{2}".format(self.namespace, name, "MAE"), title)
        for name in self.seg_mult_names:
            self.log_segment_metric(name, "{0}{1}".format(self.namespace, name))
        if hasattr(self, "EnergyEvaluator"):
            self.EnergyEvaluator.dump()
        RealDataEvaluator.dump(self)
        if self.analyze_waveforms:
            self.dump_wf_z()


    def retrieve_error_metrics(self):
        single_err_E = []
        dual_err_E = []
        single_err_E_cal = []
        dual_err_E_cal = []
        for i in range(1, self.n_E + 1):
            if np.sum(self.results["E_mult_single"][1][i, :]) > 0:
                single_err_E.append(
                    self.z_scale * np.sum(self.results["E_mult_single"][0][i, :]) / np.sum(
                        self.results["E_mult_single"][1][i, :]))
            else:
                single_err_E.append(0)
            self.logger.experiment.add_scalar("{}single_z_MAE_vs_E".format(self.namespace), single_err_E[-1],
                                              global_step=i)

            if np.sum(self.results["E_mult_dual"][1][i, :]) > 0:
                dual_err_E.append(
                    self.z_scale * np.sum(self.results["E_mult_dual"][0][i, :]) / np.sum(
                        self.results["E_mult_dual"][1][i, :]))
            else:
                dual_err_E.append(0)
            self.logger.experiment.add_scalar("{}dual_z_MAE_vs_E".format(self.namespace), dual_err_E[-1], global_step=i)
            if np.sum(self.results["E_mult_single_cal"][1][i, :]) > 0:
                single_err_E_cal.append(
                    self.z_scale * np.sum(self.results["E_mult_single_cal"][0][i, :]) / np.sum(
                        self.results["E_mult_single_cal"][1][i, :]))
            else:
                single_err_E_cal.append(0)
            self.logger.experiment.add_scalar("{}single_z_MAE_cal_vs_E".format(self.namespace), single_err_E_cal[-1],
                                              global_step=i)
            if np.sum(self.results["E_mult_dual_cal"][1][i, :]) > 0:
                dual_err_E_cal.append(
                    self.z_scale * np.sum(self.results["E_mult_dual_cal"][0][i, :]) / np.sum(
                        self.results["E_mult_dual_cal"][1][i, :]))
            else:
                dual_err_E_cal.append(0)
            self.logger.experiment.add_scalar("{}dual_z_MAE_cal_vs_E".format(self.namespace), dual_err_E_cal[-1],
                                              global_step=i)
        labels = ["single NN", "dual NN", "single cal", "dual cal"]
        xlabel = "Visible Energy [MeVee]"
        ylabel = "Z Mean Absolute Error [mm]"
        self.logger.experiment.add_figure("{}E_error_summary_mult".format(self.namespace),
                                          MultiLinePlot(self.E_bin_centers,
                                                        [single_err_E, dual_err_E, single_err_E_cal,
                                                         dual_err_E_cal],
                                                        labels, xlabel, ylabel, ylog=False))
