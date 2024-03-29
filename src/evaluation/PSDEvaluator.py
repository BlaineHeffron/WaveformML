import os

import matplotlib.pyplot as plt
from numpy import zeros
from torchmetrics.functional.classification import roc, precision_recall_curve

from src.datasets.HDF5Dataset import MAX_RANGE
from src.evaluation.MetricAggregator import *
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator
from src.utils.PlotUtils import plot_contour, plot_pr, plot_roc, plot_wfs, plot_bar, plot_hist2d, plot_hist1d, \
    plot_confusion_matrix, plot_n_contour
from src.utils.SQLiteUtils import get_gains
from src.utils.SparseUtils import average_pulse, find_matches, weighted_average_quantities, confusion_accumulate_1d, \
    finalize
# from src.utils.StatsUtils import calc_time_moments
from src.utils.util import list_matches


def calc_axis(amin, amax, n):
    return get_bin_midpoints(amin, amax, n)


def calc_bin_edges(amin, amax, n):
    return get_bins(amin, amax, n)


class PSDEvaluator(SingleEndedEvaluator):

    def __init__(self, class_names, logger, device, calgroup=None, has_SE=False, **kwargs):
        super().__init__(logger, calgroup, **kwargs)
        if not has_SE:
            self.unset_SE_segs()
        self.logger = logger
        self.device = device
        self.n_bins = 100
        self.n_mult = 10
        self.emin = 0.0
        self.emax = 5.0
        self.psd_min = 0.0
        self.psd_max = 0.6
        self.nx = 14
        self.ny = 11
        self.n_samples = 150
        self.n_confusion = 10
        self.has_SE = has_SE
        self.ene_label = "Energy [arb]"
        self.class_names = class_names
        self.n_classes = len(self.class_names)
        self.roc = roc
        self.pr = precision_recall_curve
        self.summed_waveforms = None
        self.n_wfs = [0] * (self.n_classes + 1)
        self.n_labelled_wfs = [0] * self.n_classes
        self.summed_labelled_waveforms = []
        self.metrics = []
        self.metric_pairs = None
        self.n_SE_max = 4
        self._init_results()
        if calgroup is not None:
            if "PROSPECT_CALDB" not in os.environ.keys():
                raise ValueError(
                    "Error: could not find PROSPECT_CALDB environment variable. Please set PROSPECT_CALDB to be the "
                    "path of the sqlite3 calibration database.")
            gains = get_gains(os.environ["PROSPECT_CALDB"], calgroup)
            self.gain_factor = np.divide(np.full((14, 11, 2), MAX_RANGE), gains)
            self.calibrated = True
            self.emax = self.default_bins[self.E_index][1]
            self.ene_label = "Visible Energy [MeV]"
        else:
            self.gain_factor = np.ones((14, 11, 2))
            self.calibrated = False

    def _init_results(self):
        metric_names = ["energy", "psd", "multiplicity", "x_dev", "y_dev", "$\Delta$t_dev", "E_dev", "t_variance", "n_variance"]
        metric_params = [self.default_bins[self.E_index], self.default_bins[self.PSD_index], [0.5, 10.5, 10], [0., 4., 20], [0., 3., 20], [0., 10., 20], [0.,2.,40],
                         [0., 1000.0, 40], [0.0, 0.25, 40]]
        i = 0
        for name in metric_names:
            self.metrics.append(MetricAggregator(name, *metric_params[i], self.class_names))
            i += 1
        self.metric_pairs = MetricPairAggregator(self.metrics)

        self.results = {
            "mult_acc": (zeros((self.n_mult + 2,), dtype=np.double), zeros((self.n_mult + 2,), dtype=np.long), zeros((self.n_mult + 2,), dtype=np.double)),
            "ene_acc": (zeros((self.n_bins + 2,), dtype=np.double), zeros((self.n_bins + 2,), dtype=np.long), zeros((self.n_bins + 2,), dtype=np.double)),
            "pos_acc": (
                zeros((self.nx + 2, self.ny + 2), dtype=np.double), zeros((self.nx + 2, self.ny + 2), dtype=np.long), zeros((self.nx + 2, self.ny + 2), dtype=np.double)),
            "ene_psd_acc": (zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.double),
                            zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.long), zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.double)),
            "confusion_energy": zeros((self.n_confusion + 1, self.n_classes, self.n_classes), dtype=np.long),
            "confusion_SE": zeros((self.n_SE_max+2, self.n_classes, self.n_classes), dtype=np.long)
        }
        for i in range(len(self.class_names)):
            self.results["ene_psd_prec_{}".format(self.class_names[i])] = (zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.double),
                            zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.long), zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.double))
            self.results["ene_prec_{}".format(self.class_names[i])] = \
                (zeros((self.n_bins + 2,), dtype=np.double), zeros((self.n_bins + 2,), dtype=np.long), zeros((self.n_bins + 2,), dtype=np.double))
            self.results["mult_prec_{}".format(self.class_names[i])] = \
                (zeros((self.n_mult + 2,), dtype=np.double), zeros((self.n_mult + 2,), dtype=np.long), zeros((self.n_mult + 2,), dtype=np.double))

    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.detach().cpu().numpy(), f.detach().cpu().numpy(), \
                                            labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), \
                                            output.detach().cpu().numpy()

        avg_coo, summed_pulses, output_stats, multiplicity, psdl, psdr, n_SE = zeros((predictions.shape[0], 2)), \
                                                                         zeros((predictions.shape[0],
                                                                                f.shape[1],),
                                                                               dtype=np.float32), \
                                                                         zeros((6, predictions.shape[0]),
                                                                               dtype=np.float32), \
                                                                         zeros((predictions.shape[0],),
                                                                               dtype=np.int32), \
                                                                         zeros((predictions.shape[0],),
                                                                               dtype=np.float32), \
                                                                         zeros((predictions.shape[0],),
                                                                               dtype=np.float32), \
                                                                        zeros((predictions.shape[0],),
                                                                              dtype=np.int32)
        average_pulse(c, f, self.gain_factor, np.arange(0.5, self.n_samples - 0.49, 1.0), avg_coo, summed_pulses,
                      output_stats, multiplicity, psdl, psdr, n_SE, self.seg_status)

        if self.summed_waveforms is None:
            self.summed_waveforms = np.zeros((self.n_classes + 1, summed_pulses[1].size), np.float32)
            self.summed_labelled_waveforms = \
                np.zeros((self.n_classes, summed_pulses[1].size), np.float32)
        self.n_wfs[0] += np.sum(multiplicity)
        self.summed_waveforms[0] += np.sum(summed_pulses, axis=0)
        energy = np.sum(summed_pulses, axis=1) * 0.5
        # print("first 10 energy: {}".format(energy[0:10]))

        ene_bins = get_bins(*self.default_bins[self.E_index])
        psd_bins = get_bins(*self.default_bins[self.PSD_index])
        mult_bins = np.arange(0, 21, 1)
        self.logger.experiment.add_histogram("evaluation/energy", energy, 0, max_bins=self.n_bins, bins=ene_bins)
        feature_list = [energy, psdl, psdr, multiplicity]
        feature_names = ["energy", "psd", "psd", "multiplicity"]
        bins_list = [ene_bins, psd_bins, psd_bins, mult_bins]
        missing_classes = False
        results = find_matches(predictions, labels, zeros((predictions.shape[0],)))
        for i in range(self.n_classes):
            label_class_inds = list_matches(labels, i)
            preds_class_inds = list_matches(predictions, i)
            if len(label_class_inds) == 0:
                print("warning, no data found for class {}".format(self.class_names[i]))
                missing_classes = True
                continue
            # todo, combine psdl and psdr before adding here
            self.metric_pairs.add(results[label_class_inds], np.concatenate((np.expand_dims(energy[label_class_inds],
                                                                                            axis=0),
                                                                             np.expand_dims(psdl[label_class_inds],
                                                                                            axis=0), np.expand_dims(
                multiplicity[label_class_inds], axis=0), output_stats[:, label_class_inds].squeeze()), axis=0),
                                  self.class_names[i])
            missing_classes = self.accumulate_class_data_with_inds(i, label_class_inds, preds_class_inds, feature_list,
                                                                   feature_names, bins_list)
            self.logger.experiment.add_histogram("evaluation/output_{}".format(self.class_names[i]), output[:, i], 0,
                                                 max_bins=self.n_bins, bins='fd')
            if len(label_class_inds) > 0:
                self.n_wfs[i + 1] += np.sum(multiplicity[label_class_inds])
                self.summed_waveforms[i + 1] += np.sum(summed_pulses[label_class_inds], axis=0)
            if len(preds_class_inds) > 0:
                self.n_labelled_wfs[i] += np.sum(multiplicity[preds_class_inds])
                self.summed_labelled_waveforms[i] += np.sum(summed_pulses[preds_class_inds], axis=0)

        """
        if not missing_classes:
            this_roc = self.roc(output, labels, num_classes=self.n_classes)
            this_prc = self.pr(output, labels, num_classes=self.n_classes)
            self.logger.experiment.add_figure("evaluation/roc", plot_roc(this_roc, self.class_names))
            self.logger.experiment.add_figure("evaluation/precision_recall", plot_pr(this_prc, self.class_names))
        """

        metric_accumulate_1d(results, multiplicity, *self.results["mult_acc"], get_typed_list([0.5, self.n_mult + 0.5]),
                             self.n_mult)
        """
        print("energy sample: {}".format(energy[0:100]))
        print("psd sample: {}".format(psd[0:100]))
        print("maximum en is ", np.amax(energy))
        print("maximum psd is ", np.amax(psd))
        print("minimum en is ", np.amin(energy))
        print("minimum psd is ", np.amin(psd))
        """
        confusion_accumulate_1d(predictions, labels, energy, self.results["confusion_energy"],
                                get_typed_list([0.0, self.emax]),
                                self.n_confusion)
        confusion_accumulate_1d(predictions, labels, n_SE, self.results["confusion_SE"],
                                get_typed_list([-0.5, self.n_SE_max + 0.5]),
                                self.n_SE_max + 1)
        metric_accumulate_2d(results, np.stack((energy, psdl), axis=1), *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, np.stack((energy, psdr), axis=1), *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, avg_coo, *self.results["pos_acc"], get_typed_list([0.0, float(self.nx)]),
                             get_typed_list([0.0, float(self.ny)]), self.nx, self.ny)

    def dump(self):
        self.finalize()
        self.logger.experiment.add_figure("evaluation/energy_psd_accuracy",
                                          plot_contour(calc_axis(self.emin, self.emax, self.n_bins),
                                                       calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                       safe_divide(self.results["ene_psd_acc"][0][1:self.n_bins + 1,
                                                                   1:self.n_bins + 1],
                                                                   self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                                   1:self.n_bins + 1]),
                                                       self.ene_label, "psd", "accuracy"))
        self.logger.experiment.add_figure("evaluation/position_accuracy",
                                          plot_contour(np.arange(1, self.nx + 1, 1), np.arange(1, self.ny + 1, 1),
                                                       safe_divide(
                                                           self.results["pos_acc"][0][1:self.nx + 1, 1:self.ny + 1],
                                                           self.results["pos_acc"][1][1:self.nx + 1, 1:self.ny + 1]),
                                                       "x", "y", "accuracy", filled=False))
        self.logger.experiment.add_figure("evaluation/multiplicity_accuracy",
                                          plot_bar(np.arange(1, self.n_mult + 1),
                                                   self.results["mult_acc"][0][1:self.n_mult + 1],
                                                   "multiplicity",
                                                   "accuracy"))
        # print("n_wfs  is {0}".format(self.n_wfs))
        # print("summed waveforms shape is {0}".format(self.summed_waveforms))
        self.logger.experiment.add_figure("evaluation/average_pulses",
                                          plot_wfs(self.summed_waveforms[1:], self.n_wfs[1:], self.class_names))
        self.logger.experiment.add_figure("evaluation/average_pulses_normalized",
                                          plot_wfs(self.summed_waveforms[1:], self.n_wfs[1:], self.class_names,
                                                   normalize=True, write_pulses=True))
        self.logger.experiment.add_figure("evaluation/average_pulses_labelled",
                                          plot_wfs(self.summed_labelled_waveforms, self.n_labelled_wfs,
                                                   self.class_names))
        self.logger.experiment.add_figure("evaluation/pulse",
                                          plot_wfs(np.expand_dims(self.summed_waveforms[0], axis=0), [self.n_wfs[0]],
                                                   ["total"], plot_errors=True))
        for i in range(self.n_confusion):
            bin_width = self.emax / self.n_confusion
            title = "{0:.1f} - {1:.1f} MeV".format(i * bin_width, (i + 1) * bin_width)
            self.logger.experiment.add_figure("evaluation/confusion_matrix_energy{0}".format(i),
                                              plot_confusion_matrix(self.results["confusion_energy"][i],
                                                                    self.class_names,
                                                                    normalize=True, title=title))
        for i in range(self.n_SE_max + 1):
            title = "{} SE segs".format(i)
            self.logger.experiment.add_figure("evaluation/confusion_matrix_SE_{0}".format(i),
                                              plot_confusion_matrix(self.results["confusion_SE"][i],
                                                                    self.class_names,
                                                                    normalize=True, title=title))
            self.logger.experiment.add_figure("evaluation/confusion_matrix_SE_{0}_totals".format(i),
                                              plot_confusion_matrix(self.results["confusion_SE"][i],
                                                                    self.class_names,
                                                                    normalize=False, title=title))
        self.metric_pairs.plot(self.logger)
        self._init_results()

    def accumulate_class_data(self, i, labels, predictions, feature_list, feature_names, bins_list):
        label_class_inds = list_matches(labels, i)
        preds_class_inds = list_matches(predictions, i)
        if len(label_class_inds) == 0:
            print("warning, no data found for class {}".format(self.class_names[i]))
            return True
        return self.accumulate_class_data_with_inds(i, label_class_inds, preds_class_inds, feature_list, feature_names,
                                                    bins_list)

    def accumulate_class_data_with_inds(self, i, label_class_inds, preds_class_inds, feature_list, feature_names,
                                        bins_list):
        """
        i is label
        labels is list of actual labels
        predictions is list of predicted labels
        feature_list is list of features (arbitrary shape np array) in same ordering as labels/predictions
        bins_list is list of 1d np arrays of bin edges (including 0th and N-1th bin edges)

        returns: whether or not there are missing classes in either labels or predictions
        """
        missing_classes = False
        for feat, bins, featname in zip(feature_list, bins_list, feature_names):
            if len(label_class_inds[0]) == 0:
                missing_classes = True
                #print("warning, no data found for class {}".format(self.class_names[i]))
            else:
                self.logger.experiment.add_histogram("evaluation/{0}_{1}".format(featname, self.class_names[i]),
                                                     feat[label_class_inds], 0, max_bins=len(bins) - 1, bins=bins)
            if len(preds_class_inds[0]) == 0:
                missing_classes = True
                #print("warning, no data found for class {}".format(self.class_names[i]))
                continue
            self.logger.experiment.add_histogram(
                "evaluation/{0}_labelled_{1}".format(featname, self.class_names[i]),
                feat[preds_class_inds], 0, bins=bins, max_bins=len(bins) - 1)
        return missing_classes

    def finalize(self):
        if self.is_finalized:
            return
        self.is_finalized = True
        finalize(*self.results["ene_acc"])
        finalize(*self.results["mult_acc"])
        for i in range(len(self.class_names)):
            finalize(*self.results["ene_prec_{}".format(self.class_names[i])])
            finalize(*self.results["mult_prec_{}".format(self.class_names[i])])

class PhysEvaluator(PSDEvaluator):
    """for phys pulse training, 7 features in order:
    definition:
    vs[0] = p.E / 12.;
    vs[1] = (p.dt / 200.) + 0.5;
    vs[2] = p.PE[0] / 5000.;
    vs[3] = p.PE[1] / 5000.;
    vs[4] = (p.z / 1200.0) + 0.5;
    vs[5] = p.PSD;
    vs[6] = ((Float_t)(p.t - toffset)) / 30.;
    """

    def __init__(self, class_names, logger, device):
        super(PhysEvaluator, self).__init__(class_names, logger, device)
        self.ene_label = "Visible Energy [MeV]"
        self.emax = 10.
        self.is_finalized = False

    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.detach().cpu().numpy(), f.detach().cpu().numpy(), labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), output.detach().cpu().numpy()
        ene_bins = get_bins(*self.default_bins[self.E_index])
        psd_bins = get_bins(*self.default_bins[self.PSD_index])
        PE_bins = get_bins(*self.default_bins[self.PE0_index])
        z_bins = get_bins(*self.default_bins[self.z_index])
        dt_bins = get_bins(*self.default_bins[self.dt_index])
        t0_bins = get_bins(*self.default_bins[self.toffset_index])
        energy = f[:, 0] * 12.
        dt = (f[:, 1] - 0.5) * 30.
        PEL = f[:, 2] * 5000.
        PER = f[:, 3] * 5000.
        z = (f[:, 4] - 0.5) * 1200.
        psd = f[:, 5]
        t0 = f[:, 6] * 30.0
        self.logger.experiment.add_histogram("evaluation/energy", f[:, 0] * 12., 0, max_bins=self.n_bins,
                                             bins=ene_bins)
        missing_classes = False
        full_feature_list = np.stack((energy, psd, dt, PEL, PER, z, t0), axis=0)
        feature_names = ["energy", "psd", "rise_time", "PE", "PE", "z", "start_time"]
        feature_list = zeros((full_feature_list.shape[0], predictions.shape[0]), dtype=np.float32)
        avg_coo, feature_list, mult = weighted_average_quantities(c, full_feature_list, feature_list,
                                                                  zeros((predictions.shape[0], 2)),
                                                                  zeros((predictions.shape[0],)),
                                                                  full_feature_list.shape[0])
        feature_list = np.stack((feature_list[0], feature_list[1], feature_list[2], feature_list[3], feature_list[4],
                                 feature_list[5], feature_list[6], mult), axis=0)
        feature_names.append("multiplicity")
        bins_list = [ene_bins, psd_bins, dt_bins, PE_bins, PE_bins, z_bins, t0_bins, np.arange(0, 21, 1)]
        results = find_matches(predictions, labels, zeros((predictions.shape[0],)))
        for i in range(self.n_classes):
            label_class_inds = list_matches(labels, i)
            preds_class_inds = list_matches(predictions, i)
            if len(label_class_inds) == 0:
                print("warning, no data found for class {}".format(self.class_names[i]))
                missing_classes = True
                continue
            missing_classes = self.accumulate_class_data_with_inds(i, label_class_inds, preds_class_inds, feature_list,
                                                                   feature_names, bins_list)
            self.logger.experiment.add_histogram("evaluation/output_{}".format(self.class_names[i]), output[:, i], 0,
                                                 max_bins=self.n_bins, bins='fd')
            metric_accumulate_2d(results[label_class_inds],
                                 np.stack((feature_list[0][label_class_inds], feature_list[1][label_class_inds]),
                                          axis=1),
                                 *self.results["ene_psd_prec_{}".format(self.class_names[i])],
                                 get_typed_list([self.emin, self.emax]),
                                 get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
            metric_accumulate_1d(results[label_class_inds],
                                 feature_list[0][label_class_inds],
                                 *self.results["ene_prec_{}".format(self.class_names[i])],
                                 get_typed_list([self.emin, self.emax]),
                                 self.n_bins)
            metric_accumulate_1d(results[label_class_inds],
                                 mult[label_class_inds],
                                 *self.results["mult_prec_{}".format(self.class_names[i])],
                                 get_typed_list([0.5, self.n_mult + 0.5]),
                                 self.n_mult)

        """
        if not missing_classes:
            this_roc = self.roc(output, labels, num_classes=self.n_classes)
            this_prc = self.pr(output, labels, num_classes=self.n_classes)
            self.logger.experiment.add_figure("evaluation/roc", plot_roc(this_roc, self.class_names))
            self.logger.experiment.add_figure("evaluation/precision_recall", plot_pr(this_prc, self.class_names))
        """

        confusion_accumulate_1d(predictions, labels, feature_list[0], self.results["confusion_energy"],
                                get_typed_list([0.0, self.emax]),
                                self.n_confusion)
        metric_accumulate_1d(results, mult, *self.results["mult_acc"],
                             get_typed_list([0.5, self.n_mult + 0.5]),
                             self.n_mult)
        metric_accumulate_2d(results, np.stack((feature_list[0], feature_list[1]), axis=1),
                             *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, avg_coo, *self.results["pos_acc"], get_typed_list([0.0, float(self.nx)]),
                             get_typed_list([0.0, float(self.ny)]), self.nx, self.ny)


    def dump(self):
        self.finalize()
        self.logger.experiment.add_figure("evaluation/energy_psd_accuracy",
                                          plot_contour(calc_axis(self.emin, self.emax, self.n_bins),
                                                       calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                       safe_divide(self.results["ene_psd_acc"][0][1:self.n_bins + 1,
                                                                   1:self.n_bins + 1],
                                                                   self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                                   1:self.n_bins + 1]),
                                                       "energy [MeV]", "psd", "accuracy"))
        self.logger.experiment.add_figure("evaluation/position_accuracy",
                                          plot_contour(np.arange(1, self.nx + 1, 1), np.arange(1, self.ny + 1, 1),
                                                       safe_divide(
                                                           self.results["pos_acc"][0][1:self.nx + 1, 1:self.ny + 1],
                                                           self.results["pos_acc"][1][1:self.nx + 1, 1:self.ny + 1]),
                                                       "x", "y", "accuracy", filled=False))
        self.logger.experiment.add_figure("evaluation/multiplicity_accuracy",
                                          plot_bar(np.arange(1, self.n_mult + 1),
                                                   self.results["mult_acc"][0][1:self.n_mult + 1],
                                                   "multiplicity",
                                                   "accuracy"))
        xwidth = (self.emax - self.emin) / self.n_bins
        xedges = np.arange(self.emin, self.emax + xwidth, xwidth)
        ywidth = (self.psd_max - self.psd_min) / self.n_bins
        yedges = np.arange(self.psd_min, self.psd_max + ywidth, ywidth)
        self.logger.experiment.add_figure("evaluation/EPSD",
                                          plot_hist2d(xedges, yedges,
                                                      self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                      1:self.n_bins + 1],
                                                      "Total", "Energy [MeV]", "PSD",
                                                      r'# Pulses [$MeV^{-1}PSD^{-1}$'))

        self.logger.experiment.add_figure("evaluation/multiplicity",
                                          plot_hist1d(calc_bin_edges(0.5, self.n_mult + 0.5, self.n_mult),
                                                      self.results["mult_acc"][1][1:self.n_mult + 1],
                                                      "Total", "Multiplicity", False))

        self.logger.experiment.add_figure("evaluation/EPSD_classes",
                                          plot_n_hist2d(xedges, yedges,
                                                        [self.results["ene_psd_prec_{}".format(self.class_names[i])][
                                                             1][1:self.n_bins + 1, 1:self.n_bins + 1] for i in
                                                         range(len(self.class_names))],
                                                        self.class_names,
                                                        "Energy [MeV]", "PSD"))

        self.logger.experiment.add_figure("evaluation/energy_psd_precision",
                                          plot_n_contour(calc_axis(self.emin, self.emax, self.n_bins),
                                                         calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                         [safe_divide(self.results["ene_psd_prec_{}".format(
                                                             self.class_names[i])][0][1:self.n_bins + 1,
                                                                      1:self.n_bins + 1],
                                                                      self.results["ene_psd_prec_{}".format(
                                                                          self.class_names[i])][1][1:self.n_bins + 1,
                                                                      1:self.n_bins + 1]) for i in
                                                          range(len(self.class_names))],
                                                         "Energy [MeV]", "PSD", self.class_names, cm=plt.cm.cividis))

        self.logger.experiment.add_figure("evaluation/energy_precision",
                                          plot_n_hist1d(calc_bin_edges(self.emin, self.emax, self.n_bins),
                                                        [self.results["ene_prec_{}".format(
                                                            self.class_names[i])][0][1:self.n_bins + 1]
                                                         for i in range(len(self.class_names))],
                                                        self.class_names, self.ene_label, "precision",
                                                        norm_to_bin_width=False, logy=False))
        self.logger.experiment.add_figure("evaluation/multiplicity_precision",
                                          plot_n_hist1d(calc_bin_edges(0.5, self.n_mult + 0.5, self.n_mult),
                                                        [self.results["mult_prec_{}".format(
                                                            self.class_names[i])][0][1:self.n_mult + 1]
                                                                      for i in range(len(self.class_names))],
                                                        self.class_names, "multiplicity", "precision",
                                                        norm_to_bin_width=False, logy=False))
        self.logger.experiment.add_figure("evaluation/multiplicity_classes",
                                          plot_n_hist1d(calc_bin_edges(0.5, self.n_mult + 0.5, self.n_mult),
                                                        [self.results["mult_prec_{}".format(
                                                            self.class_names[i])][1][1:self.n_mult + 1]
                                                         for i in range(len(self.class_names))],
                                                        self.class_names, "multiplicity", "total"))

        for i in range(self.n_confusion):
            bin_width = self.emax / self.n_confusion
            title = "{0:.1f} - {1:.1f} MeV".format(i * bin_width, (i + 1) * bin_width)
            self.logger.experiment.add_figure("evaluation/confusion_matrix_energy{0}".format(i),
                                              plot_confusion_matrix(self.results["confusion_energy"][i],
                                                                    self.class_names,
                                                                    normalize=True, title=title))
        self._init_results()
