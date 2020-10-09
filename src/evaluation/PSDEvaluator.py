import numpy as np
from src.utils.SparseUtils import average_pulse, find_matches, metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list, weighted_average_quantities
from src.utils.PlotUtils import plot_countour, plot_pr, plot_roc, plot_wfs, plot_bar, plot_hist2d
from src.utils.util import list_matches, safe_divide, get_bins
from numpy import zeros

from pytorch_lightning.metrics.classification import MulticlassROC, MulticlassPrecisionRecallCurve


class PSDEvaluator:

    def __init__(self, class_names, logger, device):
        self.logger = logger
        self.device = device
        self.n_bins = 100
        self.n_mult = 20
        self.emin = 0.0
        self.emax = 100.0
        self.psd_min = 0.0
        self.psd_max = 1.0
        self.nx = 14
        self.ny = 11
        self.class_names = class_names
        self.n_classes = len(self.class_names)
        self.roc = MulticlassROC(num_classes=self.n_classes)
        self.pr = MulticlassPrecisionRecallCurve(num_classes=self.n_classes)
        self.summed_waveforms = None
        self.n_wfs = [0] * (self.n_classes + 1)
        self.n_labelled_wfs = [0] * self.n_classes
        self.summed_labelled_waveforms = []
        self._init_results()

    def _init_results(self):
        self.results = {
            "mult_acc": (zeros((self.n_mult + 2,), dtype=np.float32), zeros((self.n_mult + 2,), dtype=np.int32)),
            "pos_acc": (
                zeros((self.nx + 2, self.ny + 2), dtype=np.float32), zeros((self.nx + 2, self.ny + 2), dtype=np.int32)),
            "ene_psd_acc": (zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.float32),
                            zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.int32)),
        }
        for c in self.class_names:
            self.results["ene_psd_prec_{}".format(c)] = (zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.float32),
                                                         zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.int32))

    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.detach().cpu().numpy(), f.detach().cpu().numpy(), \
                                            labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), \
                                            output.detach().cpu().numpy()
        avg_coo, summed_pulses, multiplicity, psdl, psdr = average_pulse(c, f,
                                                                         zeros((predictions.shape[0], 2)),
                                                                         zeros((predictions.shape[0], f.shape[1],),
                                                                               dtype=np.float32),
                                                                         zeros((predictions.shape[0],), dtype=np.int32),
                                                                         zeros((predictions.shape[0],),
                                                                               dtype=np.float32),
                                                                         zeros((predictions.shape[0],),
                                                                               dtype=np.float32))
        if self.summed_waveforms is None:
            self.summed_waveforms = np.zeros((self.n_classes + 1, summed_pulses[1].size), np.float32)
            self.summed_labelled_waveforms = \
                np.zeros((self.n_classes, summed_pulses[1].size), np.float32)
        self.n_wfs[0] += np.sum(multiplicity)
        self.summed_waveforms[0] += np.sum(summed_pulses, axis=0)
        energy = np.sum(summed_pulses, axis=1)
        # print("first 10 energy: {}".format(energy[0:10]))

        ene_bins = get_bins(self.emin, self.emax, self.n_bins)
        psd_bins = get_bins(0., 1., self.n_bins)
        mult_bins = np.arange(0, 20, 1)
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
            metric_accumulate_2d(results[label_class_inds],
                                 np.stack((energy[label_class_inds], psdl[label_class_inds]), axis=1),
                                 *self.results["ene_psd_prec_{}".format(self.class_names[i])],
                                 get_typed_list([self.emin, self.emax]),
                                 get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
            metric_accumulate_2d(results[label_class_inds],
                                 np.stack((energy[label_class_inds], psdr[label_class_inds]), axis=1),
                                 *self.results["ene_psd_prec_{}".format(self.class_names[i])],
                                 get_typed_list([self.emin, self.emax]),
                                 get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)

        if not missing_classes:
            this_roc = self.roc(output, labels)
            this_prc = self.pr(output, labels)
            self.logger.experiment.add_figure("evaluation/roc", plot_roc(this_roc, self.class_names))
            self.logger.experiment.add_figure("evaluation/precision_recall", plot_pr(this_prc, self.class_names))

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
        metric_accumulate_2d(results, np.stack((energy, psdl), axis=1), *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, np.stack((energy, psdr), axis=1), *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, avg_coo, *self.results["pos_acc"], get_typed_list([0.0, float(self.nx)]),
                             get_typed_list([0.0, float(self.ny)]), self.nx, self.ny)

    def dump(self):
        self.logger.experiment.add_figure("evaluation/energy_psd_accuracy",
                                          plot_countour(self.calc_axis(self.emin, self.emax, self.n_bins),
                                                        self.calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                        safe_divide(self.results["ene_psd_acc"][0][1:self.n_bins + 1,
                                                                    1:self.n_bins + 1],
                                                                    self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                                    1:self.n_bins + 1]),
                                                        "energy [arb]", "psd", "accuracy"))
        self.logger.experiment.add_figure("evaluation/position_accuracy",
                                          plot_countour(np.arange(1, self.nx + 1, 1), np.arange(1, self.ny + 1, 1),
                                                        safe_divide(
                                                            self.results["pos_acc"][0][1:self.nx + 1, 1:self.ny + 1],
                                                            self.results["pos_acc"][1][1:self.nx + 1, 1:self.ny + 1]),
                                                        "x", "y", "accuracy"))
        self.logger.experiment.add_figure("evaluation/multiplicity_accuracy",
                                          plot_bar(np.arange(1, self.n_mult + 1),
                                                   safe_divide(self.results["mult_acc"][0][1:self.n_mult + 1],
                                                               self.results["mult_acc"][1][1:self.n_mult + 1]),
                                                   "multiplicity",
                                                   "accuracy"))

        xwidth = (self.emax - self.emin) / self.n_bins
        xedges = np.arange(self.emin, self.emax + xwidth, xwidth)
        ywidth = (1.) / self.n_bins
        yedges = np.arange(0, 1.0 + ywidth, ywidth)
        self.logger.experiment.add_figure("evaluation/EPSD",
                                          plot_hist2d(xedges, yedges,
                                                      self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                      1:self.n_bins + 1],
                                                      "Total", "Energy [arb]", "PSD [arb]"))
        for i in range(len(self.class_names)):
            self.logger.experiment.add_figure("evaluation/EPSD_{}".format(self.class_names[i]),
                                              plot_hist2d(xedges, yedges,
                                                          self.results["ene_psd_prec_{}".format(self.class_names[i])][
                                                              1][1:self.n_bins + 1, 1:self.n_bins + 1],
                                                          self.class_names[i], "Energy [arb]", "PSD [arb]"))
        # print("n_wfs  is {0}".format(self.n_wfs))
        # print("summed waveforms shape is {0}".format(self.summed_waveforms))
        self.logger.experiment.add_figure("evaluation/average_pulses",
                                          plot_wfs(self.summed_waveforms[1:], self.n_wfs[1:], self.class_names))
        self.logger.experiment.add_figure("evaluation/average_pulses_labelled",
                                          plot_wfs(self.summed_labelled_waveforms, self.n_labelled_wfs,
                                                   self.class_names))
        self.logger.experiment.add_figure("evaluation/pulse",
                                          plot_wfs(np.expand_dims(self.summed_waveforms[0], axis=0), [self.n_wfs[0]],
                                                   ["total"], plot_errors=True))
        self._init_results()

    def calc_axis(self, min, max, n):
        return np.arange(min, max, (max - min) / n) + (max - min) / (2 * n)

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
            self.logger.experiment.add_histogram("evaluation/{0}_{1}".format(featname, self.class_names[i]),
                                                 feat[label_class_inds], 0, bins=bins)
            if len(preds_class_inds) == 0:
                missing_classes = True
                print("warning, no data found for class {}".format(self.class_names[i]))
                continue
            self.logger.experiment.add_histogram(
                "evaluation/{0}_labelled_{1}".format(featname, self.class_names[i]),
                feat[preds_class_inds], 0, bins=bins)
        return missing_classes


class PhysEvaluator(PSDEvaluator):
    """for phys pulse training, 7 features in order:
    definition:
    vs[0] = p.E / 300.;
    vs[1] = (p.dt / 200.) + 0.5;
    vs[2] = p.PE[0] / 125000.;
    vs[3] = p.PE[1] / 125000.;
    vs[4] = (p.z / 2200.0) + 0.5;
    vs[5] = p.PSD;
    vs[6] = ((Float_t)(p.t - toffset)) / 600.;
    """

    def __init__(self, class_names, logger, device):
        super(PhysEvaluator, self).__init__(class_names, logger, device)
        self.emax = 10.

    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.detach().cpu().numpy(), f.detach().cpu().numpy(), labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), output.detach().cpu().numpy()
        ene_bins = get_bins(self.emin, self.emax, self.n_bins)
        psd_bins = get_bins(0., 1., self.n_bins)
        PE_bins = np.arange(0, 500, 10)
        z_bins = np.arange(-1000, 1000, 10)
        dt_bins = np.arange(-90., 90., 10)
        t0_bins = np.arange(0., 500., 10)
        energy = f[:, 0] * 300.
        dt = (f[:, 1] - 0.5) * 200.
        PEL = f[:, 2] * 125000.
        PER = f[:, 3] * 125000.
        z = (f[:, 4] - 0.5) * 2200.
        psd = f[:, 5]
        t0 = f[:, 6] * 600.0
        self.logger.experiment.add_histogram("evaluation/energy", f[:, 0] * 300., 0, max_bins=self.n_bins,
                                             bins=ene_bins)
        missing_classes = False
        full_feature_list = np.stack((energy, dt, PEL, PER, z, psd, t0, np.ones((energy.shape[0],))), axis=0)
        feature_names = ["energy", "rise_time", "PE", "PE", "z", "psd", "start_time", "multiplicity"]
        feature_list = zeros((full_feature_list.shape[0], predictions.shape[0]), dtype=np.float32)
        avg_coo, feature_list = weighted_average_quantities(c, full_feature_list, feature_list,
                                                            zeros((predictions.shape[0], 2)), 8)
        bins_list = [ene_bins, dt_bins, PE_bins, PE_bins, z_bins, psd_bins, t0_bins]
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
                                 np.stack((feature_list[0][label_class_inds], feature_list[5][label_class_inds]),
                                          axis=1),
                                 *self.results["ene_psd_prec_{}".format(self.class_names[i])],
                                 get_typed_list([self.emin, self.emax]),
                                 get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)

        if not missing_classes:
            this_roc = self.roc(output, labels)
            this_prc = self.pr(output, labels)
            self.logger.experiment.add_figure("evaluation/roc", plot_roc(this_roc, self.class_names))
            self.logger.experiment.add_figure("evaluation/precision_recall", plot_pr(this_prc, self.class_names))

        metric_accumulate_1d(results, feature_list[7], *self.results["mult_acc"],
                             get_typed_list([0.5, self.n_mult + 0.5]),
                             self.n_mult)
        metric_accumulate_2d(results, np.stack((feature_list[0], feature_list[5]), axis=1),
                             *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, avg_coo, *self.results["pos_acc"], get_typed_list([0.0, float(self.nx)]),
                             get_typed_list([0.0, float(self.ny)]), self.nx, self.ny)

    def dump(self):
        self.logger.experiment.add_figure("evaluation/energy_psd_accuracy",
                                          plot_countour(self.calc_axis(self.emin, self.emax, self.n_bins),
                                                        self.calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                        safe_divide(self.results["ene_psd_acc"][0][1:self.n_bins + 1,
                                                                    1:self.n_bins + 1],
                                                                    self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                                    1:self.n_bins + 1]),
                                                        "energy [arb]", "psd", "accuracy"))
        self.logger.experiment.add_figure("evaluation/position_accuracy",
                                          plot_countour(np.arange(1, self.nx + 1, 1), np.arange(1, self.ny + 1, 1),
                                                        safe_divide(
                                                            self.results["pos_acc"][0][1:self.nx + 1, 1:self.ny + 1],
                                                            self.results["pos_acc"][1][1:self.nx + 1, 1:self.ny + 1]),
                                                        "x", "y", "accuracy"))
        self.logger.experiment.add_figure("evaluation/multiplicity_accuracy",
                                          plot_bar(np.arange(1, self.n_mult + 1),
                                                   safe_divide(self.results["mult_acc"][0][1:self.n_mult + 1],
                                                               self.results["mult_acc"][1][1:self.n_mult + 1]),
                                                   "multiplicity",
                                                   "accuracy"))
        xwidth = (self.emax - self.emin) / self.n_bins
        xedges = np.arange(self.emin, self.emax + xwidth, xwidth)
        ywidth = (1.) / self.n_bins
        yedges = np.arange(0, 1.0 + ywidth, ywidth)
        self.logger.experiment.add_figure("evaluation/EPSD",
                                          plot_hist2d(xedges, yedges,
                                                      self.results["ene_psd_acc"][1][1:self.n_bins + 1,
                                                      1:self.n_bins + 1],
                                                      "Total", "Energy [arb]", "PSD [arb]"))
        for i in range(len(self.class_names)):
            self.logger.experiment.add_figure("evaluation/EPSD_{}".format(self.class_names[i]),
                                              plot_hist2d(xedges, yedges,
                                                          self.results["ene_psd_prec_{}".format(self.class_names[i])][
                                                              1][1:self.n_bins + 1, 1:self.n_bins + 1],
                                                          self.class_names[i], "Energy [arb]", "PSD [arb]"))
        self._init_results()

    def calc_axis(self, min, max, n):
        return np.arange(min, max, (max - min) / n) + (max - min) / (2 * n)
