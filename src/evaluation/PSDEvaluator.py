import numpy as np
from src.utils.SparseUtils import average_pulse, find_matches, metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list
from src.utils.PlotUtils import plot_countour, plot_bar, plot_pr, plot_roc
from src.utils.util import extract_values
from numpy import zeros

from pytorch_lightning.metrics.classification import MulticlassROC, MulticlassPrecisionRecall


class PSDEvaluator:

    def __init__(self, class_names, logger, device):
        self.logger = logger
        self.device = device
        self.n_bins = 40
        self.n_mult = 12
        self.emin = 0.0
        self.emax = 5.0
        self.psd_min = 0.0
        self.psd_max = 1.0
        self.nx = 14
        self.ny = 11
        self.class_names = class_names
        self.n_classes = len(self.class_names)
        self.roc = MulticlassROC(num_classes=self.n_classes)
        self.pr = MulticlassPrecisionRecall(num_classes=self.n_classes)
        self._init_results()

    def _init_results(self):
        self.results = {
            "mult_acc": (zeros((self.n_mult + 2,), dtype=np.float32), zeros((self.n_mult + 2,), dtype=np.int32)),
            "pos_acc": (
                zeros((self.nx + 2, self.ny + 2), dtype=np.float32), zeros((self.nx + 2, self.ny + 2), dtype=np.int32)),
            "ene_psd_acc": (zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.float32),
                            zeros((self.n_bins + 2, self.n_bins + 2), dtype=np.int32)),
        }

    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.clone().detach().numpy(), f.clone().detach().numpy(), labels.clone().detach().numpy(), predictions.clone().detach().numpy(), output.clone().detach().numpy()
        # print("first several coords: {}".format(c[0:100]))
        # print("first several pulses: {}".format(f[0:100]))
        avg_coo, summed_pulses, multiplicity, psd = average_pulse(c, f,
                                                                  zeros((predictions.shape[0], 2)),
                                                                  zeros((predictions.shape[0], f.shape[1])),
                                                                  zeros((predictions.shape[0],), dtype=np.int32),
                                                                  zeros((predictions.shape[0],)))
        # print("first 10 avg coo: {}".format(avg_coo[0:10]))
        # print("first 10 summed_pulses: {}".format(summed_pulses[0:10]))
        # print("first 10 multiplicity: {}".format(multiplicity[0:10]))
        # print("first 10 psd: {}".format(psd[0:10]))
        # self.logger.experiment.add_histogram("evaluation/timing", summed_pulses, max_bins=summed_pulses[0].size)
        energy = np.sum(summed_pulses, axis=1)
        # print("first 10 energy: {}".format(energy[0:10]))
        self.logger.experiment.add_histogram("evaluation/energy", energy, max_bins=self.n_bins)
        for i in range(self.n_classes):
            vals = extract_values(energy, labels, i)
            if vals.size == 0:
                print("warning, no data found for class {}".format(self.class_names[i]))
                continue
            self.logger.experiment.add_histogram("evaluation/energy_{}".format(self.class_names[i]),
                                                 vals)
            self.logger.experiment.add_histogram("evaluation/psd_{}".format(self.class_names[i]),
                                                 extract_values(psd, labels, i))
            self.logger.experiment.add_histogram("evaluation/multiplicity_{}".format(self.class_names[i]),
                                                 extract_values(multiplicity, labels, i),
                                                 bins=np.arange(0.5, self.n_mult + 0.5, 1))
            self.logger.experiment.add_histogram("evaluation/output_{}".format(self.class_names[i]), output[:, i],
                                                 max_bins=self.n_bins)
        this_roc = self.roc(output, labels)
        this_prc = self.pr(output, labels)
        self.logger.experiment.add_figure("evaluation/roc", plot_roc(this_roc, self.class_names))
        self.logger.experiment.add_figure("evaluation/precision_recall", plot_pr(this_prc, self.class_names))

        results = find_matches(predictions, labels, zeros((predictions.shape[0],)))

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
        metric_accumulate_2d(results, np.stack((energy, psd), axis=1), *self.results["ene_psd_acc"],
                             get_typed_list([self.emin, self.emax]),
                             get_typed_list([self.psd_min, self.psd_max]), self.n_bins, self.n_bins)
        metric_accumulate_2d(results, avg_coo, *self.results["pos_acc"], get_typed_list([0.0, float(self.nx)]),
                             get_typed_list([0.0, float(self.ny)]), self.nx, self.ny)

    def dump(self):
        self.logger.experiment.add_figure("evaluation/energy_psd_accuracy",
                                          plot_countour(self.calc_axis(self.emin, self.emax, self.n_bins),
                                                        self.calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                        self.results["ene_psd_acc"][0][1:self.n_bins + 1,
                                                        1:self.n_bins + 1] / self.results["ene_psd_acc"][1][
                                                                             1:self.n_bins + 1, 1:self.n_bins + 1],
                                                        "energy [arb]", "psd", "accuracy"))
        self.logger.experiment.add_figure("evaluation/position_accuracy",
                                          plot_countour(np.arange(1, self.nx + 1, 1), np.arange(1, self.ny + 1, 1),
                                                        self.results["pos_acc"][0][1:self.nx + 1, 1:self.ny + 1] /
                                                        self.results["pos_acc"][1][1:self.nx + 1, 1:self.ny + 1],
                                                        "x", "y", "accuracy"))
        self.logger.experiment.add_figure("evaluation/multiplicity_accuracy",
                                          plot_bar(np.arange(1, self.n_mult + 1),
                                                   self.results["mult_acc"][0][1:self.n_mult + 1] /
                                                   self.results["mult_acc"][1][1:self.n_mult + 1], "multiplicity",
                                                   "accuracy"))
        self._init_results()

    def calc_axis(self, min, max, n):
        return np.arange(min, max, (max - min) / n) + (max - min) / (2 * n)
