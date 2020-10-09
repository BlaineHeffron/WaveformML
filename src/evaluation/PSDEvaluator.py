import numpy as np
from src.utils.SparseUtils import average_pulse, find_matches, metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list
from src.utils.PlotUtils import plot_countour, plot_pr, plot_roc, plot_wfs, plot_bar
from src.utils.util import extract_values
from numpy import zeros

from pytorch_lightning.metrics.classification import MulticlassROC, MulticlassPrecisionRecallCurve

def safe_divide(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


class PSDEvaluator:

    def __init__(self, class_names, logger, device):
        self.logger = logger
        self.device = device
        self.n_bins = 40
        self.n_mult = 20
        self.emin = 0.0
        self.emax = 400.0
        self.psd_min = 0.0
        self.psd_max = 1.0
        self.nx = 14
        self.ny = 11
        self.class_names = class_names
        self.n_classes = len(self.class_names)
        self.roc = MulticlassROC(num_classes=self.n_classes)
        self.pr = MulticlassPrecisionRecallCurve(num_classes=self.n_classes)
        self.summed_waveforms = None
        self.n_wfs = [0]*(self.n_classes+1)
        self.n_labelled_wfs = [0]*self.n_classes
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


    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.detach().cpu().numpy(), f.detach().cpu().numpy(), labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), output.detach().cpu().numpy()
        # print("first several coords: {}".format(c[0:100]))
        # print("first several pulses: {}".format(f[0:100]))
        avg_coo, summed_pulses, multiplicity, psd = average_pulse(c, f,
                                                                  zeros((predictions.shape[0], 2)),
                                                                  zeros((predictions.shape[0], f.shape[1],), dtype=np.float32),
                                                                  zeros((predictions.shape[0],), dtype=np.int32),
                                                                  zeros((predictions.shape[0],)))
        # print("first 10 avg coo: {}".format(avg_coo[0:10]))
        # print("first 10 summed_pulses: {}".format(summed_pulses[0:10]))
        # print("first 10 multiplicity: {}".format(multiplicity[0:10]))
        # print("first 10 psd: {}".format(psd[0:10]))
        if self.summed_waveforms is None:
            self.summed_waveforms = np.zeros((self.n_classes+1, summed_pulses[1].size), np.float32)
            self.summed_labelled_waveforms = \
                np.zeros((self.n_classes, summed_pulses[1].size), np.float32)
        self.n_wfs[0] += np.sum(multiplicity)
        self.summed_waveforms[0] += np.sum(summed_pulses, axis=0)
        energy = np.sum(summed_pulses, axis=1)
        # print("first 10 energy: {}".format(energy[0:10]))
        ene_bins = np.arange(self.emin,self.emax,10)
        psd_bins = np.arange(0.0,1.0,0.025)
        self.logger.experiment.add_histogram("evaluation/energy", energy, 0, max_bins=self.n_bins,bins=ene_bins)
        missing_classes = False
        for i in range(self.n_classes):
            vals = extract_values(energy, labels, i)
            if vals.size == 0:
                missing_classes = True
                print("warning, no data found for class {}".format(self.class_names[i]))
                continue
            mult = extract_values(multiplicity,labels,i)
            multpred = extract_values(multiplicity,predictions,i)
            self.logger.experiment.add_histogram("evaluation/energy_{}".format(self.class_names[i]),
                                                 vals, 0, bins=ene_bins)
            self.logger.experiment.add_histogram("evaluation/psd_{}".format(self.class_names[i]),
                                                 extract_values(psd, labels, i), 0, bins=psd_bins)
            self.logger.experiment.add_histogram("evaluation/energy_labelled_{}".format(self.class_names[i]),
                                                 extract_values(energy, predictions, i), 0, bins=ene_bins)
            self.logger.experiment.add_histogram("evaluation/psd_labelled_{}".format(self.class_names[i]),
                                                 extract_values(psd, predictions, i), 0, bins=psd_bins)
            self.logger.experiment.add_histogram("evaluation/multiplicity_{}".format(self.class_names[i]),
                                                 mult, 0,
                                                 bins=np.arange(0.5, self.n_mult + 0.5, 1))
            self.logger.experiment.add_histogram("evaluation/multiplicity_labelled_{}".format(self.class_names[i]),
                                                 multpred, 0,
                                                 bins=np.arange(0.5, self.n_mult + 0.5, 1))
            self.logger.experiment.add_histogram("evaluation/output_{}".format(self.class_names[i]), output[:, i], 0,
                                                 max_bins=self.n_bins, bins='fd')
            pulses = extract_values(summed_pulses, labels, i)
            self.n_wfs[i+1] += np.sum(mult)
            self.n_labelled_wfs[i] += np.sum(multpred)
            self.summed_waveforms[i+1] += np.sum(pulses, axis=0)
            self.summed_labelled_waveforms[i] += np.sum(extract_values(summed_pulses, predictions, i), axis=0)

        if not missing_classes:
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
                                                        safe_divide(self.results["ene_psd_acc"][0][1:self.n_bins + 1,1:self.n_bins + 1],
                                                                    self.results["ene_psd_acc"][1][1:self.n_bins + 1, 1:self.n_bins + 1]),
                                                        "energy [arb]", "psd", "accuracy"))
        self.logger.experiment.add_figure("evaluation/position_accuracy",
                                          plot_countour(np.arange(1, self.nx + 1, 1), np.arange(1, self.ny + 1, 1),
                                                        safe_divide(self.results["pos_acc"][0][1:self.nx + 1, 1:self.ny + 1],
                                                                    self.results["pos_acc"][1][1:self.nx + 1, 1:self.ny + 1]),
                                                        "x", "y", "accuracy"))
        self.logger.experiment.add_figure("evaluation/multiplicity_accuracy",
                                          plot_bar(np.arange(1, self.n_mult + 1),
                                                   safe_divide(self.results["mult_acc"][0][1:self.n_mult + 1],
                                                               self.results["mult_acc"][1][1:self.n_mult + 1]),
                                                   "multiplicity",
                                                   "accuracy"))

        #print("n_wfs  is {0}".format(self.n_wfs))
        #print("summed waveforms shape is {0}".format(self.summed_waveforms))
        self.logger.experiment.add_figure("evaluation/average_pulses",
                                          plot_wfs(self.summed_waveforms[1:], self.n_wfs[1:], self.class_names))
        self.logger.experiment.add_figure("evaluation/average_pulses_labelled",
                                          plot_wfs(self.summed_labelled_waveforms, self.n_labelled_wfs[1:], self.class_names))
        self.logger.experiment.add_figure("evaluation/pulse",
                                          plot_wfs(np.expand_dims(self.summed_waveforms[0],axis=0), [self.n_wfs[0]],
                                                   ["total"], plot_errors=True))
        self._init_results()

    def calc_axis(self, min, max, n):
        return np.arange(min, max, (max - min) / n) + (max - min) / (2 * n)

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
        self.emax = 20.

    def add(self, batch, output, predictions):
        (c, f), labels = batch
        c, f, labels, predictions, output = c.detach().cpu().numpy(), f.detach().cpu().numpy(), labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), output.detach().cpu().numpy()
        ene_bins = np.arange(0,20,0.05)
        psd_bins = np.arange(0.0,1.0,0.025)
        PE_bins = np.arange(0,500,10)
        z_bins = np.arange(-1000,1000,10)
        dt_bins = np.arange(-90.,90.,10)
        t0_bins = np.arange(0.,500.,10)
        energy = f[:,0]*300.
        dt = (f[:,1] - 0.5)*200.
        PEL = f[:,2]*125000.
        PER = f[:,3]*125000.
        z = (f[:,4] - 0.5)*2200.
        psd = f[:,5]
        t0 = f[:,6]*600.0
        self.logger.experiment.add_histogram("evaluation/energy", f[:,0]*300., 0, max_bins=self.n_bins,bins=ene_bins)
        missing_classes = False
        for i in range(self.n_classes):
            vals = extract_values(energy, labels, i)
            if vals.size == 0:
                missing_classes = True
                print("warning, no data found for class {}".format(self.class_names[i]))
                continue
            try:
                self.logger.experiment.add_histogram("evaluation/energy_{}".format(self.class_names[i]),
                                                     vals, 0, bins=ene_bins)
                self.logger.experiment.add_histogram("evaluation/psd_{}".format(self.class_names[i]),
                                                     extract_values(psd, labels, i), 0, bins=psd_bins)
                self.logger.experiment.add_histogram("evaluation/energy_labelled_{}".format(self.class_names[i]),
                                                     extract_values(energy, predictions, i), 0, bins=ene_bins)
                self.logger.experiment.add_histogram("evaluation/psd_labelled_{}".format(self.class_names[i]),
                                                     extract_values(psd, predictions, i), 0, bins=psd_bins)
                self.logger.experiment.add_histogram("evaluation/PEL_{}".format(self.class_names[i]),
                                                     extract_values(PEL, labels, i), 0, bins=PE_bins)
                self.logger.experiment.add_histogram("evaluation/PER_{}".format(self.class_names[i]),
                                                     extract_values(PER, labels, i), 0, bins=PE_bins)
                self.logger.experiment.add_histogram("evaluation/dt_{}".format(self.class_names[i]),
                                                     extract_values(dt, labels, i), 0, bins=dt_bins)
                self.logger.experiment.add_histogram("evaluation/z_{}".format(self.class_names[i]),
                                                     extract_values(z, labels, i), 0, bins=z_bins)
                self.logger.experiment.add_histogram("evaluation/t0_{}".format(self.class_names[i]),
                                                     extract_values(t0, labels, i), 0, bins=t0_bins)
                self.logger.experiment.add_histogram("evaluation/PEL_labelled_{}".format(self.class_names[i]),
                                                     extract_values(PEL, predictions, i), 0, bins=PE_bins)
                self.logger.experiment.add_histogram("evaluation/PER_labelled_{}".format(self.class_names[i]),
                                                     extract_values(PER, predictions, i), 0, bins=PE_bins)
                self.logger.experiment.add_histogram("evaluation/dt_labelled_{}".format(self.class_names[i]),
                                                     extract_values(dt, predictions, i), 0, bins=dt_bins)
                self.logger.experiment.add_histogram("evaluation/z_labelled_{}".format(self.class_names[i]),
                                                     extract_values(z, predictions, i), 0, bins=z_bins)
                self.logger.experiment.add_histogram("evaluation/t0_labelled_{}".format(self.class_names[i]),
                                                     extract_values(t0, predictions, i), 0, bins=t0_bins)
                self.logger.experiment.add_histogram("evaluation/output_{}".format(self.class_names[i]), output[:, i], 0,
                                                     max_bins=self.n_bins, bins='fd')
            except ValueError as e:
                print(e)

        if not missing_classes:
            this_roc = self.roc(output, labels)
            this_prc = self.pr(output, labels)
            self.logger.experiment.add_figure("evaluation/roc", plot_roc(this_roc, self.class_names))
            self.logger.experiment.add_figure("evaluation/precision_recall", plot_pr(this_prc, self.class_names))

        results = find_matches(predictions, labels, zeros((predictions.shape[0],)))

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

    def dump(self):
        self.logger.experiment.add_figure("evaluation/energy_psd_accuracy",
                                          plot_countour(self.calc_axis(self.emin, self.emax, self.n_bins),
                                                        self.calc_axis(self.psd_min, self.psd_max, self.n_bins),
                                                        safe_divide(self.results["ene_psd_acc"][0][1:self.n_bins + 1,1:self.n_bins + 1],
                                                                  self.results["ene_psd_acc"][1][1:self.n_bins + 1, 1:self.n_bins + 1]),
                                                        "energy [arb]", "psd", "accuracy"))
        self._init_results()

    def calc_axis(self, min, max, n):
        return np.arange(min, max, (max - min) / n) + (max - min) / (2 * n)
