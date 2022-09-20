from numpy import stack, zeros, swapaxes
from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator
from src.utils.PlotUtils import plot_confusion_matrix
from src.utils.SparseUtils import gen_multiplicity_list, confusion_accumulate_1d, get_typed_list, retrieve_n_SE, \
    calculate_class_accuracy, gen_SE_mask, confusion_accumulate
import numpy as np

PID_MAP = {
    1: 0,  # ioni
    4: 1,  # recoil
    6: 2,  # ncapt + recoil
    256: 3,  # ingress
    258: 2,  # ingress + ncapt
    512: 4  # muon
}
PID_MAPPED_NAMES = {
    0: "Ionization",
    1: "Recoil",
    2: "Neutron Capture",
    3: "Ingress",
    4: "Muon"
}

def retrieve_class_names_PIDS():
    class_names = [val for key, val in PID_MAPPED_NAMES.items()]
    class_PIDS = [None]*len(class_names)
    for key, val in PID_MAP.items():
        if class_PIDS[val] is None:
            class_PIDS[val] = [key]
        else:
            class_PIDS[val].append(key)
    return class_names, class_PIDS,


class PIDEvaluator(SingleEndedEvaluator):

    def __init__(self, logger, calgroup=None, namespace=None, e_scale=None, additional_field_names=None, **kwargs):
        super(PIDEvaluator, self).__init__(logger, calgroup=calgroup, e_scale=e_scale, **kwargs)
        self.E_bounds = self.default_bins[0][0:2]
        self.mult_bounds = [0.5, 6.5]
        self.n_mult = 6
        self.n_E = self.default_bins[0][-1]
        self.metric_name = "accuracy"
        self.metric_unit = ""
        self.metrics = []
        self.scaling = 1.0
        self.n_classes = len(PID_MAPPED_NAMES.keys())
        self.additional_field_names = additional_field_names
        self.phys_index = None
        if self.additional_field_names is not None:
            self.phys_index = self.additional_field_names.index("phys")
        if "phys" in self.additional_field_names:
            self.use_phys = True
        else:
            self.use_phys = False
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
        self.metric_names = ["energy", "psd", "multiplicity", "z"]
        self.class_names = [PID_MAPPED_NAMES[i] for i in range(5)]
        units = ["MeVee", "", "", "mm"]
        metric_params = [self.default_bins[0], self.default_bins[5], [0.5, 6.5, 6], self.default_bins[4]]
        scales = [self.E_scale, 1.0, 1.0, self.z_scale]
        i = 0
        self.n_confusion = 10
        self.n_SE_max = 6
        for name, unit, scale in zip(self.metric_names, units, scales):
            self.metrics.append(MetricAggregator(name, *metric_params[i], self.class_names,
                                                 metric_name=self.metric_name, metric_unit=self.metric_unit,
                                                 scale_factor=self.scaling,
                                                 norm_factor=scale, parameter_unit=unit,
                                                 is_multiplicity=name == "multiplicity",
                                                 is_discreet=name == "multiplicity"))
            i += 1
            self.metric_pairs = MetricPairAggregator(self.metrics)

        self.results = {
            "confusion_energy": zeros((self.n_confusion + 1, self.n_classes, self.n_classes), dtype=np.int32),
            "confusion_SE": zeros((self.n_SE_max + 2, self.n_classes, self.n_classes), dtype=np.int32),
            "SE_confusion": zeros((self.n_classes, self.n_classes), dtype=np.int32)
        }

    def add(self, results, target, c, additional_fields=None):
        """
        @param results: tensor of dimension 2: (batch, )
        @param target: tensor of dimension 1 (batch, )
        @param c: tensor of dimension 2 (batch, 3) coordinates
        @param additional_fields: list of additional fields (tensors)
        """
        if additional_fields is None:
            return
        targ = target.detach().cpu().numpy()
        coo = c.detach().cpu().numpy()
        results = results.detach().cpu().numpy()
        accuracy = np.zeros((results.shape[0], ), np.int32)
        calculate_class_accuracy(results, targ, accuracy)
        if self.phys_index is not None:
            if isinstance(additional_fields[self.phys_index], list):
                phys = additional_fields[self.phys_index][0].detach().cpu().numpy()
            else:
                phys = additional_fields[self.phys_index].detach().cpu().numpy()
        else:
            phys = additional_fields[0].detach().cpu().numpy()
        mult = zeros((phys.shape[0],))
        gen_multiplicity_list(coo[:, 2], mult)
        parameters = stack((phys[:, self.E_index], phys[:, self.PSD_index], mult,
                            phys[:, self.z_index]), axis=1)
        parameters = swapaxes(parameters, 0, 1)
        # we only care about PID accuracy in SE cells, use mask to calculate
        se_mask = np.zeros((coo.shape[0],), dtype=bool)
        gen_SE_mask(coo, self.seg_status, se_mask)
        if self.metric_pairs is not None:
            for i in range(len(self.class_names)):
                ind_match = targ == i
                ind_match *= se_mask
                self.metric_pairs.add_normalized(accuracy[ind_match], parameters[:, ind_match], self.class_names[i])
        n_SE = np.zeros((results.shape[0],), dtype=np.int32)
        retrieve_n_SE(coo, self.seg_status, n_SE)
        confusion_accumulate(results[se_mask], targ[se_mask], self.results["SE_confusion"])
        confusion_accumulate_1d(results, targ, phys[0], self.results["confusion_energy"],
                                get_typed_list([0.0, self.n_confusion / self.E_scale]),
                                self.n_confusion)
        confusion_accumulate_1d(results, targ, n_SE, self.results["confusion_SE"],
                                get_typed_list([-0.5, self.n_SE_max + 0.5]),
                                self.n_SE_max + 1)

    def dump(self):
        self.metric_pairs.plot(self.logger)
        title = "SE confusion matrix"
        self.logger.experiment.add_figure("evaluation/SE_confusion_matrix",
                                          plot_confusion_matrix(self.results["SE_confusion"],
                                                                self.class_names,
                                                                normalize=True, title=title))
        title = "SE confusion matrix totals"
        self.logger.experiment.add_figure("evaluation/SE_confusion_matrix_totals",
                                          plot_confusion_matrix(self.results["SE_confusion"],
                                                                self.class_names,
                                                                normalize=False, title=title))
        for i in range(self.n_confusion):
            bin_width = 1.0
            title = "{0:.1f} - {1:.1f} MeV".format(i * bin_width, (i + 1) * bin_width)
            self.logger.experiment.add_figure("evaluation/confusion_matrix_energy{0}".format(i),
                                              plot_confusion_matrix(self.results["confusion_energy"][i],
                                                                    self.class_names,
                                                                    normalize=True, title=title))
            self.logger.experiment.add_figure("evaluation/confusion_matrix_energy{0}_totals".format(i),
                                              plot_confusion_matrix(self.results["confusion_energy"][i],
                                                                    self.class_names,
                                                                    normalize=False, title=title))
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
