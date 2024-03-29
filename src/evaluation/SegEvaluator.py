import numpy as np

from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator
from src.evaluation.PIDEvaluator import PID_MAPPED_NAMES, PID_MAP, retrieve_class_names_PIDS
from numpy import zeros, stack, swapaxes

from src.utils.SparseUtils import gen_multiplicity_list, gen_SE_mask
from src.utils.StatsUtils import ErrorAggregator


class SegEvaluator(SingleEndedEvaluator):

    def __init__(self, logger, calgroup=None, namespace=None, e_scale=None, additional_field_names=None, **kwargs):
        super(SegEvaluator, self).__init__(logger, calgroup=calgroup, e_scale=e_scale, **kwargs)
        self.E_bounds = self.default_bins[0][0:2]
        self.mult_bounds = [0.5, 6.5]
        self.n_mult = 6
        self.n_E = self.default_bins[0][-1]
        if "target_index" in kwargs.keys():
            self.target_index = kwargs["target_index"]
        else:
            self.target_index = 4  # default to z prediction
        self.metric_name = "mean absolute error"
        self.metric_unit = self.phys_units[self.target_index]
        self.metrics = []
        self.error_aggregator = None
        self.scaling = self.scale_factor(self.target_index)
        self.PID_index = None
        self.additional_field_names = additional_field_names
        if self.additional_field_names is not None:
            if "PID" in self.additional_field_names:
                self.PID_index = self.additional_field_names.index("PID")
                self.has_PID = True
        if self.has_PID:
            self.class_names, self.class_PIDs = retrieve_class_names_PIDS()
        else:
            self.class_names = ["all"]
            self.class_PIDs = None
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
        units = ["MeVee", "", "", "mm"]
        metric_params = [self.default_bins[0], self.default_bins[5], [0.5, 6.5, 6], self.default_bins[4]]
        scales = [self.E_scale, 1.0, 1.0, self.z_scale]
        i = 0
        for name, unit, scale in zip(self.metric_names, units, scales):
            self.metrics.append(MetricAggregator(name, *metric_params[i], self.class_names,
                                                 metric_name=self.metric_name, metric_unit=self.metric_unit,
                                                 scale_factor=self.scaling,
                                                 norm_factor=scale, parameter_unit=unit,
                                                 is_multiplicity=name == "multiplicity",
                                                 is_discreet=name == "multiplicity"))
            i += 1
        self.metric_pairs = MetricPairAggregator(self.metrics)
        truth_name = "calibrated {0}".format(self.phys_names[self.target_index])
        pred_name = "predicted {0}".format(self.phys_names[self.target_index])
        self.error_aggregator = ErrorAggregator(self.phys_names[self.target_index],
                                                *self.default_bins[self.target_index], self.class_names,
                                                metric_name=self.metric_name, metric_unit=self.metric_unit,
                                                scale_factor=self.scaling, truth_name=truth_name, pred_name=pred_name)

    def add(self, results, target, c, additional_fields=None):
        """
        @param results: numpy array 1 dim (batch, ) metric values
        @param target: numpy array 2 dim (batch, n) of parameter values (n phys quantities)
        @param c: torch tensor 2 dim (batch, coord)
        @param additional_fields: list of torch tensors of additional fields (same shape as results)
        """
        PID = additional_fields[self.PID_index].detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        coo = c.detach().cpu().numpy()
        results = results.detach().cpu().numpy()
        mae = np.absolute(results - target[:, self.target_index])
        mult = zeros((target.shape[0],))
        gen_multiplicity_list(coo[:, 2], mult)
        parameters = stack((target[:, self.E_index], target[:, self.PSD_index], mult,
                            target[:, self.z_index]), axis=1)
        parameters = swapaxes(parameters, 0, 1)
        # only care about SE segment predictions
        se_mask = zeros((coo.shape[0],), dtype=bool)
        gen_SE_mask(coo, self.seg_status, se_mask)
        if self.class_PIDs is not None:
            for i in range(len(self.class_names)):
                for pid in self.class_PIDs[i]:
                    ind_match = PID == pid
                    ind_match *= se_mask
                    if results[ind_match].shape[0] > 0:
                        self.metric_pairs.add_normalized(mae[ind_match], parameters[:, ind_match], self.class_names[i])
                        self.error_aggregator.add_norm(results[ind_match], target[ind_match, self.target_index],
                                                       self.class_names[i])
        else:
            self.metric_pairs.add_normalized(mae, parameters, self.class_names[0])
            self.error_aggregator.add_norm(results, target[:, self.target_index], self.class_names[0])

    def dump(self):
        self.metric_pairs.plot(self.logger)
        self.error_aggregator.plot(self.logger)
