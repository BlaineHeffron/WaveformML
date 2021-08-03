from numpy import zeros_like, stack
from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.evaluation.SingleEndedEvaluator import SingleEndedEvaluator


def PID_class_map(PID):
    map = {1: "ionization",
           2: "neutron capture",
           3: "neutron capture + ionization",
           4: "recoil",
           5: "ionization + recoil",
           6: "neutron capture + recoil",
           7: "neutron capture + recoil + ionization"}
    if PID not in map:
        return "Unknown"
    else:
        return map[PID]


class RealDataEvaluator(SingleEndedEvaluator):
    def __init__(self, logger, calgroup=None, e_scale=None, additional_field_names=None, metric_name=None,
                 metric_unit=None, target_has_phys=False, scaling=1.0, bin_overrides=None):
        super(RealDataEvaluator, self).__init__(logger, calgroup=calgroup, e_scale=e_scale)
        self.additional_field_names = additional_field_names
        self.has_PID = False
        self.PID_index = None
        self.metric_name = metric_name
        self.metric_unit = metric_unit
        self.class_names = None
        self.target_has_phys = target_has_phys
        if self.additional_field_names is not None:
            if "PID" in self.additional_field_names:
                self.PID_index = self.additional_field_names.index("PID")
                self.has_PID = True
        self.metrics = []
        self.metric_names = []
        self.metric_pairs = None
        if bin_overrides is not None:
            self.override_default_bins(bin_overrides)
        self.scaling = scaling
        if self.has_PID:
            self.metric_names = ["energy", "psd", "multiplicity", "z"]
            self.class_names = [PID_class_map(i) for i in range(1, 8)]
            units = ["MeVee", "", "", "mm"]
            metric_params = [self.default_bins[0], self.default_bins[5], [0.5, 6.5, 10], self.default_bins[4]]
            scales = [self.E_scale, 1.0, 1.0, self.z_scale]
            i = 0
            for name, unit, scale in zip(self.metric_names, units, scales):
                self.metrics.append(MetricAggregator(name, *metric_params[i], self.class_names,
                                                     metric_name=self.metric_name, metric_unit=self.metric_unit,
                                                     scale_factor=self.scaling,
                                                     norm_factor=scale, parameter_unit=unit,
                                                     is_multiplicity=name == "multiplicity"))
                i += 1
            self.metric_pairs = MetricPairAggregator(self.metrics)

    def add(self, results, target, c, additional_fields=None):
        """
        @param results: numpy array 3 dim (batch, x, y), metric values
        @param target: numpy array 4 dim (batch, n params,  x, y) of parameter values
        @param c: torch tensor 2 dim (batch, coord)
        @param additional_fields: list of torch tensors of additional fields (same shape as results)
        """
        if not self.has_PID:
            return
        class_indices = additional_fields[self.PID_index] - 1  # map 1 to 0, 2 to 1, etc
        class_indices = self.get_dense_matrix(class_indices, c).squeeze(1)
        parameters = stack((target[:, self.E_index, :], target[:, self.PSD_index], zeros_like(target[:, self.E_index, :]),
                          target[:, self.z_index]), axis=1)
        if self.metric_pairs is not None:
            self.metric_pairs.add_dense_normalized_with_categories(results, parameters, self.metric_names, class_indices)

    def dump(self):
        if self.metric_pairs is not None:
            self.metric_pairs.plot(self.logger)
