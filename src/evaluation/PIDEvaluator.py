from numpy import stack, zeros, swapaxes
from torch import ones_like

from src.evaluation.EnergyEvaluator import EnergyEvaluatorPhys
from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.evaluation.WaveformEvaluator import WaveformEvaluator
from src.utils.SparseUtils import gen_multiplicity_list

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


class PIDEvaluator(WaveformEvaluator):

    def __init__(self, logger, calgroup=None, namespace=None, e_scale=None, additional_field_names=None, **kwargs):
        WaveformEvaluator.__init__(self, logger, calgroup=calgroup, e_scale=e_scale, **kwargs)
        if calgroup is not None:
            self.EnergyEvaluator = EnergyEvaluatorPhys(logger, calgroup=None, e_scale=e_scale, namespace=namespace)
        self.E_bounds = self.default_bins[0][0:2]
        self.mult_bounds = [0.5, 6.5]
        self.n_mult = 6
        self.n_E = self.default_bins[0][-1]
        self.metric_name = "accuracy"
        self.metric_unit = ""
        self.metrics = []
        self.scaling = 1.0
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
        @param results: tensor of dimension 2: (batch, n classes)
        @param target: tensor of dimension 1 (batch, )
        @param c: tensor of dimension 2 (batch, 3) coordinates
        @param additional_fields: list of additional fields (tensors)
        """
        if additional_fields is None:
            return
        targ = target.detach().cpu().numpy()
        coo = c.detach().cpu().numpy()
        results = results.detach().cpu().numpy()
        if self.phys_index is not None:
            phys = additional_fields[self.phys_index].detach().cpu().numpy()
        else:
            phys = additional_fields[0].detach().cpu().numpy()
        mult = zeros((phys.shape[0],))
        gen_multiplicity_list(coo[:, 2], mult)
        parameters = stack((phys[:, self.E_index], phys[:, self.PSD_index], mult,
                            phys[:, self.z_index]), axis=1)
        parameters = swapaxes(parameters, 0, 1)
        if self.metric_pairs is not None:
            for i in range(len(self.class_names)):
                ind_match = targ == i
                self.metric_pairs.add_normalized(results[ind_match], parameters[:, ind_match], self.class_names[i])

    def dump(self):
        self.metric_pairs.plot(self.logger)
