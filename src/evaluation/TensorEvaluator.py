"""
For non sparse formatted tensors
"""
from src.evaluation.AD1Evaluator import AD1Evaluator
from src.evaluation.MetricAggregator import MetricAggregator, MetricPairAggregator
from src.utils.StatsUtils import StatsAggregator


class TensorEvaluator(AD1Evaluator, StatsAggregator):
    def __init__(self, logger, calgroup=None, e_scale=None, target_has_phys=False, target_index=None, metric_name=None,
                 metric_unit=None, class_names=None, bin_overrides=None):
        AD1Evaluator.__init__(self, calgroup=calgroup, e_scale=e_scale)
        StatsAggregator.__init__(self, logger=logger)
        self.target_has_phys = target_has_phys
        self.metric_name = metric_name
        self.metric_unit = metric_unit
        self.target_index = target_index
        if bin_overrides is not None:
            self.override_default_bins(bin_overrides)
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = ["Single"]
        self.metrics = []
        if self.target_index is not None:
            if self.metric_name is None:
                self.metric_name = "mean absolute error"
            if self.metric_unit is None:
                self.metric_unit = self.phys_units[self.target_index]
                if "squared" in self.metric_name:
                    self.metric_unit += "^2"
        self._init_results()

    def _init_results(self):
        i = 0
        if self.target_has_phys:
            if self.target_index is None:
                raise RuntimeError("target is tensor of phys quantities, must pass the target index to the evaluator")
            for name in self.phys_names:
                self.metrics.append(MetricAggregator(name, *self.default_bins[i], self.class_names,
                                                     metric_name=self.metric_name, metric_unit=self.metric_unit,
                                                     scale_factor=self.scale_factor(self.target_index),
                                                     parameter_unit=self.phys_units[i], norm_factor=self.scale_factor(i)))
                i += 1
            self.metric_pairs = MetricPairAggregator(self.metrics)
        else:
            if self.target_index is not None:
                name = self.phys_names[self.target_index]
                bins = self.default_bins[self.target_index]
                unit = self.phys_units[self.target_index]
                scale_factor = self.scale_factor(self.target_index)
            else:
                if self.metric_name is None:
                    name = "unknown"
                else:
                    name = self.metric_name
                if self.metric_unit is None:
                    unit = ""
                else:
                    unit = self.metric_unit
                bins = [0., 1., 40]
                scale_factor = 1.
            self.metrics.append(MetricAggregator(name, *bins, self.class_names,
                                                 metric_name=self.metric_name, metric_unit=self.metric_unit,
                                                 scale_factor=scale_factor, parameter_unit=unit))

    def add(self, target, results):
        if target.dim() >= 2:
            target = target.permute(1, 0)
        target = target.detach().cpu().numpy()
        results = results.detach().cpu().numpy()
        if self.target_has_phys:
            self.metric_pairs.add_normalized(results, target, self.class_names[0])
        else:
            self.metrics[-1].add_normalized(results, target, self.class_names[0])

    def dump(self):
        if hasattr(self,"metric_pairs"):
            self.metric_pairs.plot(self.logger)
        else:
            self.metrics[-1].plot(self.logger)
