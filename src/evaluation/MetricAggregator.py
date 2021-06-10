import numpy as np

from src.utils.PlotUtils import plot_n_hist1d, plot_n_hist2d, plot_n_contour
from src.utils.SparseUtils import metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list
from src.utils.util import safe_divide, get_bins, get_bin_midpoints


class MetricAggregator:

    def __init__(self, name, low, high, n_bins, class_names, metric_name="precision", metric_unit="", is_discreet=False,
                 scale_factor=1.0, parameter_unit=""):
        """

        @param name: name of parameter the metric will be accumulated over
        @param low: lower bound of metric for histogramming
        @param high: upper bound of metric for histogramming
        @param n_bins: number of bins in histogram
        @param class_names: names of different classes for which the metric is aggregated over
        @param metric_name: name of the metric
        @param metric_unit: metric unit
        @param is_discreet: whether or not the parameter is discreet
        @param scale_factor: scale factor to scale results by at the end
        @param parameter_unit: unit of measurement for the parameter
        """
        self.name = name
        self.metric_name = metric_name
        self.metric_unit = metric_unit
        self.n_bins = n_bins
        self.bin_edges = get_bins(low, high, n_bins)
        self.class_names = class_names
        self.results_dict = {}
        self.is_discreet = is_discreet
        self.scale_factor = scale_factor
        self.parameter_unit = parameter_unit
        for nam in class_names:
            self.results_dict[nam] = (np.zeros((self.n_bins + 2,), dtype=np.double),
                                      np.zeros((self.n_bins + 2,), dtype=np.long))

    def add(self, results, parameter, category_name):
        metric_accumulate_1d(results, parameter, *self.results_dict[category_name],
                             get_typed_list([self.bin_edges[0], self.bin_edges[-1]]), self.n_bins)

    def add_normalized(self, results, parameter, category_name):
        """
        @param results: unscaled results containing the metric for each element
        @param parameter: parameter which the metric is aggregated over, assumes normalized from 0 to 1
        @param category_name: category of results being added
        """
        metric_accumulate_1d(results, parameter, *self.results_dict[category_name],
                             get_typed_list([0.0, 1.0]), self.n_bins)

    def bin_midpoints(self):
        return get_bin_midpoints(self.bin_edges[0], self.bin_edges[-1], self.n_bins)

    def retrieve_metric_label(self):
        if self.metric_unit:
            return "{0} [{1}]".format(self.metric_name, self.metric_unit)
        else:
            return self.metric_name

    def retrieve_parameter_label(self):
        if self.parameter_unit:
            return "{0} [{1}]".format(self.name, self.parameter_unit)
        else:
            return self.name

    def plot(self, logger):
        logger.experiment.add_figure("evaluation/{0}_{1}".format(self.name, self.metric_name),
                                     plot_n_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                   if self.is_discreet else self.bin_edges,
                                                   [safe_divide(
                                                       self.scale_factor*self.results_dict[self.class_names[i]][0][1:self.n_bins + 1],
                                                       self.results_dict[self.class_names[i]][1][1:self.n_bins + 1]
                                                   ) for i in range(len(self.class_names))],
                                                   self.class_names, self.retrieve_parameter_label(), self.retrieve_metric_label(),
                                                   norm_to_bin_width=False, logy=False))
        logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                     plot_n_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                   if self.is_discreet else self.bin_edges,
                                                   [self.results_dict[self.class_names[i]][1][1:self.n_bins + 1]
                                                    for i in range(len(self.class_names))],
                                                   self.class_names, self.retrieve_parameter_label(), "total"))


class Metric2DAggregator:

    def __init__(self, metric1: MetricAggregator, metric2: MetricAggregator):
        self.results_dict = {}
        if metric1.scale_factor != metric2.scale_factor:
            raise ValueError("Adding two metrics aggregators with different scale factors! {0} : {1}, {2} : {3}. "
                             "Scale factors must be same for 2d metric aggregator".format(metric1.name,
                                                                                          metric1.scale_factor,
                                                                                          metric2.name,
                                                                                          metric2.scale_factor))
        self.metric1 = metric1
        self.metric2 = metric2
        self.name = "{0}_{1}".format(metric1.name, metric2.name)
        for nam in metric1.class_names:
            self.results_dict[nam] = \
                (np.zeros((self.metric1.n_bins + 2, self.metric2.n_bins + 2), dtype=np.double),
                 np.zeros((self.metric1.n_bins + 2, self.metric2.n_bins + 2), dtype=np.long))

    def add(self, results, parameter1, parameter2, category_name):
        metric_accumulate_2d(results,
                             np.stack((parameter1, parameter2), axis=1),
                             *self.results_dict[category_name],
                             get_typed_list([self.metric1.bin_edges[0], self.metric1.bin_edges[-1]]),
                             get_typed_list([self.metric2.bin_edges[0], self.metric2.bin_edges[-1]]),
                             self.metric1.n_bins, self.metric2.n_bins)

    def add_normalized(self, results, parameter1, parameter2, category_name):
        metric_accumulate_2d(results,
                             np.stack((parameter1, parameter2), axis=1),
                             *self.results_dict[category_name],
                             get_typed_list([0.0, 1.0]),
                             get_typed_list([0.0, 1.0]),
                             self.metric1.n_bins, self.metric2.n_bins)

    def plot(self, logger):
        logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                     plot_n_hist2d(self.metric1.bin_edges, self.metric2.bin_edges,
                                                   [self.results_dict[self.metric1.class_names[i]][
                                                        1][1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1]
                                                    for i in
                                                    range(len(self.metric1.class_names))],
                                                   self.metric1.class_names,
                                                   self.metric1.retrieve_parameter_label(), self.metric2.retrieve_parameter_label()))

        logger.experiment.add_figure("evaluation/{}_precision".format(self.name),
                                     plot_n_contour(self.metric1.bin_midpoints(), self.metric2.bin_midpoints(),
                                                    [safe_divide(self.metric1.scale_factor*self.results_dict[self.metric1.class_names[i]][0][
                                                                 1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1],
                                                                 self.results_dict[self.metric1.class_names[i]][1][
                                                                 1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1])
                                                     for i in
                                                     range(len(self.metric1.class_names))],
                                                    self.metric1.retrieve_parameter_label(),
                                                    self.metric2.retrieve_parameter_label(), self.metric1.class_names))


class MetricPairAggregator:
    def __init__(self, metric_list):
        self.metric_list = metric_list
        self.metric_pairs = {}
        for i in range(len(metric_list) - 1):
            for j in range(i + 1, len(metric_list)):
                self.metric_pairs["{0}_{1}".format(i, j)] = Metric2DAggregator(metric_list[i], metric_list[j])

    def add(self, results, parameters, category_name):
        for i in range(parameters.shape[0] - 1):
            self.metric_list[i].add(results, parameters[i, :], category_name)
            for j in range(i + 1, parameters.shape[0]):
                self.metric_pairs["{0}_{1}".format(i, j)].add(results, parameters[i, :], parameters[j, :],
                                                              category_name)
        self.metric_list[-1].add(results, parameters[-1, :], category_name)

    def add_normalized(self, results, parameters, category_name):
        """
        @param results: unscaled results
        @param parameters: parameters to be added, assumed normalized from 0 to 1
        @param category_name: category to be added to
        """
        for i in range(parameters.shape[0] - 1):
            self.metric_list[i].add_normalized(results, parameters[i, :], category_name)
            for j in range(i + 1, parameters.shape[0]):
                self.metric_pairs["{0}_{1}".format(i, j)].add_normalized(results, parameters[i, :], parameters[j, :],
                                                                         category_name)
        self.metric_list[-1].add_normalized(results, parameters[-1, :], category_name)

    def plot(self, logger):
        for m in self.metric_list:
            m.plot(logger)
        for key in self.metric_pairs.keys():
            self.metric_pairs[key].plot(logger)
