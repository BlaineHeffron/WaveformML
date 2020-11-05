import numpy as np

from src.utils.PlotUtils import plot_n_hist1d, plot_n_hist2d, plot_n_contour
from src.utils.SparseUtils import metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list
from src.utils.util import safe_divide, get_bins, get_bin_midpoints


class MetricAggregator:

    def __init__(self, name, low, high, n_bins, class_names, is_discreet=False):
        self.name = name
        self.n_bins = n_bins
        self.bin_edges = get_bins(low, high, n_bins)
        self.class_names = class_names
        self.results_dict = {}
        self.is_discreet = is_discreet
        for nam in class_names:
            self.results_dict[nam] = (np.zeros((self.n_bins + 2,), dtype=np.float32),
                                      np.zeros((self.n_bins + 2,), dtype=np.int32))

    def add(self, results, metric, category_name):
        metric_accumulate_1d(results, metric, *self.results_dict[category_name],
                             get_typed_list([self.bin_edges[0], self.bin_edges[-1]]), self.n_bins)

    def bin_midpoints(self):
        return get_bin_midpoints(self.bin_edges[0], self.bin_edges[-1], self.n_bins),

    def plot(self, logger):
        logger.experiment.add_figure("evaluation/{}_precision".format(self.name),
                                     plot_n_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                   if self.is_discreet else self.bin_edges,
                                                   [safe_divide(
                                                       self.results_dict[self.class_names[i]][0][1:self.n_bins + 1],
                                                       self.results_dict[self.class_names[i]][1][1:self.n_bins + 1]
                                                   ) for i in range(len(self.class_names))],
                                                   self.class_names, self.name, "precision",
                                                   norm_to_bin_width=False, logy=False))
        logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                     plot_n_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                   if self.is_discreet else self.bin_edges,
                                                   [self.results_dict[self.class_names[i]][1][1:self.n_bins + 1]
                                                    for i in range(len(self.class_names))],
                                                   self.class_names, self.name, "total"))


class Metric2DAggregator:

    def __init__(self, metric1, metric2):
        self.results_dict = {}
        self.metric1 = metric1
        self.metric2 = metric2
        self.name = "{0}_{1}".format(metric1.name, metric2.name)
        for nam in metric1.class_names:
            self.results_dict[nam] = \
                (np.zeros((self.metric1.n_bins + 2, self.metric2.n_bins + 2), dtype=np.float32),
                 np.zeros((self.metric1.n_bins + 2, self.metric2.n_bins + 2), dtype=np.int32))

    def add(self, results, metric1, metric2, category_name):
        metric_accumulate_2d(results,
                             np.stack((metric1, metric2), axis=1),
                             *self.results_dict[category_name],
                             get_typed_list([self.metric1.bin_edges[0], self.metric1.bin_edges[1]]),
                             get_typed_list([self.metric2.bin_edges[0], self.metric2.bin_edges[1]]),
                             self.metric1.n_bins, self.metric2.n_bins)

    def plot(self, logger):
        logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                     plot_n_hist2d(self.metric1.bin_edges, self.metric2.bin_edges,
                                                   [self.results_dict[self.metric1.class_names[i]][
                                                        1][1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1] for i
                                                    in
                                                    range(len(self.metric1.class_names))],
                                                   self.metric1.class_names,
                                                   self.metric1.name, self.metric2.name))

        logger.experiment.add_figure("evaluation/{}_precision".format(self.name),
                                     plot_n_contour(self.metric1.bin_midpoints(), self.metric2.bin_midpoints(),
                                                    [safe_divide(self.results_dict[self.metric1.class_names[i]][0][
                                                                 1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1],
                                                                 self.results_dict[self.metric1.class_names[i]][1][
                                                                 1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1])
                                                     for i in
                                                     range(len(self.metric1.class_names))],
                                                    "Energy [MeV]", "PSD", self.metric1.class_names))


class MetricPairAggregator:
    def __init__(self, metric_list):
        self.metric_list = metric_list
        self.metric_pairs = {}
        for i in range(len(metric_list) - 1):
            for j in range(i + 1, len(metric_list)):
                self.metric_pairs["{0}_{1}".format(i, j)] = Metric2DAggregator(metric_list[i], metric_list[j])

    def add(self, results, metrics, category_name):
        for i in range(metrics.shape[0] - 1):
            self.metric_list[i].add(results, metrics[i,:], category_name)
            for j in range(i + 1, metrics.shape[0]):
                self.metric_pairs["{0}_{1}".format(i, j)].add(results, metrics[i,:], metrics[j,:], category_name)
        self.metric_list[-1].add(results, metrics[-1,:], category_name)

    def plot(self, logger):
        for m in self.metric_list:
            m.plot(logger)
        for key in self.metric_pairs.keys():
            self.metric_pairs[key].plot(logger)
