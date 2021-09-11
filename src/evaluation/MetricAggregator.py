from typing import List

import numpy as np

from src.utils.PlotUtils import plot_n_hist1d, plot_n_hist2d,  plot_hist1d, plot_hist2d
from src.utils.SparseUtils import metric_accumulate_2d, metric_accumulate_1d, \
    get_typed_list, metric_accumulate_dense_1d_with_categories, metric_accumulate_dense_2d_with_categories
from src.utils.util import safe_divide, get_bins, get_bin_midpoints


class MetricAggregator:

    def __init__(self, name, low, high, n_bins, class_names, metric_name="precision", metric_unit="", is_discreet=False,
                 scale_factor=1.0, parameter_unit="", norm_factor=None, ignore_val=0, is_multiplicity=False):
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
        @param norm_factor: normalization used for metric when using add_normalized
        @param ignore_val: for dense processing, ignore when results has this value at the index
        @param is_multiplicity: whether or not this parameter is multiplicity, used for dense processing
        """
        self.name = name
        self.metric_name = metric_name
        self.metric_unit = metric_unit
        self.n_bins = n_bins
        self.bin_edges = get_bins(low, high, n_bins)
        self.class_names = class_names
        self.is_discreet = is_discreet
        self.scale_factor = scale_factor
        self.parameter_unit = parameter_unit
        self.norm_factor = norm_factor
        self.num_classes = len(self.class_names)
        # results are indexed by class in order of class_names
        self.results_val = np.zeros((self.num_classes, self.n_bins + 2), dtype=np.double)
        self.results_num = np.zeros((self.num_classes, self.n_bins + 2), dtype=np.long)
        self.ignore_val = ignore_val
        self.is_multiplicity = is_multiplicity

    def add(self, results, parameter, category_name):
        class_ind = self.class_names.index(category_name)
        metric_accumulate_1d(results, parameter, self.results_val[class_ind], self.results_num[class_ind],
                             get_typed_list([self.bin_edges[0], self.bin_edges[-1]]), self.n_bins)

    def add_normalized(self, results, parameter, category_name):
        """
        @param results: unscaled results containing the metric for each element
        @param parameter: parameter which the metric is aggregated over, assumes normalized from 0 to 1
        @param category_name: category of results being added
        """
        if self.norm_factor is None:
            list1 = [0.0, 1.0]
        else:
            if self.bin_edges[0] < 0:
                list1 = [self.bin_edges[0] / self.norm_factor + 0.5, self.bin_edges[-1] / self.norm_factor + 0.5]
            else:
                list1 = [self.bin_edges[0] / self.norm_factor, self.bin_edges[-1] / self.norm_factor]
        class_ind = self.class_names.index(category_name)
        metric_accumulate_1d(results, parameter, self.results_val[class_ind], self.results_num[class_ind],
                             get_typed_list(list1), self.n_bins)

    def add_dense_normalized_with_categories(self, results, parameter, categories):
        """
        @param results: numpy array of dimension 3 (batch, X, Y) containing unscaled results containing the metric
                        for each element. 0 for values to be skipped
        @param parameter: numpy array of dimension 3 (batch, X, Y) parameter which the metric is aggregated over,
                          assumes normalized from 0 to 1
        @param categories: numpy array of dimension 3 (batch, X, Y) containing category indices
        """
        if self.norm_factor is None:
            list1 = [0.0, 1.0]
        else:
            if self.bin_edges[0] < 0:
                list1 = [self.bin_edges[0] / self.norm_factor + 0.5, self.bin_edges[-1] / self.norm_factor + 0.5]
            else:
                list1 = [self.bin_edges[0] / self.norm_factor, self.bin_edges[-1] / self.norm_factor]
        metric_accumulate_dense_1d_with_categories(results, parameter, self.results_val, self.results_num, categories,
                                                   get_typed_list(list1), self.n_bins, ignore_val=self.ignore_val,
                                                   use_multiplicity=self.is_multiplicity)

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
        if len(self.class_names) == 1:
            if np.sum(self.results_num[0]) < 200:
                return
            logger.experiment.add_figure("evaluation/{0}_{1}".format(self.name, self.metric_name),
                                         plot_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                     if self.is_discreet else self.bin_edges,
                                                     safe_divide(
                                                         self.scale_factor * self.results_val[0, 1:self.n_bins + 1],
                                                         self.results_num[0, 1:self.n_bins + 1]
                                                     ),
                                                     self.class_names[0], self.retrieve_parameter_label(),
                                                     self.retrieve_metric_label(),
                                                     norm_to_bin_width=False, logy=False))
            logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                         plot_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                     if self.is_discreet else self.bin_edges,
                                                     self.results_num[0, 1:self.n_bins + 1],
                                                     self.class_names[0], self.retrieve_parameter_label(), "total"))
        else:
            inds_to_plot, class_names_to_plot = self.retrieve_inds_to_plot()
            if len(class_names_to_plot) > 1:
                logger.experiment.add_figure("evaluation/{0}_{1}".format(self.name, self.metric_name),
                                             plot_n_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                           if self.is_discreet else self.bin_edges,
                                                           [safe_divide(
                                                               self.scale_factor * self.results_val[i,
                                                                                   1:self.n_bins + 1],
                                                               self.results_num[i, 1:self.n_bins + 1]
                                                           ) for i in inds_to_plot],
                                                           class_names_to_plot, self.retrieve_parameter_label(),
                                                           self.retrieve_metric_label(),
                                                           norm_to_bin_width=False, logy=False))
                logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                             plot_n_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                           if self.is_discreet else self.bin_edges,
                                                           [self.results_num[i, 1:self.n_bins + 1]
                                                            for i in inds_to_plot],
                                                           class_names_to_plot, self.retrieve_parameter_label(),
                                                           "total"))
            elif len(class_names_to_plot) == 1:
                logger.experiment.add_figure("evaluation/{0}_{1}".format(self.name, self.metric_name),
                                             plot_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                         if self.is_discreet else self.bin_edges,
                                                         safe_divide(
                                                             self.scale_factor * self.results_val[inds_to_plot[0],
                                                                                 1:self.n_bins + 1],
                                                             self.results_num[inds_to_plot[0], 1:self.n_bins + 1]
                                                         ),
                                                         self.class_names[inds_to_plot[0]],
                                                         self.retrieve_parameter_label(),
                                                         self.retrieve_metric_label(),
                                                         norm_to_bin_width=False, logy=False))
                logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                             plot_hist1d(get_bins(0.5, self.n_bins + 0.5, self.n_bins)
                                                         if self.is_discreet else self.bin_edges,
                                                         self.results_num[inds_to_plot[0], 1:self.n_bins + 1],
                                                         self.class_names[inds_to_plot[0]],
                                                         self.retrieve_parameter_label(), "total"))

    def retrieve_inds_to_plot(self):
        to_plot = []
        classes_to_plot = []
        for i in range(len(self.class_names)):
            if np.sum(self.results_num[i]) > 20:
                to_plot.append(i)
                classes_to_plot.append(self.class_names[i])
        return to_plot, classes_to_plot


class Metric2DAggregator:

    def __init__(self, metric1: MetricAggregator, metric2: MetricAggregator):
        if metric1.scale_factor != metric2.scale_factor:
            raise ValueError("Adding two metrics aggregators with different scale factors! {0} : {1}, {2} : {3}. "
                             "Scale factors must be same for 2d metric aggregator".format(metric1.name,
                                                                                          metric1.scale_factor,
                                                                                          metric2.name,
                                                                                          metric2.scale_factor))
        self.metric1 = metric1
        self.metric2 = metric2
        self.multiplicity_index = -1
        if self.metric1.is_multiplicity:
            self.multiplicity_index = 0
        elif self.metric2.is_multiplicity:
            self.multiplicity_index = 1
        self.name = "{0}_{1}".format(metric1.name, metric2.name)
        self.results_val = np.zeros((self.metric1.num_classes, self.metric1.n_bins + 2, self.metric2.n_bins + 2),
                                    dtype=np.double)
        self.results_num = np.zeros((self.metric1.num_classes, self.metric1.n_bins + 2, self.metric2.n_bins + 2),
                                    dtype=np.long)

    def add(self, results, parameter1, parameter2, category_name):
        metric_accumulate_2d(results,
                             np.stack((parameter1, parameter2), axis=1),
                             self.results_val[self.metric1.class_names.index(category_name)],
                             self.results_num[self.metric1.class_names.index(category_name)],
                             get_typed_list([self.metric1.bin_edges[0], self.metric1.bin_edges[-1]]),
                             get_typed_list([self.metric2.bin_edges[0], self.metric2.bin_edges[-1]]),
                             self.metric1.n_bins, self.metric2.n_bins)

    def get_ranges(self):
        if self.metric1.norm_factor is None:
            list1 = [0.0, 1.0]
        else:
            if self.metric1.bin_edges[0] < 0:
                list1 = [self.metric1.bin_edges[0] / self.metric1.norm_factor + 0.5,
                         self.metric1.bin_edges[-1] / self.metric1.norm_factor + 0.5]
            else:
                list1 = [self.metric1.bin_edges[0] / self.metric1.norm_factor,
                         self.metric1.bin_edges[-1] / self.metric1.norm_factor]
        if self.metric2.norm_factor is None:
            list2 = [0.0, 1.0]
        else:
            if self.metric2.bin_edges[0] < 0:
                list2 = [self.metric2.bin_edges[0] / self.metric2.norm_factor + 0.5,
                         self.metric2.bin_edges[-1] / self.metric2.norm_factor + 0.5]
            else:
                list2 = [self.metric2.bin_edges[0] / self.metric2.norm_factor,
                         self.metric2.bin_edges[-1] / self.metric2.norm_factor]
        return list1, list2

    def add_normalized(self, results, parameter1, parameter2, category_name):
        list1, list2 = self.get_ranges()
        metric_accumulate_2d(results,
                             np.stack((parameter1, parameter2), axis=1),
                             self.results_val[self.metric1.class_names.index(category_name)],
                             self.results_num[self.metric1.class_names.index(category_name)],
                             get_typed_list(list1),
                             get_typed_list(list2),
                             self.metric1.n_bins, self.metric2.n_bins)

    def add_dense_normalized_with_categories(self, results, parameter1, parameter2, categories):
        """
        @param results: numpy array of dimension 3 (batch, X, Y) containing unscaled results containing the metric
                        for each element. 0 for values to be skipped
        @param parameter1: numpy array of dimension 3 (batch, X, Y) parameter which the metric is aggregated over,
                          assumes normalized from 0 to 1
        @param parameter2: numpy array of dimension 3 (batch, X, Y) parameter which the metric is aggregated over,
                          assumes normalized from 0 to 1
        @param categories: numpy array of dimension 3 (batch, X, Y) containing category indices
        """
        list1, list2 = self.get_ranges()
        metric_accumulate_dense_2d_with_categories(results,
                                                   np.stack((parameter1, parameter2), axis=1),
                                                   self.results_val, self.results_num, categories,
                                                   get_typed_list(list1), get_typed_list(list2),
                                                   self.metric1.n_bins, self.metric2.n_bins,
                                                   ignore_val=self.metric1.ignore_val,
                                                   multiplicity_index=self.multiplicity_index)

    def plot(self, logger):
        if len(self.metric1.class_names) == 1:
            if np.sum(self.results_num[0, 1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1]) < 20:
                return
            logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                         plot_hist2d(self.metric1.bin_edges, self.metric2.bin_edges,
                                                     self.results_num[0, 1:self.metric1.n_bins + 1,
                                                     1:self.metric2.n_bins + 1],
                                                     self.metric1.class_names[0],
                                                     self.metric1.retrieve_parameter_label(),
                                                     self.metric2.retrieve_parameter_label(),
                                                     zlabel=self.metric1.retrieve_parameter_label()))

            logger.experiment.add_figure("evaluation/{}_precision".format(self.name),
                                         plot_hist2d(self.metric1.bin_midpoints(), self.metric2.bin_midpoints(),
                                                     safe_divide(self.metric1.scale_factor * self.results_val[0,
                                                                                             1:self.metric1.n_bins + 1,
                                                                                             1:self.metric2.n_bins + 1],
                                                                 self.results_num[0, 1:self.metric1.n_bins + 1,
                                                                 1:self.metric2.n_bins + 1]),
                                                     self.metric1.class_names[0],
                                                     self.metric1.retrieve_parameter_label(),
                                                     self.metric2.retrieve_parameter_label(),
                                                     zlabel=self.metric1.retrieve_parameter_label(), logz=False,
                                                     norm_to_bin_width=False))

        else:
            inds_to_plot, class_names_to_plot = self.retrieve_inds_to_plot()
            if len(class_names_to_plot) > 1:
                logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                             plot_n_hist2d(self.metric1.bin_edges, self.metric2.bin_edges,
                                                           [self.results_num[i, 1:self.metric1.n_bins + 1,
                                                            1:self.metric2.n_bins + 1] for i in
                                                            inds_to_plot],
                                                           class_names_to_plot,
                                                           self.metric1.retrieve_parameter_label(),
                                                           self.metric2.retrieve_parameter_label(), logz=False,
                                                           norm_to_bin_width=False))

                logger.experiment.add_figure("evaluation/{}_precision".format(self.name),
                                             plot_n_hist2d(self.metric1.bin_midpoints(), self.metric2.bin_midpoints(),
                                                           [safe_divide(self.metric1.scale_factor * self.results_val[i,
                                                                                                    1:self.metric1.n_bins + 1,
                                                                                                    1:self.metric2.n_bins + 1],
                                                                        self.results_num[i,
                                                                        1:self.metric1.n_bins + 1,
                                                                        1:self.metric2.n_bins + 1])
                                                            for i in inds_to_plot],
                                                           class_names_to_plot,
                                                           self.metric1.retrieve_parameter_label(),
                                                           self.metric2.retrieve_parameter_label(), logz=False,
                                                           norm_to_bin_width=False))
            elif len(class_names_to_plot) == 1:
                logger.experiment.add_figure("evaluation/{}_classes".format(self.name),
                                             plot_hist2d(self.metric1.bin_edges, self.metric2.bin_edges,
                                                         self.results_num[inds_to_plot[0], 1:self.metric1.n_bins + 1,
                                                         1:self.metric2.n_bins + 1],
                                                         self.metric1.class_names[inds_to_plot[0]],
                                                         self.metric1.retrieve_parameter_label(),
                                                         self.metric2.retrieve_parameter_label(),
                                                         zlabel=self.metric1.retrieve_parameter_label()))

                logger.experiment.add_figure("evaluation/{}_precision".format(self.name),
                                             plot_hist2d(self.metric1.bin_midpoints(), self.metric2.bin_midpoints(),
                                                          safe_divide(self.metric1.scale_factor * self.results_val[0,
                                                                                                  1:self.metric1.n_bins + 1,
                                                                                                  1:self.metric2.n_bins + 1],
                                                                      self.results_num[inds_to_plot[0],
                                                                      1:self.metric1.n_bins + 1,
                                                                      1:self.metric2.n_bins + 1]),
                                                         self.metric1.class_names[inds_to_plot[0]],
                                                          self.metric1.retrieve_parameter_label(),
                                                          self.metric2.retrieve_parameter_label(),
                                                         zlabel=self.metric1.retrieve_parameter_label(),
                                                         logz=False, norm_to_bin_width=False))

    def retrieve_inds_to_plot(self):
        inds = []
        names = []
        for i in range(self.metric1.num_classes):
            if np.sum(self.results_num[i, 1:self.metric1.n_bins + 1, 1:self.metric2.n_bins + 1]) > 20:
                inds.append(i)
                names.append(self.metric1.class_names[i])
        return inds, names


class MetricPairAggregator:
    def __init__(self, metric_list: List[MetricAggregator]):
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

    def add_dense_normalized_with_categories(self, results, parameters, parameter_names, categories):
        """
        @param results: 3 dimensional input, batch x X x Y
        @param parameters: 4 dimensional input, batch x parameter dimension x X x Y
        @param parameter_names: list of names of the parameters, used to match with metric index
        @param categories: 4 dimensional input, batch x X x Y
        """
        for i in range(len(parameter_names) - 1):
            ind1 = self.metric_index_by_name(parameter_names[i])
            self.metric_list[ind1].add_dense_normalized_with_categories(results, parameters[:, i], categories)
            for j in range(i + 1, len(parameter_names)):
                ind2 = self.metric_index_by_name(parameter_names[j])
                if ind2 < ind1:
                    name = "{0}_{1}".format(ind2, ind1)
                    self.metric_pairs[name].add_dense_normalized_with_categories(results, parameters[:, j],
                                                                                 parameters[:, i],
                                                                                 categories)
                else:
                    name = "{0}_{1}".format(ind1, ind2)
                    self.metric_pairs[name].add_dense_normalized_with_categories(results, parameters[:, i],
                                                                                 parameters[:, j],
                                                                                 categories)
            ind = self.metric_index_by_name(parameter_names[-1])
            self.metric_list[ind].add_dense_normalized_with_categories(results, parameters[:, -1], categories)

    def metric_index_by_name(self, name):
        for i, m in enumerate(self.metric_list):
            if m.name == name:
                return i
        raise ValueError("no name {} in metric list".format(name))

    def plot(self, logger):
        for m in self.metric_list:
            m.plot(logger)
        for key in self.metric_pairs.keys():
            self.metric_pairs[key].plot(logger)
